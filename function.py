import io
import json
import re
import tempfile
import os
import csv
import subprocess
import ijson
from urllib.parse import urlparse

from google.cloud import storage
from PyPDF2 import PdfMerger

from google import genai
from google.genai import types

from config import *
from total import TOTAL_SYSTEM_INSTRUCTION
from container import CONTAINER_SYSTEM_INSTRUCTION
from detail import build_detail_prompt
from row import ROW_SYSTEM_INSTRUCTION


BATCH_SIZE = 5

storage_client = storage.Client()

genai_client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
)

# ==============================
# JSON SAFE PARSER
# (FIX: versi kamu salah karena startswith("") selalu True dan tidak bisa handle prefix)
# ==============================
def _parse_json_safe(raw_text):
    if not raw_text:
        raise Exception("Gemini returned empty response")

    s = raw_text.strip()

    # strip code fences kalau ada
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        s = s.strip()

    # helper: kalau json.loads menghasilkan string JSON, decode lagi
    def _maybe_double_decode(obj):
        if isinstance(obj, str):
            t = obj.strip()
            if t and t[0] in "{[":
                try:
                    return json.loads(t)
                except:
                    return obj
        return obj

    # 1) direct json.loads
    try:
        obj = json.loads(s)
        return _maybe_double_decode(obj)
    except:
        pass

    # 2) scan cari JSON pertama yang valid (atasi prefix "Here is the JSON requested:")
    decoder = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch not in "{[\"":
            continue
        try:
            obj, _ = decoder.raw_decode(s[i:])
            obj = _maybe_double_decode(obj)
            return obj
        except json.JSONDecodeError:
            continue

    # 3) regex fallback (object / array)
    match_obj = re.search(r"\{.*\}", s, re.DOTALL)
    if match_obj:
        try:
            return json.loads(match_obj.group())
        except:
            pass

    match_arr = re.search(r"\[.*\]", s, re.DOTALL)
    if match_arr:
        try:
            return json.loads(match_arr.group())
        except:
            pass

    raise Exception(f"Gemini output bukan JSON valid:\n{s[:1000]}")


# ==============================
# MERGE PDF
# ==============================
def _merge_pdfs(pdf_paths):
    merger = PdfMerger()

    for p in pdf_paths:
        merger.append(p)

    out = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    merger.write(out.name)
    merger.close()

    return out.name


# ==============================
# COMPRESS PDF
# ==============================
def _compress_pdf_if_needed(input_path, max_mb=45):
    size_mb = os.path.getsize(input_path) / (1024 * 1024)

    if size_mb <= max_mb:
        return input_path

    compressed_path = input_path.replace(".pdf", "_compressed.pdf")

    cmd = [
        "gs",
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        "-dPDFSETTINGS=/ebook",
        "-dNOPAUSE",
        "-dQUIET",
        "-dBATCH",
        f"-sOutputFile={compressed_path}",
        input_path,
    ]

    subprocess.run(cmd, check=True)

    return compressed_path


# ==============================
# UPLOAD PDF TO GCS
# ==============================
def _upload_temp_pdf_to_gcs(local_path, invoice_name):
    bucket = storage_client.bucket(BUCKET_NAME)

    blob_path = f"tmp/gemini_input/{invoice_name}_merged.pdf"
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)

    return f"gs://{BUCKET_NAME}/{blob_path}"


# ==============================
# GEMINI CALL
# (FIX kecil: parts bisa None -> hindari NoneType iterable)
# ==============================
def _call_gemini(pdf_path, prompt, invoice_name):
    file_uri = _upload_temp_pdf_to_gcs(pdf_path, invoice_name)

    parts = [
        types.Part.from_uri(
            file_uri=file_uri,
            mime_type="application/pdf",
        )
    ]
    parts.append(types.Part.from_text(text=prompt))

    try:
        response = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(
                    role="user",
                    parts=parts,
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.5,
                top_p=1,
                max_output_tokens=65535,
            ),
        )

        if not response:
            raise Exception("Empty response from Gemini")

        # structured output kadang ada response.parsed
        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            try:
                return json.dumps(parsed, ensure_ascii=False)
            except:
                return str(parsed)

        if hasattr(response, "text") and response.text:
            return response.text.strip()

        if getattr(response, "candidates", None):
            cand0 = response.candidates[0]
            content = getattr(cand0, "content", None)
            parts_resp = getattr(content, "parts", None)

            text_output = ""
            if isinstance(parts_resp, list):
                for p in parts_resp:
                    if hasattr(p, "text") and p.text:
                        text_output += p.text

            if text_output.strip():
                return text_output.strip()

        raise Exception("Gemini response tidak mengandung text")

    except Exception as e:
        raise Exception(f"Gemini call failed: {str(e)}")


# ==============================
# (NEW) SAVE RAW/TMP OUTPUT to GCS (untuk flow kamu: simpan temp result lalu tarik kembali)
# ==============================
def _save_tmp_text(invoice_name, name, text):
    bucket = storage_client.bucket(BUCKET_NAME)
    blob_path = f"{TMP_PREFIX}/{invoice_name}_{name}.txt"
    bucket.blob(blob_path).upload_from_string(text or "", content_type="text/plain")
    return blob_path

def _save_tmp_json(invoice_name, name, data):
    bucket = storage_client.bucket(BUCKET_NAME)
    blob_path = f"{TMP_PREFIX}/{invoice_name}_{name}.json"
    bucket.blob(blob_path).upload_from_string(
        json.dumps(data, ensure_ascii=False, indent=2),
        content_type="application/json",
    )
    return blob_path

def _load_tmp_json(blob_path):
    bucket = storage_client.bucket(BUCKET_NAME)
    txt = bucket.blob(blob_path).download_as_text()
    return json.loads(txt)


# ==============================
# GET TOTAL ROW
# ==============================
def _get_total_row(pdf_path, invoice_name):
    raw = _call_gemini(
        pdf_path,
        ROW_SYSTEM_INSTRUCTION,
        invoice_name,
    )

    print("=== RAW TOTAL ROW RESPONSE ===")
    print(raw)

    data = _parse_json_safe(raw)

    if isinstance(data, dict) and "total_row" in data:
        return int(data["total_row"])

    # fallback kalau model ngasih array
    if isinstance(data, list):
        for it in data:
            if isinstance(it, dict) and "total_row" in it:
                return int(it["total_row"])

    raise Exception(f"total_row tidak ditemukan di response: {data}")


# ==============================
# SAVE BATCH TMP
# ==============================
def _save_batch_tmp(invoice_name, batch_no, json_array):
    if not isinstance(json_array, list):
        raise Exception("Batch result bukan array")

    bucket = storage_client.bucket(BUCKET_NAME)

    blob_path = f"{TMP_PREFIX}/{invoice_name}_batch_{batch_no}.json"

    bucket.blob(blob_path).upload_from_string(
        json.dumps(json_array, indent=2),
        content_type="application/json"
    )


# ==============================
# MERGE ALL BATCHES
# ==============================
def _merge_all_batches(invoice_name):
    bucket = storage_client.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix=f"{TMP_PREFIX}/{invoice_name}_batch_"))

    all_rows = []

    for blob in blobs:
        content = blob.download_as_text()
        data = json.loads(content)
        if isinstance(data, list):
            all_rows.extend(data)

    return all_rows


# ==============================
# GET PO JSON URI (DIRECT FROM GCS)
# ==============================
def _get_po_json_uri():
    bucket = storage_client.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix=f"{PO_PREFIX}/"))

    json_files = [
        b for b in blobs
        if b.name.endswith(".json") and not b.name.endswith("/")
    ]

    if not json_files:
        raise Exception("PO JSON tidak ditemukan di folder po/")

    if len(json_files) > 1:
        raise Exception("Lebih dari 1 PO JSON ditemukan. Harus hanya 1 file.")

    po_blob = json_files[0]

    return f"gs://{BUCKET_NAME}/{po_blob.name}"


# ==============================
# FILTER PO JSON
# ==============================
def _stream_filter_po_lines(target_po_numbers):
    po_uri = _get_po_json_uri()
    parsed = urlparse(po_uri)

    bucket = storage_client.bucket(parsed.netloc)
    blob = bucket.blob(parsed.path.lstrip("/"))

    matched = []

    with blob.open("rb") as f:
        for item in ijson.items(f, "item"):
            po_no = item.get("po_no")
            if po_no in target_po_numbers:
                matched.append(item)

    return matched


# ==============================
# PO MAPPING
# ==============================
def _map_po_to_details(po_lines, detail_rows):
    po_index = {}
    for line in po_lines:
        po_no = line.get("po_no")
        po_index.setdefault(po_no, []).append(line)

    used_po_lines = set()

    for row in detail_rows:
        inv_po = row.get("inv_customer_po_no")
        inv_article = (row.get("inv_spart_item_no") or "").strip()
        inv_desc = (row.get("inv_description") or "").strip()

        if inv_po not in po_index:
            row["_po_mapped"] = False
            continue

        candidates = po_index[inv_po]

        chosen = None
        chosen_key = None

        for idx, po_line in enumerate(candidates):
            if (inv_po, idx) in used_po_lines:
                continue

            vendor_article = (po_line.get("vendor_article_no") or "").strip()
            sap_article = (po_line.get("sap_article_no") or "").strip()

            if inv_article and (
                inv_article == vendor_article
                or inv_article == sap_article
            ):
                chosen = po_line
                chosen_key = (inv_po, idx)
                break

            if inv_desc and (
                inv_desc == vendor_article
                or inv_desc == sap_article
            ):
                chosen = po_line
                chosen_key = (inv_po, idx)
                break

        if chosen:
            used_po_lines.add(chosen_key)
            row["_po_mapped"] = True
            row["_po_data"] = chosen
        else:
            row["_po_mapped"] = False

    return detail_rows


# =========================================================
# VALIDATE PO DATA
# =========================================================
def _validate_po(detail_rows):
    for row in detail_rows:
        if not row.get("_po_mapped"):
            row["match_score"] = "false"
            row["match_description"] = "PO item tidak ditemukan"
            continue

        po_data = row.get("_po_data")

        vendor_article = po_data.get("vendor_article_no")
        sap_article = po_data.get("sap_article_no")

        final_vendor_article = vendor_article or sap_article or "null"

        row["po_no"] = po_data.get("po_no", "null")
        row["po_vendor_article_no"] = final_vendor_article
        row["po_text"] = po_data.get("po_text", "null")
        row["po_sap_article_no"] = sap_article or "null"
        row["po_line"] = po_data.get("po_line", "null")
        row["po_quantity"] = po_data.get("po_quantity", "null")
        row["po_unit"] = po_data.get("po_unit", "null")
        row["po_price"] = po_data.get("po_price", "null")
        row["po_currency"] = po_data.get("po_currency", "null")

        errors = []

        inv_price = row.get("inv_unit_price")
        po_price = po_data.get("po_price")

        inv_currency = row.get("inv_price_unit")
        po_currency = po_data.get("po_currency")

        if inv_price != po_price:
            errors.append(f"po_price mismatch (inv: {inv_price}, po: {po_price})")

        if inv_currency != po_currency:
            errors.append(f"po_currency mismatch (inv: {inv_currency}, po: {po_currency})")

        if errors:
            row["match_score"] = "false"

            prev = row.get("match_description")

            if prev and prev != "null":
                row["match_description"] = prev + "; " + "; ".join(errors)
            else:
                row["match_description"] = "; ".join(errors)

        row.pop("_po_data", None)
        row.pop("_po_mapped", None)

    return detail_rows


# ==============================
# (NEW) MAP PO -> TOTAL (hanya untuk field po_quantity & po_price)
# ==============================
def _append_total_error(total_obj, msg):
    total_obj["match_score"] = "false"
    prev = total_obj.get("match_description") or "null"
    if prev == "null":
        total_obj["match_description"] = msg
    else:
        total_obj["match_description"] = prev + "; " + msg

def _map_po_to_total(total_data, po_lines, po_numbers_from_detail):
    """
    total_data: output OCR total (idealnya JSON ARRAY dengan 1 object)
    """
    if total_data is None:
        return total_data

    # normalize ke list[dict]
    if isinstance(total_data, dict):
        total_data = [total_data]

    if not isinstance(total_data, list) or not total_data or not isinstance(total_data[0], dict):
        return total_data

    total_obj = total_data[0]
    total_obj.setdefault("match_score", "true")
    total_obj.setdefault("match_description", "null")

    po_numbers = {p for p in po_numbers_from_detail if p}
    if not po_numbers:
        _append_total_error(total_obj, "PO number tidak ditemukan pada output detail")
        return total_data

    lines = [l for l in po_lines if l.get("po_no") in po_numbers]
    if not lines:
        _append_total_error(total_obj, "PO lines tidak ditemukan di master PO JSON")
        return total_data

    # expected qty sum
    qty_sum = 0.0
    qty_found = False
    for l in lines:
        q = l.get("po_quantity")
        if q is None:
            continue
        try:
            qty_sum += float(str(q).strip())
            qty_found = True
        except:
            pass

    # expected price (harus unik)
    price_set = set()
    for l in lines:
        p = l.get("po_price")
        if p is None:
            continue
        try:
            price_set.add(float(str(p).strip()))
        except:
            pass

    expected_price = None
    if len(price_set) == 1:
        expected_price = list(price_set)[0]
    elif len(price_set) > 1:
        _append_total_error(total_obj, "PO memiliki lebih dari 1 po_price (ambiguous untuk total)")

    # fill kalau kosong
    if total_obj.get("po_quantity") in (None, "", "null") and qty_found:
        total_obj["po_quantity"] = qty_sum

    if total_obj.get("po_price") in (None, "", "null") and expected_price is not None:
        total_obj["po_price"] = expected_price

    # validate mismatch kalau sudah ada nilainya
    try:
        if qty_found and total_obj.get("po_quantity") not in (None, "", "null"):
            if float(total_obj["po_quantity"]) != float(qty_sum):
                _append_total_error(total_obj, f"po_quantity mismatch (total: {total_obj['po_quantity']}, po_sum: {qty_sum})")
    except:
        pass

    try:
        if expected_price is not None and total_obj.get("po_price") not in (None, "", "null"):
            if float(total_obj["po_price"]) != float(expected_price):
                _append_total_error(total_obj, f"po_price mismatch (total: {total_obj['po_price']}, po: {expected_price})")
    except:
        pass

    return total_data


# =========================================================
# (NEW) CONVERT TO CSV WITH CUSTOM PATH (agar bisa 3 output)
# =========================================================
def _convert_to_csv_path(blob_path, rows):
    if rows is None:
        raise Exception("Tidak ada data untuk CSV")

    # normalize dict -> list
    if isinstance(rows, dict):
        rows = [rows]

    if not isinstance(rows, list) or not rows:
        raise Exception("Tidak ada data untuk CSV")

    # ambil union keys biar kolom lengkap
    keys = []
    seen = set()
    for r in rows:
        if isinstance(r, dict):
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)

    if not keys:
        raise Exception("Row CSV tidak memiliki kolom")

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")

    with open(tmp_file.name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r if isinstance(r, dict) else {})

    bucket = storage_client.bucket(BUCKET_NAME)
    bucket.blob(blob_path).upload_from_filename(tmp_file.name)

    return f"gs://{BUCKET_NAME}/{blob_path}"


# ==============================
# MAIN RUN OCR
# ==============================
def run_ocr(invoice_name, uploaded_pdf_paths, with_total_container):
    """
    Flow yang dihasilkan dari fungsi ini:

    1) UI -> parameter masuk ke sini (invoice_name, uploaded_pdf_paths, with_total_container)
    2) merge + compress bila perlu
    3) OCR: detail (batch), total (1x), container (1x)
    4) mapping PO: detail + total saja
    5) postprocess: convert ke CSV per output
    6) simpan CSV per output ke GCS
    """

    bucket = storage_client.bucket(BUCKET_NAME)

    # 2) MERGE & COMPRESS PDF
    merged_pdf = _merge_pdfs(uploaded_pdf_paths)
    merged_pdf = _compress_pdf_if_needed(merged_pdf)

    # 3a) DETAIL: GET TOTAL ROW FROM GEMINI
    total_row = _get_total_row(merged_pdf, invoice_name)

    # 3a) DETAIL: BATCH EXTRACTION
    first_index = 1
    batch_no = 1

    while first_index <= total_row:
        last_index = min(first_index + BATCH_SIZE - 1, total_row)

        prompt = build_detail_prompt(
            total_row=total_row,
            first_index=first_index,
            last_index=last_index
        )

        raw = _call_gemini(merged_pdf, prompt, invoice_name)

        # simpan raw per batch (temp result)
        _save_tmp_text(invoice_name, f"detail_raw_batch_{batch_no}", raw)

        json_array = _parse_json_safe(raw)
        _save_batch_tmp(invoice_name, batch_no, json_array)

        first_index = last_index + 1
        batch_no += 1

    # MERGE ALL DETAIL BATCHES
    all_rows = _merge_all_batches(invoice_name)
    if not all_rows:
        raise Exception("Tidak ada data detail hasil Gemini")

    # simpan merged detail ke temp result dan TARIK KEMBALI (sesuai flow kamu)
    detail_tmp_blob = _save_tmp_json(invoice_name, "detail_merged", all_rows)
    all_rows = _load_tmp_json(detail_tmp_blob)

    # 3b) TOTAL & CONTAINER OCR (tanpa indexing)
    total_data = None
    container_data = None

    if with_total_container:
        # TOTAL
        raw_total = _call_gemini(merged_pdf, TOTAL_SYSTEM_INSTRUCTION, invoice_name)
        _save_tmp_text(invoice_name, "total_raw", raw_total)
        total_data = _parse_json_safe(raw_total)

        total_tmp_blob = _save_tmp_json(invoice_name, "total", total_data)
        total_data = _load_tmp_json(total_tmp_blob)  # tarik kembali

        # CONTAINER
        raw_container = _call_gemini(merged_pdf, CONTAINER_SYSTEM_INSTRUCTION, invoice_name)
        _save_tmp_text(invoice_name, "container_raw", raw_container)
        container_data = _parse_json_safe(raw_container)

        container_tmp_blob = _save_tmp_json(invoice_name, "container", container_data)
        container_data = _load_tmp_json(container_tmp_blob)  # tarik kembali

    # 4) MAPPING PO: hanya DETAIL + TOTAL
    po_numbers = {
        row.get("inv_customer_po_no")
        for row in all_rows
        if isinstance(row, dict) and row.get("inv_customer_po_no")
    }

    po_lines = _stream_filter_po_lines(po_numbers)

    # DETAIL mapping + validate
    all_rows = _map_po_to_details(po_lines, all_rows)
    all_rows = _validate_po(all_rows)

    # TOTAL mapping (isi/validasi po_quantity & po_price)
    if total_data is not None:
        total_data = _map_po_to_total(total_data, po_lines, po_numbers)

    # 5-6) POSTPROCESS -> CSV per output & SIMPAN ke GCS
    # (path output dibuat per output)
    detail_csv_uri = _convert_to_csv_path(f"output/{invoice_name}_detail.csv", all_rows)

    total_csv_uri = None
    if total_data is not None:
        total_csv_uri = _convert_to_csv_path(f"output/{invoice_name}_total.csv", total_data)

    container_csv_uri = None
    if container_data is not None:
        container_csv_uri = _convert_to_csv_path(f"output/{invoice_name}_container.csv", container_data)

    # CLEAN TEMP FILES (prefix TMP_PREFIX)
    for blob in bucket.list_blobs(prefix=TMP_PREFIX):
        blob.delete()

    # Return semua output (biar UI bisa tampilkan 3 link)
    return {
        "detail_csv": detail_csv_uri,
        "total_csv": total_csv_uri,
        "container_csv": container_csv_uri,
    }
