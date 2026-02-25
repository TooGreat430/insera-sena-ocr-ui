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
genai_client = genai.Client( vertexai=True, project=PROJECT_ID, location=LOCATION, ) 

# ============================== # JSON SAFE PARSER # ============================== 
def _parse_json_safe(raw_text):
    if not raw_text:
        raise Exception("Gemini returned empty response")

    s = raw_text.strip()

    # strip code fences kalau ada
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        s = s.strip()

    # 1) coba direct
    try:
        return json.loads(s)
    except:
        pass

    # 2) cari JSON pertama yang valid (handle prefix "Here is ...")
    decoder = json.JSONDecoder()

    # PRIORITAS: coba mulai dari '[' dulu (array)
    idx_arr = s.find("[")
    if idx_arr != -1:
        try:
            obj, _ = decoder.raw_decode(s[idx_arr:])
            return obj
        except:
            pass

    # lalu coba mulai dari '{' (object)
    idx_obj = s.find("{")
    if idx_obj != -1:
        try:
            obj, _ = decoder.raw_decode(s[idx_obj:])
            return obj
        except:
            pass

    # 3) fallback regex: ARRAY dulu baru OBJECT
    match_arr = re.search(r"\[.*\]", s, re.DOTALL)
    if match_arr:
        try:
            return json.loads(match_arr.group())
        except:
            pass

    match_obj = re.search(r"\{.*\}", s, re.DOTALL)
    if match_obj:
        try:
            return json.loads(match_obj.group())
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
                temperature=0.05,
                top_p=1,
                max_output_tokens=65535,
            ),
        )

        if not response:
            raise Exception("Empty response from Gemini")

        if hasattr(response, "text") and response.text:
            return response.text.strip()

        if response.candidates:
            parts_resp = response.candidates[0].content.parts
            text_output = ""
            for p in parts_resp:
                if hasattr(p, "text") and p.text:
                    text_output += p.text
            if text_output:
                return text_output.strip()

        raise Exception("Gemini response tidak mengandung text")

    except Exception as e:
        raise Exception(f"Gemini call failed: {str(e)}")

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

    # 🔥 LANGSUNG RETURN URI ASLI
    return f"gs://{BUCKET_NAME}/{po_blob.name}"

# ==============================
# Nomalize PO NO
# ==============================

def _norm_po_number(x):
    if x is None:
        return ""
    s = str(x).strip()
    s = re.sub(r"\D", "", s)  # ambil angka saja
    return s.lstrip("0")      # buang leading zero untuk compare

# ==============================
# FILTER PO JSON
# ==============================

def _stream_filter_po_lines(target_po_numbers):

    target_po_numbers = {
        _norm_po_number(x)
        for x in (target_po_numbers or set())
        if x is not None
    }

    po_uri = _get_po_json_uri()
    parsed = urlparse(po_uri)

    bucket = storage_client.bucket(parsed.netloc)
    blob = bucket.blob(parsed.path.lstrip("/"))

    matched = []

    with blob.open("rb") as f:
        for item in ijson.items(f, "item"):
            po_no = item.get("po_no")
            if po_no is None:
                continue

            if _norm_po_number(po_no) in target_po_numbers:
                matched.append(item)

    return matched

# ==============================
# PO MAPPING
# ==============================

def _norm_key(x):
    if x is None:
        return ""
    s = str(x).strip().upper()
    s = re.sub(r"\s+", "", s)          # hapus spasi
    s = re.sub(r"[^A-Z0-9]", "", s)    # hapus dash, slash, dll
    return s

def _map_po_to_details(po_lines, detail_rows):
    """
    Join key:
    - inv_customer_po_no  <-> po_no
    - inv_spart_item_no   <-> vendor_article_no OR sap_article_no

    Normalisasi hanya untuk matching.
    Data yang dipakai untuk output tetap value asli (po_line asli).
    """

    # index: (po_no_norm, article_norm) -> list of (idx_in_po_lines, po_line_asli)
    po_index = {}

    for idx, line in enumerate(po_lines):
        po_no_norm = _norm_po_number(line.get("po_no"))
        if not po_no_norm:
            continue

        v_norm = _norm_key(line.get("vendor_article_no") or line.get("po_vendor_article_no"))
        s_norm = _norm_key(line.get("sap_article_no") or line.get("po_sap_article_no"))


        if v_norm:
            po_index.setdefault((po_no_norm, v_norm), []).append((idx, line))
        if s_norm:
            po_index.setdefault((po_no_norm, s_norm), []).append((idx, line))

    used = set()  # (po_no_norm, idx_in_po_lines)

    for row in detail_rows:
        if not isinstance(row, dict):
            continue

        inv_po_raw = row.get("inv_customer_po_no")
        inv_article_raw = row.get("inv_spart_item_no")

        inv_po_norm = _norm_po_number(inv_po_raw)
        inv_article_norm = _norm_key(inv_article_raw)

        if not inv_po_norm or not inv_article_norm:
            row["_po_mapped"] = False
            continue

        candidates = po_index.get((inv_po_norm, inv_article_norm), [])
        chosen = None
        chosen_key = None

        for idx, po_line in candidates:
            key = (inv_po_norm, idx)
            if key in used:
                continue
            chosen = po_line          # ✅ PO line ASLI
            chosen_key = key
            break

        if chosen:
            used.add(chosen_key)
            row["_po_mapped"] = True
            row["_po_data"] = chosen  # ✅ simpan ASLI untuk dipakai _validate_po
        else:
            row["_po_mapped"] = False

    return detail_rows

# =========================================================
# VALIDATE PO DATA
# =========================================================

def _to_num(x):
    if x is None:
        return None
    try:
        return float(str(x).strip().replace(",", ""))
    except:
        return None
def _validate_po(detail_rows):

    for row in detail_rows:

        if not row.get("_po_mapped"):
            row["match_score"] = "false"
            row["match_description"] = "PO item tidak ditemukan"
            continue

        po_data = row.get("_po_data")

        vendor_article = po_data.get("vendor_article_no") or po_data.get("po_vendor_article_no")
        sap_article = po_data.get("sap_article_no") or po_data.get("po_sap_article_no")


        # FALLBACK VENDOR ARTICLE
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
        row["po_info_record_price"] = po_data.get("po_info_record_price", "null")
        row["po_info_record_currency"] = po_data.get("po_info_record_currency", "null")

        errors = []

        inv_price = _to_num(row.get("inv_unit_price"))
        po_price = _to_num(po_data.get("po_price"))

        inv_currency = str(row.get("inv_price_unit") or "").strip()
        po_currency = str(po_data.get("po_currency") or "").strip()

        if inv_price is not None and po_price is not None and inv_price != po_price:
            errors.append(f"po_price mismatch (inv: {inv_price}, po: {po_price})")

        if inv_currency and po_currency and inv_currency != po_currency:
            errors.append(f"po_currency mismatch (inv: {inv_currency}, po: {po_currency})")


        # APPLY ERRORS
        if errors:
            row["match_score"] = "false"

            prev = row.get("match_description")

            if prev and prev != "null":
                row["match_description"] = prev + "; " + "; ".join(errors)
            else:
                row["match_description"] = "; ".join(errors)

        # cleanup temp
        row.pop("_po_data", None)
        row.pop("_po_mapped", None)

    return detail_rows

# ==============================
# (NEW) MAP PO -> TOTAL
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
    total_data bisa dict atau list[dict]
    Kita isi/validasi field PO di TOTAL berdasarkan po_lines yang relevan.
    """
    if total_data is None:
        return None

    # normalize dict -> list
    if isinstance(total_data, dict):
        total_data = [total_data]

    if not isinstance(total_data, list) or not total_data or not isinstance(total_data[0], dict):
        return total_data

    total_obj = total_data[0]
    total_obj.setdefault("match_score", "true")
    total_obj.setdefault("match_description", "null")

    po_numbers = {
        _norm_po_number(p)
        for p in po_numbers_from_detail
        if p is not None
    }
    if not po_numbers:
        _append_total_error(total_obj, "PO number tidak ditemukan pada output detail")
        return total_data

    lines = [
        l for l in po_lines
        if _norm_po_number(l.get("po_no")) in po_numbers
    ]
    if not lines:
        _append_total_error(total_obj, "PO lines tidak ditemukan di master PO JSON")
        return total_data

    # contoh isi: total po_quantity = sum
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

    # contoh isi: po_price harus unik
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

    # fill kalau kosong/null
    if total_obj.get("po_quantity") in (None, "", "null") and qty_found:
        total_obj["po_quantity"] = qty_sum

    if total_obj.get("po_price") in (None, "", "null") and expected_price is not None:
        total_obj["po_price"] = expected_price

    return total_data


# ==============================
# (NEW) CONVERT TO CSV -> CUSTOM FOLDER/PATH
# ==============================
def _convert_to_csv_path(blob_path, rows):
    if rows is None:
        raise Exception("Tidak ada data untuk CSV")

    # normalize dict -> list
    if isinstance(rows, dict):
        rows = [rows]

    if not isinstance(rows, list) or not rows:
        raise Exception("Tidak ada data untuk CSV")

    # union keys biar kolom lengkap
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


# =========================================================
# CONVERT TO CSV
# =========================================================

def _convert_to_csv(invoice_name, rows):

    if not rows:
        raise Exception("Tidak ada data untuk CSV")

    keys = rows[0].keys()

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")

    with open(tmp_file.name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    bucket = storage_client.bucket(BUCKET_NAME)
    blob_path = f"output/{invoice_name}.csv"

    bucket.blob(blob_path).upload_from_filename(tmp_file.name)

    return f"gs://{BUCKET_NAME}/{blob_path}"


# ==============================
# MAIN RUN OCR
# ==============================

def run_ocr(invoice_name, uploaded_pdf_paths, with_total_container):

    bucket = storage_client.bucket(BUCKET_NAME)

    # MERGE & COMPRESS PDF
    merged_pdf = _merge_pdfs(uploaded_pdf_paths)
    merged_pdf = _compress_pdf_if_needed(merged_pdf)

    # GET TOTAL ROW FROM GEMINI
    total_row = _get_total_row(merged_pdf, invoice_name)

    # BATCH DETAIL EXTRACTION
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
        
        print("========== RAW GEMINI DETAIL ==========")
        print(raw)
        print("========================================")
        
        json_array = _parse_json_safe(raw)
        if isinstance(json_array, dict):
            json_array = [json_array]

        print("========== PARSED TYPE ==========")
        print(type(json_array))
        print("==================================")

        _save_batch_tmp(invoice_name, batch_no, json_array)

        first_index = last_index + 1
        batch_no += 1

    # MERGE ALL GEMINI BATCHES
    all_rows = _merge_all_batches(invoice_name)

    if not all_rows:
        raise Exception("Tidak ada data detail hasil Gemini")

   # LOAD RELEVANT PO LINES
    po_numbers = {
        row.get("inv_customer_po_no")
        for row in all_rows
        if isinstance(row, dict) and row.get("inv_customer_po_no")
    }


    po_lines = _stream_filter_po_lines(po_numbers)
    print("PO NUMBERS:", po_numbers)
    print("PO LINES FOUND:", len(po_lines))

    total_data = None
    container_data = None

    if with_total_container:
        # OCR TOTAL
        raw_total = _call_gemini(merged_pdf, TOTAL_SYSTEM_INSTRUCTION, invoice_name)
        total_data = _parse_json_safe(raw_total)
        if isinstance(total_data, dict):
            total_data = [total_data]

        # OCR CONTAINER
        raw_container = _call_gemini(merged_pdf, CONTAINER_SYSTEM_INSTRUCTION, invoice_name)
        container_data = _parse_json_safe(raw_container)
        if isinstance(container_data, dict):
            container_data = [container_data]

    # MAP PO TO DETAIL
    all_rows = _map_po_to_details(po_lines, all_rows)

    # VALIDATE PO
    all_rows = _validate_po(all_rows)

    # ==============================
    # (NEW) MAP PO TO TOTAL (DETAIL tetap batch, TOTAL tidak batch)
    # ==============================
    if total_data is not None:
        total_data = _map_po_to_total(total_data, po_lines, po_numbers)

    # CONVERT TO CSV
    # ==============================
    # (NEW) OUTPUT PER FOLDER
    # ==============================
    detail_csv_uri = _convert_to_csv_path(
        f"output/detail/{invoice_name}_detail.csv", all_rows
    )

    total_csv_uri = None
    if total_data is not None:
        total_csv_uri = _convert_to_csv_path(
            f"output/total/{invoice_name}_total.csv", total_data
        )

    container_csv_uri = None
    if container_data is not None:
        container_csv_uri = _convert_to_csv_path(
            f"output/container/{invoice_name}_container.csv", container_data
        )


    # CLEAN TEMP FILES
    for blob in bucket.list_blobs(prefix=TMP_PREFIX):
        blob.delete()

    return {
        "detail_csv": detail_csv_uri,
        "total_csv": total_csv_uri,
        "container_csv": container_csv_uri,
    }