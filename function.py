import json
import re
import tempfile
import os
import csv
import subprocess
import ijson
import uuid
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

# ==============================
# SETTINGS
# ==============================
BATCH_SIZE = 5

# "unique" = pakai run_id random (aman dari overwrite)
# "single" = pakai run_id tetap "single" (asumsi tidak ada rerun/parallel)
RUN_ID_MODE = "unique"

storage_client = storage.Client()

genai_client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
)

# ==============================
# JSON SAFE PARSER (robust)
# ==============================
def _parse_json_safe(raw_text: str):
    if not raw_text:
        raise Exception("Gemini returned empty response")

    s = raw_text.strip()

    # strip code fences
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        s = s.strip()

    # helper: jika hasil json.loads adalah string JSON, decode lagi
    def _maybe_double_decode(obj):
        if isinstance(obj, str):
            t = obj.strip()
            if t and t[0] in "{[":
                try:
                    return json.loads(t)
                except:
                    return obj
        return obj

    # 1) coba direct json.loads
    try:
        obj = json.loads(s)
        return _maybe_double_decode(obj)
    except:
        pass

    # 2) scan cari JSON pertama yang valid (atasi prefix "Here is ...")
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

    raise Exception(f"Gemini output bukan JSON valid:\n{s[:1000]}")

# ==============================
# PDF MERGE
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
# PDF COMPRESS
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
# GCS HELPERS
# ==============================
def _gcs_upload_json(blob_path: str, data) -> str:
    bucket = storage_client.bucket(BUCKET_NAME)
    bucket.blob(blob_path).upload_from_string(
        json.dumps(data, ensure_ascii=False, indent=2),
        content_type="application/json",
    )
    return f"gs://{BUCKET_NAME}/{blob_path}"

def _gcs_download_json(blob_path: str):
    bucket = storage_client.bucket(BUCKET_NAME)
    txt = bucket.blob(blob_path).download_as_text()
    return json.loads(txt)

def _gcs_upload_text(blob_path: str, text: str) -> str:
    bucket = storage_client.bucket(BUCKET_NAME)
    bucket.blob(blob_path).upload_from_string(text or "", content_type="text/plain")
    return f"gs://{BUCKET_NAME}/{blob_path}"

def _cleanup_prefix(prefix: str):
    bucket = storage_client.bucket(BUCKET_NAME)
    for blob in bucket.list_blobs(prefix=prefix):
        blob.delete()

# ==============================
# UPLOAD MERGED PDF ONCE
# ==============================
def _upload_merged_pdf_to_gcs(local_path, invoice_name, run_id):
    bucket = storage_client.bucket(BUCKET_NAME)
    blob_path = f"tmp/gemini_input/{invoice_name}/{run_id}/merged.pdf"
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    return f"gs://{BUCKET_NAME}/{blob_path}"

# ==============================
# GEMINI CALL USING FILE_URI
# ==============================
def _extract_response_text(response) -> str:
    if not response:
        return ""

    # 1) Structured output (response_schema) biasanya masuk ke sini
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        try:
            return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            return str(parsed)

    # 2) Normal text shortcut (kalau ada)
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    # 3) Kandidat -> content -> parts (parts bisa None)
    cands = getattr(response, "candidates", None) or []
    if cands:
        cand0 = cands[0]
        content = getattr(cand0, "content", None)

        parts = getattr(content, "parts", None)
        if isinstance(parts, list):
            out = ""
            for part in parts:
                t = getattr(part, "text", None)
                if isinstance(t, str) and t:
                    out += t
            if out.strip():
                return out.strip()

        # beberapa versi mungkin punya content.text
        ctext = getattr(content, "text", None)
        if isinstance(ctext, str) and ctext.strip():
            return ctext.strip()

    # 4) Last resort: serialize response supaya debug kamu tetap ada isi
    for meth in ("model_dump_json", "to_json"):
        fn = getattr(response, meth, None)
        if callable(fn):
            try:
                s = fn()
                if isinstance(s, str) and s.strip():
                    return s
            except Exception:
                pass

    return str(response)


def _call_gemini_uri(file_uri: str, prompt: str, config: types.GenerateContentConfig) -> str:
    parts = [
        types.Part.from_uri(file_uri=file_uri, mime_type="application/pdf"),
        types.Part.from_text(text=prompt),
    ]

    response = genai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[types.Content(role="user", parts=parts)],
        config=config,
    )
    return _extract_response_text(response)

def _call_gemini_json_uri(
    file_uri: str,
    prompt: str,
    run_prefix: str,
    debug_name: str,
    config: types.GenerateContentConfig,
    retries: int = 2,
):
    last_err = None
    p = prompt

    for attempt in range(1, retries + 1):
        raw = _call_gemini_uri(file_uri, p, config)

        # simpan raw untuk audit/debug
        _gcs_upload_text(f"{run_prefix}/debug/{debug_name}_attempt_{attempt}.txt", raw)

        try:
            return raw, _parse_json_safe(raw)
        except Exception as e:
            last_err = e
            p = p + (
                "\n\nCRITICAL: Output HARUS murni JSON valid sesuai skema. "
                "DILARANG ada kalimat pembuka seperti 'Here is...'. "
                "DILARANG markdown/kode."
            )

    raise last_err

# ==============================
# GET TOTAL ROW (STRUCTURED OUTPUT)
# ==============================
def _get_total_row_structured(file_uri: str, run_prefix: str):
    # schema native SDK (lebih aman)
    schema = types.Schema(
        type=types.Type.OBJECT,
        required=["total_row"],
        properties={
            "total_row": types.Schema(type=types.Type.INTEGER),
        },
    )

    prompt = ROW_SYSTEM_INSTRUCTION + "\n\nOUTPUT RULE: HANYA JSON valid sesuai schema."

    cfg = types.GenerateContentConfig(
        temperature=0,
        top_p=1,
        max_output_tokens=64,
        response_mime_type="application/json",
        response_schema=schema,
        system_instruction="Output harus JSON valid, tanpa teks tambahan.",
    )

    _, data = _call_gemini_json_uri(
        file_uri=file_uri,
        prompt=prompt,
        run_prefix=run_prefix,
        debug_name="total_row",
        config=cfg,
        retries=3,
    )

    if isinstance(data, dict) and "total_row" in data:
        return int(data["total_row"])

    if isinstance(data, list):
        for it in data:
            if isinstance(it, dict) and "total_row" in it:
                return int(it["total_row"])

    raise Exception(f"total_row tidak ditemukan di response: {data}")

# ==============================
# SAVE / MERGE DETAIL BATCHES
# ==============================
def _save_detail_batch_tmp(run_prefix, batch_no, json_array):
    if not isinstance(json_array, list):
        raise Exception("Batch result bukan array")
    blob_path = f"{run_prefix}/detail/batch_{batch_no:04d}.json"
    _gcs_upload_json(blob_path, json_array)

def _merge_all_detail_batches(run_prefix):
    bucket = storage_client.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix=f"{run_prefix}/detail/batch_"))
    blobs.sort(key=lambda b: b.name)

    all_rows = []
    for blob in blobs:
        data = json.loads(blob.download_as_text())
        if isinstance(data, list):
            all_rows.extend(data)
    return all_rows

# ==============================
# GET PO JSON URI & FILTER (stream)
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
# PO MAPPING -> DETAIL
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

            if inv_article and (inv_article == vendor_article or inv_article == sap_article):
                chosen = po_line
                chosen_key = (inv_po, idx)
                break

            if inv_desc and (inv_desc == vendor_article or inv_desc == sap_article):
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

def _validate_po(detail_rows):
    for row in detail_rows:
        row.setdefault("match_score", "true")
        row.setdefault("match_description", "null")

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
            prev = row.get("match_description", "null")
            if prev and prev != "null":
                row["match_description"] = prev + "; " + "; ".join(errors)
            else:
                row["match_description"] = "; ".join(errors)

        row.pop("_po_data", None)
        row.pop("_po_mapped", None)

    return detail_rows

# ==============================
# MAP PO -> TOTAL (isi po_quantity & po_price)
# ==============================
def _append_total_error(total_obj, msg):
    total_obj["match_score"] = "false"
    prev = total_obj.get("match_description") or "null"
    if prev == "null":
        total_obj["match_description"] = msg
    else:
        total_obj["match_description"] = prev + "; " + msg

def _map_po_to_total(total_data, po_lines, po_numbers_from_detail):
    if not isinstance(total_data, list) or not total_data:
        return total_data

    total_obj = total_data[0]
    if not isinstance(total_obj, dict):
        return total_data

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

    qty_sum = 0.0
    qty_found = False
    for l in lines:
        q = l.get("po_quantity")
        if q is None:
            continue
        try:
            qty_sum += float(q)
            qty_found = True
        except:
            pass

    price_set = set()
    for l in lines:
        p = l.get("po_price")
        if p is None:
            continue
        try:
            price_set.add(float(p))
        except:
            pass

    po_price_value = None
    if len(price_set) == 1:
        po_price_value = list(price_set)[0]
    elif len(price_set) > 1:
        _append_total_error(total_obj, "PO memiliki lebih dari 1 po_price (ambiguous untuk total)")

    if total_obj.get("po_quantity") in (None, "", "null") and qty_found:
        total_obj["po_quantity"] = qty_sum

    if total_obj.get("po_price") in (None, "", "null") and po_price_value is not None:
        total_obj["po_price"] = po_price_value

    return total_data

# ==============================
# CONVERT ANY JSON -> CSV (upload to GCS)
# ==============================
def _convert_any_to_csv_gcs(blob_path, data):
    if data is None:
        raise Exception("Tidak ada data untuk CSV")

    if isinstance(data, dict):
        rows = [data]
    elif isinstance(data, list):
        rows = [r for r in data if isinstance(r, dict)]
    else:
        raise Exception(f"Tipe data tidak bisa di-CSV: {type(data)}")

    if not rows:
        raise Exception("Tidak ada row dict untuk CSV")

    keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    with open(tmp_file.name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    bucket = storage_client.bucket(BUCKET_NAME)
    bucket.blob(blob_path).upload_from_filename(tmp_file.name)

    return f"gs://{BUCKET_NAME}/{blob_path}"

# ==============================
# MAIN RUN OCR
# ==============================
def run_ocr(invoice_name, uploaded_pdf_paths, with_total_container=True):
    # run_id
    run_id = "single" if RUN_ID_MODE == "single" else uuid.uuid4().hex[:10]
    run_prefix = f"{TMP_PREFIX}/results/{invoice_name}/{run_id}"

    detail_out_csv = f"output/{invoice_name}/{run_id}/detail.csv"
    total_out_csv = f"output/{invoice_name}/{run_id}/total.csv"
    container_out_csv = f"output/{invoice_name}/{run_id}/container.csv"

    # 2) merge + 3) compress
    merged_pdf = _merge_pdfs(uploaded_pdf_paths)
    merged_pdf = _compress_pdf_if_needed(merged_pdf)

    # upload merged pdf once
    file_uri = _upload_merged_pdf_to_gcs(merged_pdf, invoice_name, run_id)

    # config umum (json-only fence)  ✅ stop_sequences DIHAPUS
    cfg_json_only = types.GenerateContentConfig(
        temperature=0,
        top_p=1,
        max_output_tokens=65535,
        system_instruction=(
            "Keluarkan HANYA JSON valid sesuai skema. "
            "DILARANG output teks lain, markdown, atau kode."
        ),
        response_mime_type="application/json",
    )

    # 4) total_row (structured)
    total_row = _get_total_row_structured(file_uri, run_prefix)

    # 4) detail batching
    first_index = 1
    batch_no = 1

    while first_index <= total_row:
        last_index = min(first_index + BATCH_SIZE - 1, total_row)

        prompt = build_detail_prompt(
            total_row=total_row,
            first_index=first_index,
            last_index=last_index,
        )

        _, json_array = _call_gemini_json_uri(
            file_uri=file_uri,
            prompt=prompt,
            run_prefix=run_prefix,
            debug_name=f"detail_batch_{batch_no:04d}",
            config=cfg_json_only,
            retries=2,
        )

        _save_detail_batch_tmp(run_prefix, batch_no, json_array)

        first_index = last_index + 1
        batch_no += 1

    all_rows = _merge_all_detail_batches(run_prefix)
    _gcs_upload_json(f"{run_prefix}/detail/merged.json", all_rows)

    if not all_rows:
        raise Exception("Tidak ada data detail hasil Gemini")

    # 5) total & container (single call)
    total_data = None
    container_data = None

    if with_total_container:
        _, total_data = _call_gemini_json_uri(
            file_uri=file_uri,
            prompt=TOTAL_SYSTEM_INSTRUCTION,
            run_prefix=run_prefix,
            debug_name="total",
            config=cfg_json_only,
            retries=2,
        )
        _gcs_upload_json(f"{run_prefix}/total/total.json", total_data)

        _, container_data = _call_gemini_json_uri(
            file_uri=file_uri,
            prompt=CONTAINER_SYSTEM_INSTRUCTION,
            run_prefix=run_prefix,
            debug_name="container",
            config=cfg_json_only,
            retries=2,
        )
        _gcs_upload_json(f"{run_prefix}/container/container.json", container_data)

    # 6) mapping PO hanya Detail + Total
    po_numbers = {
        row.get("inv_customer_po_no")
        for row in all_rows
        if row.get("inv_customer_po_no")
    }
    po_lines = _stream_filter_po_lines(po_numbers)

    all_rows = _map_po_to_details(po_lines, all_rows)
    all_rows = _validate_po(all_rows)
    _gcs_upload_json(f"{run_prefix}/detail/detail_mapped.json", all_rows)

    if total_data is not None:
        total_data = _map_po_to_total(total_data, po_lines, po_numbers)
        _gcs_upload_json(f"{run_prefix}/total/total_mapped.json", total_data)

    # 7-8) csv output
    detail_csv_uri = _convert_any_to_csv_gcs(detail_out_csv, all_rows)

    total_csv_uri = None
    if total_data is not None:
        total_csv_uri = _convert_any_to_csv_gcs(total_out_csv, total_data)

    container_csv_uri = None
    if container_data is not None:
        container_csv_uri = _convert_any_to_csv_gcs(container_out_csv, container_data)

    _cleanup_prefix(run_prefix)

    return {
        "run_id": run_id,
        "detail_csv": detail_csv_uri,
        "total_csv": total_csv_uri,
        "container_csv": container_csv_uri,
    }
