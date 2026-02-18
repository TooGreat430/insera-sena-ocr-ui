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
# ==============================

def _parse_json_safe(raw_text):

    if not raw_text:
        raise Exception("Gemini returned empty response")

    raw_text = raw_text.strip()

    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```json", "", raw_text)
        raw_text = re.sub(r"^```", "", raw_text)
        raw_text = raw_text.rstrip("```").strip()

    try:
        return json.loads(raw_text)
    except:
        pass

    match_obj = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match_obj:
        try:
            return json.loads(match_obj.group())
        except:
            pass

    match_arr = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if match_arr:
        try:
            return json.loads(match_arr.group())
        except:
            pass

    raise Exception(f"Gemini output bukan JSON valid:\n{raw_text[:1000]}")

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
                temperature=0.5,
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

            # EXACT MATCH CONDITIONS
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

        errors = []

        inv_price = row.get("inv_unit_price")
        po_price = po_data.get("po_price")

        inv_currency = row.get("inv_price_unit")
        po_currency = po_data.get("po_currency")

        # PRICE VALIDATION (DETAILED)
        if inv_price != po_price:
            errors.append(
                f"po_price mismatch (inv: {inv_price}, po: {po_price})"
            )

        # CURRENCY VALIDATION (DETAILED)
        if inv_currency != po_currency:
            errors.append(
                f"po_currency mismatch (inv: {inv_currency}, po: {po_currency})"
            )

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
        if row.get("inv_customer_po_no")
    }

    po_lines = _stream_filter_po_lines(po_numbers)

    # MAP PO TO DETAIL
    all_rows = _map_po_to_details(po_lines, all_rows)

    # VALIDATE PO
    all_rows = _validate_po(all_rows)

    # CONVERT TO CSV
    csv_uri = _convert_to_csv(invoice_name, all_rows)

    # CLEAN TEMP FILES
    for blob in bucket.list_blobs(prefix=TMP_PREFIX):
        blob.delete()

    return csv_uri
