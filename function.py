import io
import json
import re
import tempfile
import os
import csv
import subprocess

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
# GET TOTAL ROW
# ==============================

def _get_total_row(pdf_path, invoice_name):

    raw = _call_gemini(
        pdf_path,
        ROW_SYSTEM_INSTRUCTION,
        invoice_name,
        po_json_uri=None
    )

    print("=== RAW TOTAL ROW RESPONSE ===")
    print(raw)

    data = _parse_json_safe(raw)

    if isinstance(data, dict) and "total_row" in data:
        return int(data["total_row"])

    raise Exception(f"total_row tidak ditemukan di response: {data}")


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

def _call_gemini(pdf_path, prompt, invoice_name, po_json_uri=None):

    file_uri = _upload_temp_pdf_to_gcs(pdf_path, invoice_name)

    parts = [
        types.Part.from_uri(
            file_uri=file_uri,
            mime_type="application/pdf",
        )
    ]

    if po_json_uri:
        parts.append(
            types.Part.from_uri(
                file_uri=po_json_uri,
                mime_type="text/plain",
            )
        )

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
                temperature=0.1,
                top_p=1,
                max_output_tokens=16384,
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
# CONVERT TO CSV
# ==============================

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

    # 🔥 DIRECT PO URI (NO TEMP COPY)
    po_json_uri = _get_po_json_uri()

    merged_pdf = _merge_pdfs(uploaded_pdf_paths)
    merged_pdf = _compress_pdf_if_needed(merged_pdf)

    total_row = _get_total_row(merged_pdf, invoice_name)

    first_index = 1
    batch_no = 1

    while first_index <= total_row:

        last_index = min(first_index + BATCH_SIZE - 1, total_row)

        prompt = build_detail_prompt(
            total_row=total_row,
            first_index=first_index,
            last_index=last_index
        )

        raw = _call_gemini(
            merged_pdf,
            prompt,
            invoice_name,
            po_json_uri=po_json_uri
        )

        json_array = _parse_json_safe(raw)

        _save_batch_tmp(invoice_name, batch_no, json_array)

        first_index = last_index + 1
        batch_no += 1

    all_rows = _merge_all_batches(invoice_name)

    csv_uri = _convert_to_csv(invoice_name, all_rows)

    for blob in bucket.list_blobs(prefix=TMP_PREFIX):
        blob.delete()

    return csv_uri
