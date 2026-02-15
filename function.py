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
# DOWNLOAD PO JSON
# ==============================

def _download_single_po_json():
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

    return json_files[0].download_as_text()


# ==============================
# MERGE PDF (INVOICE ONLY)
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
# COMPRESS PDF (NO SPLIT)
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

    new_size = os.path.getsize(compressed_path) / (1024 * 1024)

    if new_size > max_mb:
        compressed_path2 = input_path.replace(".pdf", "_compressed2.pdf")

        cmd2 = [
            "gs",
            "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.4",
            "-dPDFSETTINGS=/screen",
            "-dNOPAUSE",
            "-dQUIET",
            "-dBATCH",
            f"-sOutputFile={compressed_path2}",
            input_path,
        ]

        subprocess.run(cmd2, check=True)

        return compressed_path2

    return compressed_path


# ==============================
# GEMINI CALL (PDF + JSON TEXT)
# ==============================

def _upload_temp_pdf_to_gcs(local_path, invoice_name):
    bucket = storage_client.bucket(BUCKET_NAME)

    blob_path = f"tmp/gemini_input/{invoice_name}_merged.pdf"
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)

    return f"gs://{BUCKET_NAME}/{blob_path}"


def _call_gemini(pdf_path, prompt, invoice_name, po_json_text=None):

    file_uri = _upload_temp_pdf_to_gcs(pdf_path, invoice_name)

    parts = [
        types.Part.from_uri(
            file_uri=file_uri,
            mime_type="application/pdf",
        )
    ]

    if po_json_text:
        parts.append(
            types.Part.from_text(
                text="PURCHASE ORDER DATA (JSON):\n" + po_json_text
            )
        )

    parts.append(
        types.Part.from_text(text=prompt)
    )

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
            parts = response.candidates[0].content.parts
            text_output = ""
            for p in parts:
                if hasattr(p, "text") and p.text:
                    text_output += p.text
            if text_output:
                return text_output.strip()

        raise Exception("Gemini response tidak mengandung text")

    except Exception as e:
        raise Exception(f"Gemini call failed: {str(e)}")


# ==============================
# JSON SAFE PARSER
# ==============================

def _parse_json_safe(raw_text):
    if not raw_text:
        raise Exception("Gemini returned empty response")

    raw_text = raw_text.strip()

    # Remove markdown block ```json ```
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```json", "", raw_text)
        raw_text = re.sub(r"^```", "", raw_text)
        raw_text = raw_text.rstrip("```").strip()

    # Try direct parse
    try:
        return json.loads(raw_text)
    except:
        pass

    # Try extract JSON object
    match_obj = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match_obj:
        try:
            return json.loads(match_obj.group())
        except:
            pass

    # Try extract JSON array
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
    raw = _call_gemini(pdf_path, ROW_SYSTEM_INSTRUCTION, invoice_name)

    print("=== RAW TOTAL ROW RESPONSE ===")
    print(raw)

    data = _parse_json_safe(raw)

    if "total_row" not in data:
        raise Exception(f"total_row tidak ditemukan di response: {data}")

    return int(data["total_row"])


# ==============================
# SAVE BATCH TEMP
# ==============================

def _save_batch_tmp(invoice_name, batch_no, json_array):
    bucket = storage_client.bucket(BUCKET_NAME)

    path = f"tmp/result/{invoice_name}_detail__{batch_no}.json"

    bucket.blob(path).upload_from_string(
        json.dumps(json_array),
        content_type="application/json"
    )


# ==============================
# MERGE ALL BATCHES
# ==============================

def _merge_all_batches(invoice_name):
    bucket = storage_client.bucket(BUCKET_NAME)

    prefix = f"tmp/result/{invoice_name}_detail__"
    blobs = list(bucket.list_blobs(prefix=prefix))

    all_rows = []

    for b in blobs:
        data = json.loads(b.download_as_text())
        all_rows.extend(data)

    for row in all_rows:
        row.pop("index", None)

    return all_rows, blobs


# ==============================
# CONVERT TO CSV
# ==============================

def _convert_to_csv(data):
    if not data:
        return ""

    output = io.StringIO()
    fieldnames = list(data[0].keys())

    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for row in data:
        writer.writerow(row)

    return output.getvalue()


# ==============================
# MAIN RUN OCR
# ==============================

def run_ocr(invoice_name, uploaded_pdf_paths, with_total_container):

    bucket = storage_client.bucket(BUCKET_NAME)

    # 1️⃣ Download PO JSON (tetap JSON)
    po_json = _download_single_po_json()

    # 2️⃣ Merge invoice PDF saja
    merged_pdf = _merge_pdfs(uploaded_pdf_paths)

    # 3️⃣ Compress jika perlu
    merged_pdf = _compress_pdf_if_needed(merged_pdf)

    # 4️⃣ Get total row
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
            po_json_text=po_json
        )

        json_array = _parse_json_safe(raw)

        _save_batch_tmp(invoice_name, batch_no, json_array)

        first_index = last_index + 1
        batch_no += 1

    # 5️⃣ Merge batches
    all_rows, blobs = _merge_all_batches(invoice_name)
    final_csv = _convert_to_csv(all_rows)

    bucket.blob(
        f"{RESULT_PREFIX}/detail/{invoice_name}_detail.csv"
    ).upload_from_string(final_csv, content_type="text/csv")

    for b in blobs:
        b.delete()

    # ================= TOTAL & CONTAINER =================

    if with_total_container:

        raw_total = _call_gemini(
            merged_pdf,
            TOTAL_SYSTEM_INSTRUCTION,
            invoice_name,
            po_json_text=po_json
        )

        total_json = _parse_json_safe(raw_total)

        bucket.blob(
            f"{RESULT_PREFIX}/total/{invoice_name}_total.json"
        ).upload_from_string(
            json.dumps(total_json),
            content_type="application/json"
        )

        raw_container = _call_gemini(
            merged_pdf,
            CONTAINER_SYSTEM_INSTRUCTION,
            invoice_name,
            po_json_text=po_json
        )

        container_json = _parse_json_safe(raw_container)

        bucket.blob(
            f"{RESULT_PREFIX}/container/{invoice_name}_container.json"
        ).upload_from_string(
            json.dumps(container_json),
            content_type="application/json"
        )

    # Cleanup tmp files
    for blob in bucket.list_blobs(prefix=TMP_PREFIX):
        blob.delete()
