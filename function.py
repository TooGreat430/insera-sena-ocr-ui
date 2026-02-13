import io
import json
import tempfile
import os
import csv
from google.cloud import storage
from PyPDF2 import PdfMerger
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

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
# JSON → PDF
# ==============================

def _json_to_pdf(json_text):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(tmp.name, pagesize=A4)

    y = 800
    data = json.loads(json_text)

    def draw_line(text):
        nonlocal y
        c.drawString(40, y, text)
        y -= 14
        if y < 40:
            c.showPage()
            y = 800

    if isinstance(data, dict):
        for k, v in data.items():
            draw_line(f"{k}: {v if v is not None else 'null'}")

    elif isinstance(data, list):
        for i, item in enumerate(data):
            draw_line(f"--- ROW {i+1} ---")
            if isinstance(item, dict):
                for k, v in item.items():
                    draw_line(f"{k}: {v if v is not None else 'null'}")
            else:
                draw_line(str(item))
    else:
        draw_line(str(data))

    c.save()
    return tmp.name


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
# GEMINI CALL (GEN AI SDK)
# ==============================

def _call_gemini(pdf_path, prompt):

    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    try:
        response = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(
                            data=file_bytes,
                            mime_type="application/pdf",
                        ),
                        types.Part.from_text(
                            text=prompt
                        ),
                    ],
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                top_p=1,
                max_output_tokens=16384,  # 🔥 turunkan biar stabil
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="OFF",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="OFF",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="OFF",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="OFF",
                    ),
                ],
            ),
        )

        # 🔥 Defensive handling
        if not response:
            raise Exception("Empty response from Gemini")

        if hasattr(response, "text") and response.text:
            return response.text

        # fallback manual extract
        if response.candidates:
            parts = response.candidates[0].content.parts
            text_output = ""
            for p in parts:
                if hasattr(p, "text") and p.text:
                    text_output += p.text
            if text_output:
                return text_output

        raise Exception("Gemini response tidak mengandung text")

    except Exception as e:
        raise Exception(f"Gemini call failed: {str(e)}")



# ==============================
# JSON SAFE PARSER
# ==============================

def _parse_json_safe(text):
    try:
        return json.loads(text)
    except:
        text = text.strip()
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
        raise Exception("Gemini output bukan JSON valid")


# ==============================
# GET TOTAL ROW (1x)
# ==============================

def _get_total_row(merged_pdf):
    raw = _call_gemini(merged_pdf, ROW_SYSTEM_INSTRUCTION)
    data = json.loads(raw)
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

    # DROP INDEX COLUMN
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

    # 1️⃣ PO JSON → PDF
    po_json = _download_single_po_json()
    po_pdf = _json_to_pdf(po_json)

    # 2️⃣ Merge PDF
    merged_pdf = _merge_pdfs(uploaded_pdf_paths + [po_pdf])

    # 3️⃣ Get total_row (1x)
    total_row = _get_total_row(merged_pdf)

    first_index = 1
    batch_no = 1

    while first_index <= total_row:

        last_index = min(first_index + BATCH_SIZE - 1, total_row)

        prompt = build_detail_prompt(
            total_row=total_row,
            first_index=first_index,
            last_index=last_index
        )

        raw = _call_gemini(merged_pdf, prompt)
        json_array = _parse_json_safe(raw)

        _save_batch_tmp(invoice_name, batch_no, json_array)

        first_index = last_index + 1
        batch_no += 1

    # 4️⃣ Merge Semua Batch
    all_rows, blobs = _merge_all_batches(invoice_name)
    final_csv = _convert_to_csv(all_rows)

    bucket.blob(
        f"{RESULT_PREFIX}/detail/{invoice_name}_detail.csv"
    ).upload_from_string(final_csv, content_type="text/csv")

    # 5️⃣ Cleanup tmp/result/
    for b in blobs:
        b.delete()

    # ================= TOTAL & CONTAINER =================

    if with_total_container:

        raw_total = _call_gemini(merged_pdf, TOTAL_SYSTEM_INSTRUCTION)
        total_json = _parse_json_safe(raw_total)

        bucket.blob(
            f"{RESULT_PREFIX}/total/{invoice_name}_total.json"
        ).upload_from_string(
            json.dumps(total_json),
            content_type="application/json"
        )

        raw_container = _call_gemini(
            merged_pdf,
            CONTAINER_SYSTEM_INSTRUCTION
        )

        container_json = _parse_json_safe(raw_container)

        bucket.blob(
            f"{RESULT_PREFIX}/container/{invoice_name}_container.json"
        ).upload_from_string(
            json.dumps(container_json),
            content_type="application/json"
        )

    # Cleanup tmp upload files
    for blob in bucket.list_blobs(prefix=TMP_PREFIX):
        blob.delete()
