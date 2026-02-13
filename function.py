import io
import json
import tempfile
import os
import csv
from google.cloud import storage
from PyPDF2 import PdfMerger
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part

from config import *
from detail import build_detail_prompt
from row import ROW_SYSTEM_INSTRUCTION

BATCH_SIZE = 5

storage_client = storage.Client()

vertexai.init(
    project=PROJECT_ID,
    location=LOCATION
)

model = GenerativeModel(MODEL_NAME)

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

    # 🔥 HANDLE DICT
    if isinstance(data, dict):
        for k, v in data.items():
            if v is None:
                v = "null"
            draw_line(f"{k}: {v}")

    # 🔥 HANDLE LIST OF DICT
    elif isinstance(data, list):
        for i, item in enumerate(data):
            draw_line(f"--- ROW {i+1} ---")
            if isinstance(item, dict):
                for k, v in item.items():
                    if v is None:
                        v = "null"
                    draw_line(f"{k}: {v}")
            else:
                draw_line(str(item))

    else:
        draw_line(str(data))

    c.save()
    return tmp.name



def _merge_pdfs(pdf_paths):
    merger = PdfMerger()

    for p in pdf_paths:
        merger.append(p)

    out = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    merger.write(out.name)
    merger.close()

    return out.name


def _call_gemini(pdf_path, system_instruction):

    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    response = model.generate_content(
        contents=[
            Part.from_data(
                mime_type="application/pdf",
                data=file_bytes
            ),
            Part.from_text(system_instruction)
        ],
        generation_config={
            "temperature": 0.1
        }
    )

    return response.text


def _parse_json_safe(text):
    try:
        return json.loads(text)
    except:
        # try cleaning
        text = text.strip()
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
        raise Exception("Gemini output bukan JSON valid")


def _get_total_row(merged_pdf):
    raw = _call_gemini(merged_pdf, ROW_SYSTEM_INSTRUCTION)
    data = json.loads(raw)
    return int(data["total_row"])


def _save_batch_tmp(invoice_name, batch_no, json_array):
    bucket = storage_client.bucket(BUCKET_NAME)

    path = f"tmp/result/{invoice_name}_detail__{batch_no}.json"

    bucket.blob(path).upload_from_string(
        json.dumps(json_array),
        content_type="application/json"
    )


def _merge_all_batches(invoice_name):
    bucket = storage_client.bucket(BUCKET_NAME)

    prefix = f"tmp/result/{invoice_name}_detail__"
    blobs = list(bucket.list_blobs(prefix=prefix))

    all_rows = []

    for b in blobs:
        data = json.loads(b.download_as_text())
        all_rows.extend(data)

    # DROP COLUMN index
    for row in all_rows:
        if "index" in row:
            del row["index"]

    return all_rows, blobs


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

def run_ocr(invoice_name, uploaded_pdf_paths, with_total_container):

    bucket = storage_client.bucket(BUCKET_NAME)

    # 1️⃣ PO JSON → PDF
    po_json = _download_single_po_json()
    po_pdf = _json_to_pdf(po_json)

    # 2️⃣ Merge All PDF
    merged_pdf = _merge_pdfs(uploaded_pdf_paths + [po_pdf])

    # 3️⃣ Hitung total_row (1x)
    total_row = _get_total_row(merged_pdf)

    # 4️⃣ AUTO LOOP BATCH
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

        if last_index == total_row:
            break

    # 5️⃣ Merge Semua Batch
    all_rows, blobs = _merge_all_batches(invoice_name)

    final_csv = _convert_to_csv(all_rows)

    # 6️⃣ Upload Final CSV
    bucket.blob(
        f"{RESULT_PREFIX}/detail/{invoice_name}_detail.csv"
    ).upload_from_string(final_csv)

    # 7️⃣ Cleanup tmp/result batch files
    for b in blobs:
        b.delete()

    # 8️⃣ Cleanup tmp upload files
    for blob in bucket.list_blobs(prefix=TMP_PREFIX):
        blob.delete()
