import io
import json
import tempfile
import os
from google.cloud import storage
from PyPDF2 import PdfMerger
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part

from config import *
from detail import DETAIL_SYSTEM_INSTRUCTION
from total import TOTAL_SYSTEM_INSTRUCTION
from container import CONTAINER_SYSTEM_INSTRUCTION


storage_client = storage.Client()

vertexai.init(
    project=PROJECT_ID,
    location=LOCATION
)

model = GenerativeModel(MODEL_NAME)

def _download_single_po_json():
    bucket = storage_client.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix=PO_PREFIX))

    if not blobs:
        raise Exception("PO JSON tidak ditemukan di folder PO")

    return blobs[0].download_as_text()


def _json_to_pdf(json_text):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(tmp.name, pagesize=A4)

    y = 800
    data = json.loads(json_text)

    for k, v in data.items():
        c.drawString(40, y, f"{k}: {v}")
        y -= 14
        if y < 40:
            c.showPage()
            y = 800

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

def run_ocr(invoice_name, uploaded_pdf_paths, with_total_container):

    bucket = storage_client.bucket(BUCKET_NAME)

    po_json = _download_single_po_json()
    po_pdf = _json_to_pdf(po_json)

    merged_pdf = _merge_pdfs(uploaded_pdf_paths + [po_pdf])

    detail_csv = _call_gemini(
        merged_pdf,
        DETAIL_SYSTEM_INSTRUCTION
    )

    bucket.blob(
        f"{RESULT_PREFIX}/detail/{invoice_name}_detail.csv"
    ).upload_from_string(detail_csv)

    if with_total_container:

        total_csv = _call_gemini(
            merged_pdf,
            TOTAL_SYSTEM_INSTRUCTION
        )

        container_csv = _call_gemini(
            merged_pdf,
            CONTAINER_SYSTEM_INSTRUCTION
        )

        bucket.blob(
            f"{RESULT_PREFIX}/total/{invoice_name}_total.csv"
        ).upload_from_string(total_csv)

        bucket.blob(
            f"{RESULT_PREFIX}/container/{invoice_name}_container.csv"
        ).upload_from_string(container_csv)

    for blob in bucket.list_blobs(prefix=TMP_PREFIX):
        blob.delete()
