import io 
import json 
import re 
import tempfile 
import os 
import csv 
import subprocess 
import ijson 
import uuid
from decimal import Decimal, InvalidOperation
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
def _parse_json_safe(raw_text: str):
    if not raw_text:
        raise Exception("Gemini returned empty response")

    raw_text = raw_text.strip()

    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)
        raw_text = raw_text.strip()

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

    raise Exception(f"Gemini output bukan JSON valid:\n{raw_text[:1500]}")

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

    try:
        subprocess.run(cmd, check=True)
        return compressed_path
    except:
        # fallback kalau ghostscript tidak ada/gagal
        return input_path

# ==============================
# GCS HELPERS
# ==============================
def _gcs_upload_json(blob_path, data):
    bucket = storage_client.bucket(BUCKET_NAME)
    bucket.blob(blob_path).upload_from_string(
        json.dumps(data, ensure_ascii=False, indent=2),
        content_type="application/json",
    )
    return f"gs://{BUCKET_NAME}/{blob_path}"

def _gcs_download_json(blob_path):
    bucket = storage_client.bucket(BUCKET_NAME)
    return json.loads(bucket.blob(blob_path).download_as_text(encoding="utf-8"))

def _gcs_list(prefix):
    bucket = storage_client.bucket(BUCKET_NAME)
    return list(bucket.list_blobs(prefix=prefix))

def _gcs_cleanup_prefix(prefix):
    bucket = storage_client.bucket(BUCKET_NAME)
    for blob in bucket.list_blobs(prefix=prefix):
        blob.delete()

# ==============================
# UPLOAD PDF TO GCS
# ==============================

def _upload_temp_pdf_to_gcs(local_path, invoice_name, run_prefix):
    bucket = storage_client.bucket(BUCKET_NAME)

    blob_path = f"{run_prefix}/gemini_input/{invoice_name}_merged.pdf"
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)

    return f"gs://{BUCKET_NAME}/{blob_path}"

# ==============================
# GEMINI CALL
# ==============================

def _call_gemini(pdf_path, prompt, invoice_name, run_prefix, expect_json=True):

    file_uri = _upload_temp_pdf_to_gcs(pdf_path, invoice_name, run_prefix)

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

def _get_total_row(pdf_path, invoice_name, run_prefix):

    raw = _call_gemini(
        pdf_path,
        ROW_SYSTEM_INSTRUCTION,
        invoice_name,
        run_prefix,
        expect_json=True
    )

    data = _parse_json_safe(raw)

    if isinstance(data, dict) and "total_row" in data:
        return int(data["total_row"])

    raise Exception(f"total_row tidak ditemukan di response: {data}")


# ==============================
# SAVE DETAIL BATCH TMP
# ==============================

def _save_detail_batch_tmp(run_prefix, batch_no, json_array):

    if not isinstance(json_array, list):
        raise Exception("Batch result bukan array")

    blob_path = f"{run_prefix}/tmp_result/detail/batch_{batch_no:04d}.json"
    _gcs_upload_json(blob_path, json_array)
    return blob_path

# ==============================
# MERGE ALL BATCHES
# ==============================

def _merge_all_detail_batches(run_prefix):
    blobs = _gcs_list(prefix=f"{run_prefix}/tmp_result/detail/batch_")
    blobs.sort(key=lambda b: b.name)

    all_rows = []
    for blob in blobs:
        data = json.loads(blob.download_as_text(encoding="utf-8"))
        if isinstance(data, list):
            all_rows.extend(data)

    return all_rows

# ==============================
# EXTRACT TOTAL + CONTAINER to TMP
# ==============================
def _extract_total_to_tmp(pdf_path, invoice_name, run_prefix):
    raw = _call_gemini(pdf_path, TOTAL_SYSTEM_INSTRUCTION, invoice_name, run_prefix, expect_json=True)
    total_data = _parse_json_safe(raw)
    blob_path = f"{run_prefix}/tmp_result/total/raw.json"
    _gcs_upload_json(blob_path, total_data)
    return blob_path

def _extract_container_to_tmp(pdf_path, invoice_name, run_prefix):
    raw = _call_gemini(pdf_path, CONTAINER_SYSTEM_INSTRUCTION, invoice_name, run_prefix, expect_json=True)
    container_data = _parse_json_safe(raw)
    blob_path = f"{run_prefix}/tmp_result/container/raw.json"
    _gcs_upload_json(blob_path, container_data)
    return blob_path

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
    
    return f"gs://{BUCKET_NAME}/{json_files[0].name}"

# ==============================
# FILTER PO JSON
# ==============================

def _stream_filter_po_lines(target_po_numbers):
    target_po_numbers = {str(x).strip() for x in (target_po_numbers or set()) if x is not None}

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
            if str(po_no).strip() in target_po_numbers:
                matched.append(item)

    return matched

# ==============================
# PO MAPPING DETAIL
# ==============================

def _map_po_to_details(po_lines, detail_rows):
    po_index = {}
    for line in (po_lines or []):
        po_no = line.get("po_no")
        if po_no is None:
            continue
        po_no = str(po_no).strip()
        po_index.setdefault(po_no, []).append(line)

    used_po_lines = set()

    for row in (detail_rows or []):
        inv_po = row.get("inv_customer_po_no")
        inv_po = str(inv_po).strip() if inv_po is not None else None

        inv_article = (row.get("inv_spart_item_no") or "").strip()
        inv_desc = (row.get("inv_description") or "").strip()

        if not inv_po or inv_po not in po_index:
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

# =========================================================
# VALIDATE PO DATA
# =========================================================

def _validate_po(detail_rows):
    for row in (detail_rows or []):
        if not row.get("_po_mapped"):
            row["match_score"] = "false"
            row["match_description"] = "PO item tidak ditemukan"
            row.pop("_po_data", None)
            row.pop("_po_mapped", None)
            continue

        po_data = row.get("_po_data") or {}

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

        # price validation
        inv_price_dec = _to_decimal(inv_price)
        po_price_dec = _to_decimal(po_price)
        if inv_price_dec is None or po_price_dec is None:
            errors.append("po_price atau inv_unit_price null/tidak valid")
        elif inv_price_dec != po_price_dec:
            errors.append(
                f"po_price mismatch (inv: {inv_price}, po: {po_price})"
            )
        
        # currency validation
        if str(inv_currency).strip() != str(po_currency).strip():
            errors.append(f"po_currency mismatch (inv: {inv_currency}, po: {po_currency})")

        if errors:
            row["match_score"] = "false"
            prev = row.get("match_description")
            if prev and prev != "null":
                row["match_description"] = prev + "; " + "; ".join(errors)
            else:
                row["match_description"] = "; ".join(errors)
        else:
            if row.get("match_score") != "false":
                row["match_score"] = "true"
            if not row.get("match_description"):
                row["match_description"] = "null"

        row.pop("_po_data", None)
        row.pop("_po_mapped", None)

    return detail_rows

# =========================================================
# TOTAL: SUPER SIMPLE PO FILL + 1 VALIDATION RULE (STRICT SCHEMA)
# - Fill: po_quantity, po_price
# - Validate: po_price == inv_unit_price
#   inv_unit_price = inv_total_amount / inv_total_quantity
# - If invalid:
#     match_score = "false"
#     match_description:
#       * if previous match_score already false and desc not null -> append ; "msg"
#       * else replace "null" with msg
# - No new fields
# =========================================================

TOTAL_SCHEMA_KEYS = [
    "match_score",
    "match_description",

    "inv_quantity",
    "inv_amount",
    "inv_amount_unit",
    "inv_total_quantity",
    "inv_total_amount",
    "inv_total_nw",
    "inv_total_gw",
    "inv_total_volume",
    "inv_total_package",

    "pl_package_unit",
    "pl_package_count",
    "pl_weight_unit",
    "pl_nw",
    "pl_gw",
    "pl_volume_unit",
    "pl_volume",
    "pl_total_quantity",
    "pl_total_amount",
    "pl_total_nw",
    "pl_total_gw",
    "pl_total_volume",
    "pl_total_package",

    "po_quantity",
    "po_price",

    "bl_shipper_name",
    "bl_shipper_address",
    "bl_no",
    "bl_date",
    "bl_consignee_name",
    "bl_consignee_address",
    "bl_consignee_tax_id",
    "bl_seller_name",
    "bl_seller_address",
    "bl_lc_number",
    "bl_notify_party",
    "bl_vessel",
    "bl_voyage_no",
    "bl_port_of_loading",
    "bl_port_of_destination",
    "bl_gw_unit",
    "bl_gw",
    "bl_volume_unit",
    "bl_volume",
    "bl_package_count",
    "bl_package_unit",
]

def _to_decimal(x):
    if x is None:
        return None
    if isinstance(x, (int, float, Decimal)):
        return Decimal(str(x))
    s = str(x).strip()
    if not s:
        return None
    s = re.sub(r"[^\d,.\-]", "", s)
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    else:
        if "," in s:
            parts = s.split(",")
            if len(parts[-1]) == 3:
                s = s.replace(",", "")
            else:
                s = s.replace(",", ".")
    try:
        return Decimal(s)
    except InvalidOperation:
        return None

def _almost_equal(a: Decimal, b: Decimal, abs_tol=Decimal("0.01")):
    if a is None or b is None:
        return False
    return abs(a - b) <= abs_tol

def _as_total_rows(total_data):
    if total_data is None:
        return []
    if isinstance(total_data, dict):
        return [total_data]
    if isinstance(total_data, list):
        return [x for x in total_data if isinstance(x, dict)]
    return []

def _filter_total_schema(row: dict):
    return {k: row.get(k, None) for k in TOTAL_SCHEMA_KEYS}

def _append_po_error(row: dict, msg: str):
    prev_score = str(row.get("match_score") or "true").lower()
    prev_desc = row.get("match_description")

    has_prev_desc = prev_desc is not None and str(prev_desc).strip().lower() != "null" and str(prev_desc).strip() != ""

    row["match_score"] = "false"
    if prev_score == "false" and has_prev_desc:
        row["match_description"] = f'{prev_desc}; "{msg}"'
    else:
        row["match_description"] = msg

def _aggregate_po_qty_and_price(po_lines):
    # weighted average price over ALL po_lines passed (already filtered by invoice PO numbers)
    qty_sum = Decimal("0")
    amt_sum = Decimal("0")

    for line in (po_lines or []):
        qty = _to_decimal(line.get("po_quantity"))
        price = _to_decimal(line.get("po_price"))
        if qty is None:
            continue
        qty_sum += qty
        if price is not None:
            amt_sum += qty * price

    if qty_sum == 0:
        return None, None

    return qty_sum, (amt_sum / qty_sum)

def _inv_unit_price_from_total(row: dict):
    qty = _to_decimal(row.get("inv_total_quantity"))
    amt = _to_decimal(row.get("inv_total_amount"))
    if qty is None or amt is None or qty == 0:
        return None
    return amt / qty

def fill_po_and_validate_total(total_data, po_lines):
    rows = _as_total_rows(total_data)
    if not rows:
        return []

    po_qty, po_price = _aggregate_po_qty_and_price(po_lines)

    out = []
    for r in rows:
        row = _filter_total_schema(r)

        # fill only PO fields
        if po_qty is not None:
            row["po_quantity"] = float(po_qty)
        if po_price is not None:
            row["po_price"] = float(po_price)

        inv_unit = _inv_unit_price_from_total(row)

        # if cannot validate -> mark false with message
        if po_price is None or inv_unit is None:
            _append_po_error(row, "po_price atau inv_unit_price tidak dapat dihitung")
            out.append(row)
            continue

        # 1 validation rule
        if not _almost_equal(_to_decimal(row.get("po_price")), inv_unit, abs_tol=Decimal("0.01")):
            _append_po_error(
                row,
                f"po_price ({row.get('po_price')}) tidak sama dengan inv_unit_price ({str(inv_unit)})"
            )

        out.append(row)

    return out

# ==============================
# CSV (SIMPLE)
# ==============================
def _csv_cell(v):
    if v is None:
        return "null"
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return v

def _write_csv_to_gcs(blob_path: str, rows: list[dict], fieldnames: list[str] | None = None):
    if not rows:
        rows = [{}]

    if fieldnames is None:
        keys = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fieldnames = keys

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    with open(tmp.name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _csv_cell(r.get(k)) for k in fieldnames})

    bucket = storage_client.bucket(BUCKET_NAME)
    bucket.blob(blob_path).upload_from_filename(tmp.name, content_type="text/csv")
    return f"gs://{BUCKET_NAME}/{blob_path}"

def _to_rows_any(data):
    if data is None:
        return []
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        if all(isinstance(x, dict) for x in data):
            return data
        return [{"value": x} for x in data]
    return [{"value": data}]

# ==============================
# MAIN
# ==============================

def run_ocr(invoice_name, uploaded_pdf_paths, with_total_container=True):
    run_id = uuid.uuid4().hex[:10]
    run_prefix = f"{TMP_PREFIX}/runs/{invoice_name}/{run_id}"

    # 1) merge + compress
    merged_pdf = _merge_pdfs(uploaded_pdf_paths)
    merged_pdf = _compress_pdf_if_needed(merged_pdf)

    # 2) total_row
    total_row = _get_total_row(merged_pdf, invoice_name, run_prefix)

    # 3) detail batches
    first = 1
    batch_no = 1
    while first <= total_row:
        last = min(first + BATCH_SIZE - 1, total_row)
        prompt = build_detail_prompt(total_row=total_row, first_index=first, last_index=last)

        raw = _call_gemini(merged_pdf, prompt, invoice_name, run_prefix, expect_json=True)
        arr = _parse_json_safe(raw)
        _save_detail_batch_tmp(run_prefix, batch_no, arr)

        first = last + 1
        batch_no += 1

    detail_rows = _merge_all_detail_batches(run_prefix)
    if not detail_rows:
        raise Exception("Tidak ada data detail hasil Gemini")

    # 4) total + container
    total_data = None
    container_data = None
    if with_total_container:
        total_blob = _extract_total_to_tmp(merged_pdf, invoice_name, run_prefix)
        cont_blob = _extract_container_to_tmp(merged_pdf, invoice_name, run_prefix)
        total_data = _gcs_download_json(total_blob)
        container_data = _gcs_download_json(cont_blob)

    # 5) filter po by po numbers from detail
    po_numbers = {str(r.get("inv_customer_po_no")).strip() for r in detail_rows if r.get("inv_customer_po_no") is not None}
    po_lines = _stream_filter_po_lines(po_numbers) if po_numbers else []

    # 6) map + validate detail
    detail_rows = _map_po_to_details(po_lines, detail_rows)
    detail_rows = _validate_po(detail_rows)

    # 7) fill + validate total (only PO fields + 1 rule)
    total_rows = []
    if total_data is not None:
        total_rows = fill_po_and_validate_total(total_data, po_lines)
        _gcs_upload_json(f"{run_prefix}/tmp_result/total/mapped.json", total_rows)

    # 8) output csvs
    detail_csv = _write_csv_to_gcs(
        blob_path=f"output/{invoice_name}_{run_id}_detail.csv",
        rows=detail_rows,
        fieldnames=None
    )

    total_csv = None
    if total_rows:
        total_csv = _write_csv_to_gcs(
            blob_path=f"output/{invoice_name}_{run_id}_total.csv",
            rows=total_rows,
            fieldnames=TOTAL_SCHEMA_KEYS
        )

    container_csv = None
    if container_data is not None:
        container_csv = _write_csv_to_gcs(
            blob_path=f"output/{invoice_name}_{run_id}_container.csv",
            rows=_to_rows_any(container_data),
            fieldnames=None
        )

    # 9) cleanup this run only
    _gcs_cleanup_prefix(run_prefix)

    return {
        "run_id": run_id,
        "detail_csv": detail_csv,
        "total_csv": total_csv,
        "container_csv": container_csv,
    }