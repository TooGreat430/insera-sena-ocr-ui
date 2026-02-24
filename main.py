import streamlit as st
import tempfile
from function import run_ocr
from google.cloud import storage
from config import BUCKET_NAME, TMP_PREFIX
import os
import re
from datetime import timezone, timedelta

st.set_page_config(layout="wide")

st.markdown("""
<style>
    section[data-testid="stSidebar"] * {
        font-size: 18px !important;
    }
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 5])

with col1:
    st.image("logo-polygon-insera-sena.jpg", width=120) 

with col2:
    st.markdown('<div class="main-title">OCR Gemini</div>', unsafe_allow_html=True)

menu = st.sidebar.radio("Menu", ["Upload", "Report"])

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

if menu == "Upload":

    st.subheader("Upload Documents")

    invoice = st.file_uploader("Invoice*", type="pdf")
    packing = st.file_uploader("Packing List*", type="pdf")
    bl = st.file_uploader("Bill of Lading", type="pdf")
    coo = st.file_uploader("COO", type="pdf")

    output_name = st.text_input("Output file name (default invoice name)")

    if st.button("Extract"):

        if not invoice or not packing:
            st.warning("Invoice dan Packing List wajib diupload")

        else:
            pdf_paths = []

            for f in [invoice, packing, bl, coo]:
                if f:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    tmp.write(f.read())
                    tmp.close()
                    pdf_paths.append(tmp.name)

                    # upload ke GCS tmp
                    bucket.blob(f"{TMP_PREFIX}/{f.name}") \
                        .upload_from_filename(tmp.name)

            run_ocr(
                invoice_name=output_name or invoice.name.replace('.pdf',''),
                uploaded_pdf_paths=pdf_paths,
                with_total_container=bool(bl and coo)
            )

            st.success("OCR selesai diproses")

if menu == "Report":

    st.subheader("Download OCR Result")

    report_type = st.selectbox(
        "Pilih Report",
        ["detail", "total", "container"]
    )

    result_prefix = f"output/{report_type}/"
    tmp_prefix = f"{TMP_PREFIX}/"

    result_blobs = list(storage_client.list_blobs(BUCKET_NAME, prefix=result_prefix))
    tmp_blobs = list(storage_client.list_blobs(BUCKET_NAME, prefix=tmp_prefix))

    files_data = []

    for blob in result_blobs:
        if blob.name.endswith("/"):
            continue

        files_data.append({
            "invoice": os.path.basename(blob.name),
            "status": "DONE",
            "updated": blob.updated,
            "path": blob.name
        })

    running_invoices = set()

    batch_re = re.compile(r"^(?P<inv>.+)_batch_\d+\.json$")

    for blob in tmp_blobs:
        name = os.path.basename(blob.name)
        m = batch_re.match(name)
        if m:
            running_invoices.add(m.group("inv"))


    for invoice in running_invoices:
        # check if already marked done
        already_done = any(f["invoice"] == f"{invoice}_{report_type}.csv" for f in files_data)

        if not already_done:
            files_data.append({
                "invoice": invoice,
                "status": "RUNNING",
                "updated": None,
                "path": None
            })

    if not files_data:
        st.warning("Belum ada file result.")
    else:
        # Sort by updated time (DONE first newest)
        files_data = sorted(
            files_data,
            key=lambda x: (x["status"], x["updated"] or 0),
            reverse=True
        )

        for f in files_data:

            col1, col2, col3, col4 = st.columns([3, 2, 3, 2])

            with col1:
                st.write(f["invoice"])

            with col2:
                if f["status"] == "DONE":
                    st.success("DONE")
                else:
                    st.warning("RUNNING")

            with col3:
                if f["updated"]:
                    wib_time = f["updated"].astimezone(timezone(timedelta(hours=7)))
                    st.write(wib_time.strftime("%Y-%m-%d %H:%M:%S"))
                else:
                    st.write("-")

            with col4:
                if f["status"] == "DONE":
                    blob = bucket.blob(f["path"])
                    file_bytes = blob.download_as_bytes()

                    st.download_button(
                        label="Download",
                        data=file_bytes,
                        file_name=f["invoice"],
                        mime="application/octet-stream"
                    )

