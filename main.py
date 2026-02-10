import streamlit as st
import tempfile
from function import run_ocr
from google.cloud import storage
from config import BUCKET_NAME, TMP_PREFIX

st.set_page_config(layout="wide")
st.title("OCR Gemini – Streamlit UI")

menu = st.sidebar.radio("Menu", ["Upload", "Report"])

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

if menu == "Upload":
    invoice = st.file_uploader("Invoice*", type="pdf")
    packing = st.file_uploader("Packing List*", type="pdf")
    bl = st.file_uploader("Bill of Lading", type="pdf")
    coo = st.file_uploader("COO", type="pdf")

    output_name = st.text_input("Output file name (default invoice name)")

    if st.button("Extract"):
        if not invoice or not packing:
            st.warning("Invoice dan Packing List wajib diupload")
        elif (bl and not coo) or (coo and not bl):
            st.warning("BL dan COO harus diupload bersamaan")
        else:
            pdf_paths = []
            for f in [invoice, packing, bl, coo]:
                if f:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    tmp.write(f.read())
                    tmp.close()
                    pdf_paths.append(tmp.name)
                    bucket.blob(f"{TMP_PREFIX}/{f.name}").upload_from_filename(tmp.name)

            run_ocr(
                invoice_name=output_name or invoice.name.replace('.pdf',''),
                uploaded_pdf_paths=pdf_paths,
                with_total_container=bool(bl and coo)
            )
            st.success("OCR selesai diproses")

if menu == "Report":
    st.info("Report diambil langsung dari GCS result folder")
