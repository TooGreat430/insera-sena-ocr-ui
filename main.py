import streamlit as st
import tempfile
from function import run_ocr
from google.cloud import storage
from config import BUCKET_NAME, TMP_PREFIX
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(layout="wide")

# =========================
# CUSTOM CSS (BESARIN SIDEBAR)
# =========================
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
    st.image("logo-polygon-insera-sena.jpg", width=120)  # <-- taruh file logo.png di root project

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

    prefix = f"result/{report_type}/"

    blobs = list(storage_client.list_blobs(BUCKET_NAME, prefix=prefix))

    if not blobs:
        st.warning("Belum ada file result.")
    else:
        file_names = [os.path.basename(b.name) for b in blobs if not b.name.endswith("/")]

        selected_file = st.selectbox("Pilih file", file_names)

        if selected_file:
            blob = bucket.blob(f"{prefix}{selected_file}")
            file_bytes = blob.download_as_bytes()

            st.download_button(
                label="Download File",
                data=file_bytes,
                file_name=selected_file,
                mime="application/pdf"
            )
