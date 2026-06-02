# =========================================================
# FANCY STREAMLIT FINGERPRINT RECONSTRUCTION WEB APP
# =========================================================

import streamlit as st
import cv2
import numpy as np
import pyfing as pf
import tempfile
import zipfile
import os
from pathlib import Path

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Fingerprint Reconstruction",
    page_icon="🔍",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.big-title {
    font-size: 45px;
    font-weight: bold;
    color: white;
    text-align: center;
    margin-bottom: 10px;
}

.subtitle {
    font-size: 18px;
    color: #BBBBBB;
    text-align: center;
    margin-bottom: 30px;
}

.stButton>button {
    width: 100%;
    border-radius: 12px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
    background-color: #00ADB5;
    color: white;
    border: none;
}

.stDownloadButton>button {
    width: 100%;
    border-radius: 12px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
    background-color: #00C853;
    color: white;
    border: none;
}

.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="big-title">🔍 Fingerprint Reconstruction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a ZIP folder containing fingerprint images '
    'and automatically reconstruct them using Deep Learning.'
    '</div>',
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📌 Instructions")

    st.write("""
    ### Supported Formats
    - BMP
    - PNG
    - JPG
    - JPEG

    ### Workflow
    1. Upload ZIP file
    2. Images get extracted
    3. Fingerprints reconstructed
    4. Download output ZIP
    """)

    st.success("Deep Learning Reconstruction Enabled")

# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_zip = st.file_uploader(
    "📂 Upload ZIP File",
    type=["zip"]
)

# =========================================================
# PROCESS FUNCTION
# =========================================================

def reconstruct_fingerprint(image_path, output_path):

    fingerprint = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if fingerprint is None:
        return False

    try:

        # =================================================
        # SEGMENTATION
        # =================================================

        mask = pf.fingerprint_segmentation(
            fingerprint,
            method="SUFS"
        )

        mask = (mask > 0).astype(np.uint8) * 255

        # =================================================
        # ORIENTATION + FREQUENCY
        # =================================================

        orientations = pf.orientation_field_estimation(
            fingerprint,
            mask,
            method="SNFOE"
        )

        frequencies = pf.frequency_estimation(
            fingerprint,
            orientations,
            mask,
            method="SNFFE"
        )

        # =================================================
        # ENHANCEMENT
        # =================================================

        enhanced = pf.fingerprint_enhancement(
            fingerprint,
            orientations,
            frequencies,
            mask,
            method="SNFEN"
        )

        enhanced = enhanced.astype(np.uint8)
        enhanced = cv2.flip(enhanced, 1)
        # =================================================
        # BLACK BACKGROUND
        # =================================================

        final_output = np.zeros_like(enhanced)

        final_output[mask > 0] = enhanced[mask > 0]

        # =================================================
        # CLEANING
        # =================================================

        kernel = np.ones((2, 2), np.uint8)

        final_output = cv2.morphologyEx(
            final_output,
            cv2.MORPH_OPEN,
            kernel
        )

        # =================================================
        # SAVE OUTPUT
        # =================================================

        cv2.imwrite(str(output_path), final_output)

        return True

    except Exception as e:

        st.error(f"Error processing {Path(image_path).name}")
        st.error(str(e))

        return False

# =========================================================
# MAIN PROCESSING
# =========================================================

if uploaded_zip is not None:

    st.info("ZIP uploaded successfully")

    with tempfile.TemporaryDirectory() as temp_dir:

        zip_path = os.path.join(temp_dir, "input.zip")

        # Save uploaded zip
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.read())

        # Extract
        extract_folder = os.path.join(
            temp_dir,
            "extracted"
        )

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)

        # Output folder
        output_folder = os.path.join(
            temp_dir,
            "output"
        )

        os.makedirs(output_folder, exist_ok=True)

        # Supported formats
        extensions = [
            ".bmp",
            ".png",
            ".jpg",
            ".jpeg"
        ]

        image_files = []

        for root, dirs, files in os.walk(extract_folder):

            for file in files:

                if Path(file).suffix.lower() in extensions:

                    image_files.append(
                        os.path.join(root, file)
                    )

        st.write(f"### 🖼 Found {len(image_files)} images")

        # =================================================
        # PROCESS BUTTON
        # =================================================

        if st.button("🚀 Start Reconstruction"):

            progress_bar = st.progress(0)

            status_text = st.empty()

            processed_count = 0

            preview_placeholder = st.empty()

            for idx, image_file in enumerate(image_files):

                image_name = Path(image_file).stem

                output_path = os.path.join(
                    output_folder,
                    f"{image_name}_reconstructed.png"
                )

                status_text.write(
                    f"Processing: {Path(image_file).name}"
                )

                success = reconstruct_fingerprint(
                    image_file,
                    output_path
                )

                if success:
                    processed_count += 1

                    # Show preview
                    preview_image = cv2.imread(
                        output_path,
                        cv2.IMREAD_GRAYSCALE
                    )

                    preview_placeholder.image(
                        preview_image,
                        caption=f"Latest Output: {image_name}",
                        width=350
                    )

                progress_bar.progress(
                    (idx + 1) / len(image_files)
                )

            st.success(
                f"✅ Successfully processed "
                f"{processed_count} images"
            )

            # =================================================
            # CREATE ZIP
            # =================================================

            final_zip = os.path.join(
                temp_dir,
                "reconstructed_fingerprints.zip"
            )

            with zipfile.ZipFile(
                final_zip,
                "w"
            ) as zipf:

                for root, dirs, files in os.walk(output_folder):

                    for file in files:

                        file_path = os.path.join(
                            root,
                            file
                        )

                        zipf.write(
                            file_path,
                            arcname=file
                        )

            # =================================================
            # DOWNLOAD
            # =================================================

            with open(final_zip, "rb") as f:

                st.download_button(
                    label="⬇ Download Reconstructed ZIP",
                    data=f,
                    file_name="reconstructed_fingerprints.zip",
                    mime="application/zip"
                )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

