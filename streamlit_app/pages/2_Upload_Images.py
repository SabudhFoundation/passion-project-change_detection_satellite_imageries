"""Page 2 — Upload Images: upload T1/T2 .tif band files and preview bands."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components.sidebar import render_sidebar
from components.band_selector import render_band_selector
from utils.helpers import ensure_src_on_path, bands_to_display

render_sidebar()

st.title("📂 Upload Images")
st.caption("Upload your bi-temporal Sentinel-2 band files (T1 = before, T2 = after).")

BAND_FILES = [
    "B01.tif", "B02.tif", "B03.tif", "B04.tif",
    "B05.tif", "B06.tif", "B07.tif", "B08.tif",
    "B8A.tif", "B09.tif", "B10.tif", "B11.tif", "B12.tif"
]

def save_uploaded_files(uploaded_files: list, prefix: str) -> Path | None:
    if not uploaded_files:
        return None
    tmp_dir = Path(tempfile.gettempdir()) / f"ucdnet_{prefix}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for f in uploaded_files:
        dest = tmp_dir / f.name
        dest.write_bytes(f.read())
    return tmp_dir

st.divider()

# ── File uploaders ────────────────────────────────────────────────────────────
st.subheader("Upload Band Files")
st.markdown(
    "Select all **13 `.tif` band files** from each image folder. "
    "For OSCD cities these are inside `imgs_1_rect/` (T1) and `imgs_2_rect/` (T2)."
)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**🕐 T1 — Before image**")
    t1_files = st.file_uploader(
        "Upload T1 band files (.tif)",
        type=["tif", "tiff"],
        accept_multiple_files=True,
        key="t1_uploader",
        help="Select all 13 band .tif files from imgs_1_rect/ folder",
    )
    if t1_files:
        st.success(f"✅ {len(t1_files)} file(s) uploaded")
        for f in sorted(t1_files, key=lambda x: x.name):
            st.caption(f"• {f.name} ({f.size/1024:.0f} KB)")

with col2:
    st.markdown("**🕑 T2 — After image**")
    t2_files = st.file_uploader(
        "Upload T2 band files (.tif)",
        type=["tif", "tiff"],
        accept_multiple_files=True,
        key="t2_uploader",
        help="Select all 13 band .tif files from imgs_2_rect/ folder",
    )
    if t2_files:
        st.success(f"✅ {len(t2_files)} file(s) uploaded")
        for f in sorted(t2_files, key=lambda x: x.name):
            st.caption(f"• {f.name} ({f.size/1024:.0f} KB)")

label_file = st.file_uploader(
    "🏷️ Ground-truth label (optional — cm.tif or cm.png)",
    type=["png", "tif", "tiff"],
    key="label_uploader",
    help="From cm/ folder in OSCD city. Used to compute F1, precision, recall.",
)
if label_file:
    st.success(f"✅ Label uploaded: {label_file.name}")

st.divider()

# ── Save & Load ───────────────────────────────────────────────────────────────
if t1_files and t2_files:
    if st.button("💾 Save & Load Images", type="primary", use_container_width=True):
        with st.spinner("Saving uploaded files…"):
            try:
                t1_dir = save_uploaded_files(t1_files, "t1")
                t2_dir = save_uploaded_files(t2_files, "t2")

                label_path = None
                if label_file:
                    tmp_label = Path(tempfile.gettempdir()) / f"ucdnet_label_{label_file.name}"
                    tmp_label.write_bytes(label_file.read())
                    label_path = str(tmp_label)

                st.session_state["t1_path"] = str(t1_dir)
                st.session_state["t2_path"] = str(t2_dir)
                st.session_state["label_path"] = label_path

                # Load images using read_bands directly from flat temp dir
                ensure_src_on_path()
                import os, rasterio
                from rasterio.enums import Resampling

                def read_flat_dir(band_dir):
                    b04 = os.path.join(band_dir, "B04.tif")
                    with rasterio.open(b04) as src:
                        target_h, target_w = src.height, src.width
                    bands = []
                    for bfile in BAND_FILES:
                        fpath = os.path.join(band_dir, bfile)
                        with rasterio.open(fpath) as src:
                            data = src.read(
                                1,
                                out_shape=(target_h, target_w),
                                resampling=Resampling.bilinear
                            ).astype(np.float32)
                        bands.append(data)
                    img = np.stack(bands, axis=-1)
                    return np.clip(img, 0, 10000) / 10000.0

                img1 = read_flat_dir(str(t1_dir))
                img2 = read_flat_dir(str(t2_dir))
                st.session_state["_img1_preview"] = img1
                st.session_state["_img2_preview"] = img2
                st.success(f"✅ Images loaded! Shape: {img1.shape} (H × W × 13 bands)")

            except Exception as e:
                st.error(f"Failed to load images: {e}")
                import traceback
                st.code(traceback.format_exc())

# ── Preview ──────────────────────────────────────────────────────────────────
img1 = st.session_state.get("_img1_preview")
img2 = st.session_state.get("_img2_preview")

if img1 is not None and img2 is not None:
    st.subheader("Image Preview")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**T1 — Before**")
        mode1, sel1 = render_band_selector("T1 band")
        disp1 = bands_to_display(img1, sel1 if isinstance(sel1, list) else [sel1])
        st.image(disp1, caption=f"T1 · {img1.shape[0]}×{img1.shape[1]}", use_column_width=True)

    with col_right:
        st.markdown("**T2 — After**")
        mode2, sel2 = render_band_selector("T2 band")
        disp2 = bands_to_display(img2, sel2 if isinstance(sel2, list) else [sel2])
        st.image(disp2, caption=f"T2 · {img2.shape[0]}×{img2.shape[1]}", use_column_width=True)

    with st.expander("Band statistics (T1)"):
        import pandas as pd
        stats = {
            "Band": [f"B{i+1:02d}" for i in range(img1.shape[2])],
            "Min": img1.min(axis=(0, 1)).tolist(),
            "Max": img1.max(axis=(0, 1)).tolist(),
            "Mean": img1.mean(axis=(0, 1)).tolist(),
            "Std": img1.std(axis=(0, 1)).tolist(),
        }
        st.dataframe(pd.DataFrame(stats).set_index("Band"), use_container_width=True)

elif not t1_files or not t2_files:
    st.info("Upload T1 and T2 band files above, then click **Save & Load Images**.")