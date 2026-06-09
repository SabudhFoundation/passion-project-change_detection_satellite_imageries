"""Page 2 — Upload Images: set T1 / T2 paths and preview bands."""

import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components.sidebar import render_sidebar
from components.band_selector import render_band_selector
from utils.helpers import ensure_src_on_path, load_image_pair_cached, bands_to_display

render_sidebar()

st.title("📂 Upload Images")
st.caption("Set paths to your bi-temporal Sentinel-2 image folders (T1 = before, T2 = after).")

# ── Path inputs ──────────────────────────────────────────────────────────────
st.subheader("Image Paths")
st.markdown(
    "Point to the `imgs_1_rect/` and `imgs_2_rect/` folders for any OSCD city, "
    "or any pair of directories containing Sentinel-2 `.tif` band files."
)

col1, col2 = st.columns(2)
with col1:
    t1_input = st.text_input(
        "T1 path (before image)",
        value=st.session_state.get("t1_path") or "",
        placeholder="src/data/raw/.../city/imgs_1_rect",
    )
with col2:
    t2_input = st.text_input(
        "T2 path (after image)",
        value=st.session_state.get("t2_path") or "",
        placeholder="src/data/raw/.../city/imgs_2_rect",
    )

label_input = st.text_input(
    "Ground-truth label path (optional)",
    value=st.session_state.get("label_path") or "",
    placeholder="src/data/raw/.../city/cm/cm.png",
    help="Provide to compute F1, precision, recall, etc. after prediction.",
)

normalize = st.selectbox(
    "Normalisation",
    ["reflectance", "per_band"],
    help="reflectance: divide by 10000. per_band: z-score per band.",
)

# Validate and store paths
t1_ok = t1_input and Path(t1_input).exists()
t2_ok = t2_input and Path(t2_input).exists()

if t1_input and not t1_ok:
    st.error(f"T1 path not found: `{t1_input}`")
if t2_input and not t2_ok:
    st.error(f"T2 path not found: `{t2_input}`")

if t1_ok:
    st.session_state["t1_path"] = t1_input
if t2_ok:
    st.session_state["t2_path"] = t2_input
if label_input:
    st.session_state["label_path"] = label_input if Path(label_input).is_file() else None
st.session_state["normalize"] = normalize

st.divider()

# ── Preview ──────────────────────────────────────────────────────────────────
st.subheader("Image Preview")

if t1_ok and t2_ok:
    if st.button("Load & preview images", type="primary"):
        with st.spinner("Loading image pair…"):
            try:
                ensure_src_on_path()
                img1, img2 = load_image_pair_cached(t1_input, t2_input, normalize=normalize)
                st.session_state["_img1_preview"] = img1
                st.session_state["_img2_preview"] = img2
                st.success(f"Loaded. Shape: {img1.shape} (H × W × 13 bands)")
            except Exception as e:
                st.error(f"Failed to load images: {e}")

    img1 = st.session_state.get("_img1_preview")
    img2 = st.session_state.get("_img2_preview")

    if img1 is not None and img2 is not None:
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

        # Band stats expander
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
else:
    st.info("Enter valid T1 and T2 paths above, then click **Load & preview images**.")
