"""Band selector for Sentinel-2 13-band imagery preview."""

import streamlit as st

SENTINEL2_BANDS = {
    "B01 — Coastal aerosol (443 nm)": 0,
    "B02 — Blue (490 nm)": 1,
    "B03 — Green (560 nm)": 2,
    "B04 — Red (665 nm)": 3,
    "B05 — Vegetation red edge (705 nm)": 4,
    "B06 — Vegetation red edge (740 nm)": 5,
    "B07 — Vegetation red edge (783 nm)": 6,
    "B08 — NIR (842 nm)": 7,
    "B8A — Narrow NIR (865 nm)": 8,
    "B09 — Water vapour (945 nm)": 9,
    "B10 — SWIR cirrus (1375 nm)": 10,
    "B11 — SWIR (1610 nm)": 11,
    "B12 — SWIR (2190 nm)": 12,
}

RGB_PRESETS = {
    "True colour (B04, B03, B02)": [3, 2, 1],
    "False colour / vegetation (B08, B04, B03)": [7, 3, 2],
    "SWIR composite (B12, B08, B04)": [12, 7, 3],
    "Agriculture (B11, B08, B02)": [11, 7, 1],
}


def render_band_selector(label: str = "Band / RGB composite"):
    """Returns (mode, selection) where mode is 'single' or 'rgb'."""
    st.markdown(f"**{label}**")
    mode = st.radio("View mode", ["Single band", "RGB composite"], horizontal=True, key=f"bmode_{label}")

    if mode == "Single band":
        band_name = st.selectbox("Select band", list(SENTINEL2_BANDS.keys()), key=f"bsel_{label}")
        return "single", SENTINEL2_BANDS[band_name]
    else:
        preset = st.selectbox("RGB preset", list(RGB_PRESETS.keys()), key=f"bpre_{label}")
        return "rgb", RGB_PRESETS[preset]
