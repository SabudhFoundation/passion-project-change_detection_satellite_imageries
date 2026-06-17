"""Utility helpers: caching, session state, array → display image."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st


# ── sys.path helper ──────────────────────────────────────────────────────────
def ensure_src_on_path():
    """Add src/ to sys.path so project modules are importable."""
    project_root = Path(__file__).resolve().parents[2]
    src = project_root / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


# ── Model loader (cached so it loads only once) ──────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model_cached(model_path: str):
    ensure_src_on_path()
    from src.models.ucdnet_architecture import build_ucdnet
    model = build_ucdnet(input_shape=(512, 512, 13), num_classes=2)
    model.load_weights(model_path)
    return model


# ── Image normalisation for display ─────────────────────────────────────────
def bands_to_display(img: np.ndarray, band_indices: list[int]) -> np.ndarray:
    """Convert a (H, W, 13) float array to an (H, W, 3) uint8 RGB for st.image."""
    if len(band_indices) == 1:
        ch = img[:, :, band_indices[0]]
        ch = _percentile_stretch(ch)
        return np.stack([ch, ch, ch], axis=-1)
    else:
        rgb = np.stack([_percentile_stretch(img[:, :, b]) for b in band_indices], axis=-1)
        return rgb


def _percentile_stretch(arr: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    p_lo, p_hi = np.percentile(arr, [lo, hi])
    stretched = np.clip((arr - p_lo) / (p_hi - p_lo + 1e-9), 0, 1)
    return (stretched * 255).astype(np.uint8)


def prob_map_to_display(prob_map: np.ndarray) -> np.ndarray:
    """Convert float probability map to (H, W, 3) uint8 hot colormap."""
    import matplotlib.cm as cm
    cmap = cm.get_cmap("hot")
    rgba = cmap(prob_map)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def change_map_overlay(
    base_rgb: np.ndarray,
    change_map: np.ndarray,
    color: tuple = (230, 50, 50),
    alpha: float = 0.6,
) -> np.ndarray:
    """Blend change pixels onto a base RGB image."""
    out = base_rgb.copy().astype(np.float32)
    mask = change_map.astype(bool)
    overlay = np.array(color, dtype=np.float32)
    out[mask] = out[mask] * (1 - alpha) + overlay * alpha
    return out.clip(0, 255).astype(np.uint8)


# ── Load image pair using new read_bands API ─────────────────────────────────
@st.cache_data(show_spinner="Loading image pair…")
def load_image_pair_cached(t1_dir: str, t2_dir: str):
    """
    Load T1 and T2 images from directories containing individual band .tif files.
    Uses the new read_bands() function from oscd_loader.py.
    The uploaded files are saved flat in the temp dir (no subdir),
    so we pass the dir directly as band_dir.
    """
    ensure_src_on_path()
    import os
    import rasterio
    from rasterio.enums import Resampling

    BAND_FILES = [
        "B01.tif", "B02.tif", "B03.tif", "B04.tif",
        "B05.tif", "B06.tif", "B07.tif", "B08.tif",
        "B8A.tif", "B09.tif", "B10.tif", "B11.tif", "B12.tif"
    ]

    def read_dir(band_dir):
        # Get target size from B04
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
        img = np.clip(img, 0, 10000) / 10000.0
        return img

    img1 = read_dir(t1_dir)
    img2 = read_dir(t2_dir)
    return img1, img2


# ── Metrics CSV loader ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_metrics_csv(csv_path: str):
    import pandas as pd
    return pd.read_csv(csv_path)