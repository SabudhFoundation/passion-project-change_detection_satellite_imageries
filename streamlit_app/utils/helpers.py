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
    # Also add project root (for config.py at src level)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


# ── Model loader (cached so it loads only once) ──────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model_cached(model_path: str):
    ensure_src_on_path()
    from models.predict_model import load_model
    return load_model(model_path)


# ── Settings loader ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_settings_cached(**overrides):
    ensure_src_on_path()
    from config import load_settings
    return load_settings(**overrides)


# ── Image normalisation for display ─────────────────────────────────────────
def bands_to_display(img: np.ndarray, band_indices: list[int]) -> np.ndarray:
    """
    Convert a (H, W, 13) float array to an (H, W, 3) uint8 RGB for st.image.
    band_indices: list of 1 or 3 band indices.
    """
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
    import matplotlib.pyplot as plt
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


# ── Load image pair helper ───────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading image pair…")
def load_image_pair_cached(t1_path: str, t2_path: str, normalize: str = "reflectance"):
    ensure_src_on_path()
    from preprocessing_data.oscd_loader import load_image_pair
    return load_image_pair(t1_path, t2_path, normalize=normalize)


# ── Metrics CSV loader ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_metrics_csv(csv_path: str):
    import pandas as pd
    return pd.read_csv(csv_path)
