"""Page 4 — Results: visualise change map, probability map, and download outputs."""

import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components.sidebar import render_sidebar
from components.metrics_card import render_metrics_card
from components.map_viewer import render_map_viewer, render_change_map_image
from components.band_selector import render_band_selector
from utils.helpers import bands_to_display, prob_map_to_display, change_map_overlay, load_image_pair_cached, ensure_src_on_path

render_sidebar()

st.title("🗺️ Results")
st.caption("Explore the change map, probability map, and side-by-side image comparison.")

change_map = st.session_state.get("change_map")
prob_map = st.session_state.get("prob_map")

if change_map is None:
    st.info("No prediction available. Go to **Run Detection** first.")
    st.stop()

# ── Summary numbers ──────────────────────────────────────────────────────────
changed_pct = 100.0 * change_map.sum() / change_map.size
c1, c2, c3, c4 = st.columns(4)
c1.metric("Changed pixels", f"{change_map.sum():,}")
c2.metric("Unchanged pixels", f"{(change_map == 0).sum():,}")
c3.metric("Changed area", f"{changed_pct:.2f}%")
c4.metric("Image size", f"{change_map.shape[0]}×{change_map.shape[1]}")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Change Map", "Probability Map", "Side-by-Side", "Metrics"])

with tab1:
    st.subheader("Binary change map")
    out_path = st.session_state.get("_out_path")

    view_mode = st.radio("View", ["Static image", "Interactive map"], horizontal=True)

    if view_mode == "Interactive map":
        render_map_viewer(change_map, prob_map, geotiff_path=out_path)
    else:
        render_change_map_image(change_map, prob_map)

    # Download
    if out_path and Path(out_path).is_file():
        with open(out_path, "rb") as f:
            st.download_button("⬇ Download change map (.tif)", f.read(),
                               file_name="change_map.tif", mime="image/tiff")

    base = Path(out_path).with_suffix("") if out_path else None
    overlay_path = Path(f"{base}_overlay.png") if base else None
    if overlay_path and overlay_path.is_file():
        with open(overlay_path, "rb") as f:
            st.download_button("⬇ Download overlay (.png)", f.read(),
                               file_name="change_map_overlay.png", mime="image/png")

with tab2:
    st.subheader("Probability map")
    if prob_map is not None:
        prob_disp = prob_map_to_display(prob_map)
        st.image(prob_disp, caption="Change probability (hot colormap: white=high, black=low)",
                 use_column_width=True)

        # Histogram
        import plotly.express as px
        fig = px.histogram(x=prob_map.ravel(), nbins=100,
                           title="Probability distribution",
                           labels={"x": "Probability", "y": "Pixel count"})
        fig.add_vline(x=st.session_state.get("_threshold", 0.5),
                      line_dash="dash", line_color="red",
                      annotation_text="threshold")
        st.plotly_chart(fig, use_container_width=True)

        base = Path(out_path).with_suffix("") if out_path else None
        prob_path = Path(f"{base}_probability.tif") if base else None
        if prob_path and prob_path.is_file():
            with open(prob_path, "rb") as f:
                st.download_button("⬇ Download probability map (.tif)", f.read(),
                                   file_name="probability_map.tif", mime="image/tiff")
    else:
        st.info("No probability map in session.")

with tab3:
    st.subheader("T1 / T2 side-by-side comparison")
    t1 = st.session_state.get("t1_path")
    t2 = st.session_state.get("t2_path")

    if t1 and t2:
        normalize = st.session_state.get("normalize", "reflectance")
        try:
            ensure_src_on_path()
            img1, img2 = load_image_pair_cached(t1, t2, normalize=normalize)

            col_l, col_r = st.columns(2)
            with col_l:
                mode1, sel1 = render_band_selector("T1 display")
                disp1 = bands_to_display(img1, sel1 if isinstance(sel1, list) else [sel1])
                st.image(disp1, caption="T1 (before)", use_column_width=True)

            with col_r:
                mode2, sel2 = render_band_selector("T2 display")
                disp2 = bands_to_display(img2, sel2 if isinstance(sel2, list) else [sel2])
                # Overlay change pixels on T2
                overlay = change_map_overlay(disp2, change_map)
                st.image(overlay, caption="T2 (after) — changed areas in red", use_column_width=True)

        except Exception as e:
            st.error(f"Could not load images for comparison: {e}")
    else:
        st.info("Image paths not set. Go to **Upload Images** first.")

with tab4:
    render_metrics_card(st.session_state.get("metrics", {}))
