"""Folium-based map viewer for change maps with satellite tile overlay."""

from __future__ import annotations

import numpy as np
import streamlit as st

try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def render_change_map_image(change_map: np.ndarray, prob_map: np.ndarray | None = None):
    """Fallback: display change map as a plain image when folium is unavailable."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    fig, axes = plt.subplots(1, 2 if prob_map is not None else 1, figsize=(12, 5))
    if prob_map is None:
        axes = [axes]

    axes[0].imshow(change_map, cmap="RdYlGn_r", vmin=0, vmax=1)
    axes[0].set_title("Change map (binary)")
    axes[0].axis("off")

    if prob_map is not None:
        im = axes[1].imshow(prob_map, cmap="hot", vmin=0, vmax=1)
        axes[1].set_title("Probability map")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_map_viewer(
    change_map: np.ndarray,
    prob_map: np.ndarray | None = None,
    geotiff_path: str | None = None,
):
    """Render change map — with folium if available, plain image fallback otherwise."""
    if not HAS_FOLIUM:
        st.warning("Install `folium` and `streamlit-folium` for interactive map view. Showing static image.")
        render_change_map_image(change_map, prob_map)
        return

    # Try to get bounds from GeoTIFF
    bounds = None
    center = [0, 0]
    if geotiff_path and HAS_RASTERIO:
        try:
            with rasterio.open(geotiff_path) as src:
                b = transform_bounds(src.crs, CRS.from_epsg(4326), *src.bounds)
                bounds = [[b[1], b[0]], [b[3], b[2]]]
                center = [(b[1] + b[3]) / 2, (b[0] + b[2]) / 2]
        except Exception:
            pass

    m = folium.Map(location=center, zoom_start=13, tiles="CartoDB dark_matter")

    # Overlay change map as image
    try:
        import matplotlib.pyplot as plt
        import io, base64
        from PIL import Image

        rgba = np.zeros((*change_map.shape, 4), dtype=np.uint8)
        rgba[change_map == 1] = [230, 50, 50, 200]
        rgba[change_map == 0] = [0, 0, 0, 60]
        pil_img = Image.fromarray(rgba, "RGBA")
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        if bounds:
            folium.raster_layers.ImageOverlay(
                image=f"data:image/png;base64,{b64}",
                bounds=bounds,
                opacity=0.7,
                name="Change map",
            ).add_to(m)
        folium.LayerControl().add_to(m)
    except Exception as e:
        st.warning(f"Could not render overlay: {e}")

    st_folium(m, width=700, height=450)
