"""Page 3 — Run Detection: configure and execute sliding-window inference."""

import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components.sidebar import render_sidebar
from utils.helpers import ensure_src_on_path, load_model_cached

render_sidebar()

st.title("🔍 Run Detection")
st.caption("Configure inference parameters and run the sliding-window change detector.")

# ── Pre-flight check ─────────────────────────────────────────────────────────
t1 = st.session_state.get("t1_path")
t2 = st.session_state.get("t2_path")
model_path = st.session_state.get("model_path")

if not t1 or not t2:
    st.warning("⚠️ T1 and T2 paths not set. Go to **Upload Images** first.")
    st.stop()

st.success(f"✅ T1: `{t1}`")
st.success(f"✅ T2: `{t2}`")

# ── Inference parameters ─────────────────────────────────────────────────────
st.subheader("Inference Parameters")

col1, col2, col3 = st.columns(3)
with col1:
    patch_size = st.number_input("Patch size (px)", min_value=64, max_value=1024,
                                  value=512, step=64)
with col2:
    overlap = st.number_input("Overlap (px)", min_value=0, max_value=256,
                               value=64, step=16)
with col3:
    batch_size = st.number_input("Batch size", min_value=1, max_value=16,
                                  value=4, step=1)

col4, col5 = st.columns(2)
with col4:
    threshold = st.slider("Change threshold", 0.0, 1.0, 0.5, 0.01)
with col5:
    normalize = st.selectbox("Normalisation", ["reflectance", "per_band"], index=0)

output_dir = Path(st.session_state.get("output_dir", "src/data/processed/artifacts"))
out_path = st.text_input(
    "Output path (.tif)",
    value=str(output_dir / "predictions" / "change_map.tif"),
)

mp_input = st.text_input(
    "Model path (.keras / .weights.h5)",
    value=model_path or str(output_dir / "best_model.weights.h5"),
)
if mp_input and Path(mp_input).is_file():
    st.session_state["model_path"] = mp_input
    model_path = mp_input
elif mp_input:
    st.error(f"Model not found: `{mp_input}`")

label_path = st.session_state.get("label_path")
if label_path:
    st.info(f"Ground-truth label: `{label_path}` → metrics will be computed.")

st.divider()

# ── Sliding window inference (local implementation) ──────────────────────────
def predict_sliding_window(model, img1, img2, patch_size=512, overlap=64, batch_size=4):
    h, w = img1.shape[:2]
    stride = max(1, patch_size - overlap)
    prob_acc = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)

    coords = [(y0, x0) for y0 in range(0, h, stride) for x0 in range(0, w, stride)]
    p1_batch, p2_batch, coord_batch = [], [], []

    def flush():
        nonlocal p1_batch, p2_batch, coord_batch
        if not p1_batch:
            return
        t1b = np.stack(p1_batch)
        t2b = np.stack(p2_batch)
        preds = model.predict({"T1": t1b, "T2": t2b}, verbose=0)
        for (y0, x0), pred in zip(coord_batch, preds):
            y2 = min(y0 + patch_size, h)
            x2 = min(x0 + patch_size, w)
            ph, pw = y2 - y0, x2 - x0
            prob_acc[y0:y2, x0:x2] += pred[:ph, :pw, 1]
            count[y0:y2, x0:x2] += 1.0
        p1_batch.clear(); p2_batch.clear(); coord_batch.clear()

    for y0, x0 in coords:
        y2 = min(y0 + patch_size, h)
        x2 = min(x0 + patch_size, w)
        p1 = img1[y0:y2, x0:x2]
        p2 = img2[y0:y2, x0:x2]
        ph = patch_size - p1.shape[0]
        pw = patch_size - p1.shape[1]
        if ph > 0 or pw > 0:
            p1 = np.pad(p1, ((0, ph), (0, pw), (0, 0)), mode="constant")
            p2 = np.pad(p2, ((0, ph), (0, pw), (0, 0)), mode="constant")
        p1_batch.append(p1); p2_batch.append(p2); coord_batch.append((y0, x0))
        if len(p1_batch) == batch_size:
            flush()
    flush()

    count = np.where(count == 0, 1.0, count)
    return (prob_acc / count).astype(np.float32)


# ── Run ──────────────────────────────────────────────────────────────────────
if st.button("▶ Run Change Detection", type="primary", use_container_width=True):
    if not model_path or not Path(model_path).is_file():
        st.error("Model file not found.")
        st.stop()

    ensure_src_on_path()
    progress = st.progress(0, text="Loading model…")

    try:
        model = load_model_cached(model_path)
        progress.progress(20, text="Model loaded. Loading images…")

        # Load images using read_bands
        from src.preprocessing_data.oscd_loader import read_bands
        img1 = read_bands(t1, subdir="")  # files are flat in temp dir
        img2 = read_bands(t2, subdir="")
        progress.progress(40, text=f"Images loaded {img1.shape}. Running inference…")

        prob_map = predict_sliding_window(
            model, img1, img2,
            patch_size=int(patch_size),
            overlap=int(overlap),
            batch_size=int(batch_size),
        )
        progress.progress(80, text="Inference done. Saving outputs…")

        change_map = (prob_map >= threshold).astype(np.uint8)
        changed_pct = 100.0 * change_map.sum() / change_map.size

        # Save outputs
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        try:
            import rasterio
            from PIL import Image
            prob_tif = out.with_name(out.stem + "_probability.tif")
            with rasterio.open(
                str(out), "w", driver="GTiff",
                height=change_map.shape[0], width=change_map.shape[1],
                count=1, dtype="uint8"
            ) as dst:
                dst.write(change_map.astype(np.uint8), 1)
            with rasterio.open(
                str(prob_tif), "w", driver="GTiff",
                height=prob_map.shape[0], width=prob_map.shape[1],
                count=1, dtype="float32"
            ) as dst:
                dst.write(prob_map, 1)
        except Exception:
            from PIL import Image
            Image.fromarray((change_map * 255).astype(np.uint8)).save(
                str(out.with_suffix(".png")))

        # Save overlay
        try:
            from PIL import Image
            grey = (prob_map * 200).clip(0, 200).astype(np.uint8)
            rgb = np.stack([grey, grey, grey], axis=2)
            rgb[change_map.astype(bool)] = [230, 30, 30]
            overlay_path = out.with_name(out.stem + "_overlay.png")
            Image.fromarray(rgb.astype(np.uint8)).save(str(overlay_path))
        except Exception:
            pass

        progress.progress(90, text="Computing metrics…")

        metrics = {}
        if label_path and Path(label_path).is_file():
            try:
                import rasterio
                from src.models.metrics import compute_metrics

                with rasterio.open(label_path) as src:
                    label_raw = src.read(1).astype(np.int32)

                # Handle both OSCD raw (1=no-change, 2=change)
                # and pre-binarised (0=no-change, 1=change)
                if label_raw.max() == 2:
                    label = np.where(label_raw == 2, 1, 0)
                else:
                    label = (label_raw > 0).astype(np.int32)

                # Resize label to match change_map if shapes differ
                if label.shape != change_map.shape:
                    from PIL import Image as PILImage
                    label_pil = PILImage.fromarray(label.astype(np.uint8))
                    label_pil = label_pil.resize(
                        (change_map.shape[1], change_map.shape[0]),
                        PILImage.NEAREST
                    )
                    label = np.array(label_pil).astype(np.int32)

                metrics = compute_metrics(label, change_map)

            except Exception as e:
                st.warning(f"Could not compute metrics: {e}")
                import traceback
                st.code(traceback.format_exc())

        st.session_state["prob_map"] = prob_map
        st.session_state["change_map"] = change_map
        st.session_state["metrics"] = metrics
        st.session_state["_out_path"] = out_path
        st.session_state["_threshold"] = threshold

        progress.progress(100, text="Done ✅")
        st.success(
            f"✅ Detection complete! Changed: {change_map.sum():,} px ({changed_pct:.2f}%). "
            f"Go to **Results** to view the change map."
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Changed pixels", f"{change_map.sum():,}")
        c2.metric("Changed area", f"{changed_pct:.2f}%")
        c3.metric("Image size", f"{change_map.shape[0]}×{change_map.shape[1]}")

        if metrics:
            st.subheader("Quick metrics")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("F1", f"{metrics['f1']*100:.2f}%")
            mc2.metric("Precision", f"{metrics['precision']*100:.2f}%")
            mc3.metric("Recall", f"{metrics['recall']*100:.2f}%")

    except Exception as e:
        progress.empty()
        st.error(f"Prediction failed: {e}")
        import traceback
        st.code(traceback.format_exc())