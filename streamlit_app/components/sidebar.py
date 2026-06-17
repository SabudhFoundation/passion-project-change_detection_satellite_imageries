"""Sidebar: navigation info + model upload + global config."""

import tempfile
import streamlit as st
from pathlib import Path


def render_sidebar():
    with st.sidebar:
        st.markdown("## 🛰️ UCDNet")
        st.caption("Urban Change Detection · Sentinel-2 · OSCD")
        st.divider()

        st.markdown("### 🤖 Model")

        # If we already have a model loaded, show it with option to change
        if st.session_state.get("_model_file_name") and st.session_state.get("_model_bytes"):
            st.success(f"✅ {st.session_state['_model_file_name']}")
            if st.button("🔄 Change model", use_container_width=True):
                st.session_state["_model_bytes"] = None
                st.session_state["_model_file_name"] = None
                st.session_state["model_path"] = None
                st.rerun()
        else:
            # Show uploader only when no model is loaded
            model_file = st.file_uploader(
                "Upload model weights",
                type=["h5", "keras"],
                key="model_uploader",
                help="Upload your model.weights.h5 or best_model.keras file",
            )

            if model_file:
                file_bytes = model_file.read()
                st.session_state["_model_bytes"] = file_bytes
                st.session_state["_model_file_name"] = model_file.name

                tmp_model = Path(tempfile.gettempdir()) / f"ucdnet_{model_file.name}"
                tmp_model.write_bytes(file_bytes)
                st.session_state["model_path"] = str(tmp_model)
                st.rerun()  # re-render to switch to "loaded" view

        # Recreate temp file if OS wiped /tmp (e.g. long session)
        if st.session_state.get("_model_bytes") and st.session_state.get("model_path"):
            tmp = Path(st.session_state["model_path"])
            if not tmp.exists():
                tmp.write_bytes(st.session_state["_model_bytes"])

        # Fallback: manual path input
        with st.expander("Or enter path manually"):
            model_input = st.text_input(
                "Model path",
                value=st.session_state.get("model_path") or "",
                placeholder="src/data/processed/artifacts/best_model.keras",
            )
            if model_input and Path(model_input).is_file():
                st.session_state["model_path"] = model_input
                st.success("Model found ✅")
            elif model_input:
                st.error("File not found")

        st.divider()

        # Output directory
        st.markdown("### 📁 Output")
        out_dir = st.text_input(
            "Output directory",
            value=st.session_state.get("output_dir", "src/data/processed/artifacts"),
        )
        st.session_state["output_dir"] = out_dir

        st.divider()
        st.markdown("### 📊 Session")

        s = st.session_state
        st.write(f"- T1: `{Path(s.t1_path).name if s.t1_path else '—'}`")
        st.write(f"- T2: `{Path(s.t2_path).name if s.t2_path else '—'}`")
        st.write(f"- Model: `{Path(s.model_path).name if s.model_path else '—'}`")
        st.write(f"- Prediction: {'done ✅' if s.change_map is not None else '—'}")
        if s.metrics:
            st.write(f"- F1: `{s.metrics.get('f1', 0)*100:.1f}%`")

        st.divider()
        if st.button("🗑️ Clear session", use_container_width=True):
            for key in ["t1_path", "t2_path", "prob_map", "change_map",
                        "metrics", "_img1_preview", "_img2_preview",
                        "_model_bytes", "_model_file_name", "model_path"]:
                st.session_state[key] = None if key != "metrics" else {}
            st.rerun()

        st.caption("UCDNet · IEEE TGRS 2022")