"""Sidebar: navigation info + global config paths."""

import streamlit as st
from pathlib import Path


def render_sidebar():
    with st.sidebar:
        st.markdown("## 🛰️ UCDNet")
        st.caption("Urban Change Detection · Sentinel-2 · OSCD")
        st.divider()

        st.markdown("### 📁 Paths")

        # Output directory
        out_dir = st.text_input(
            "Output directory",
            value=st.session_state.get("output_dir", "src/data/processed/artifacts"),
            help="Where models, metrics, and predictions are saved.",
        )
        st.session_state["output_dir"] = out_dir

        # Model path
        model_input = st.text_input(
            "Model path (.keras)",
            value=st.session_state.get("model_path") or "",
            placeholder="src/data/processed/artifacts/best_model.keras",
            help="Path to a trained best_model.keras file.",
        )
        if model_input and Path(model_input).is_file():
            st.session_state["model_path"] = model_input
            st.success("Model found ✅")
        elif model_input:
            st.error("File not found")

        st.divider()
        st.markdown("### 📊 Session")

        s = st.session_state
        st.write(f"- T1: `{Path(s.t1_path).name if s.t1_path else '—'}`")
        st.write(f"- T2: `{Path(s.t2_path).name if s.t2_path else '—'}`")
        st.write(f"- Prediction: {'done ✅' if s.change_map is not None else '—'}")
        if s.metrics:
            st.write(f"- F1: `{s.metrics.get('f1', 0)*100:.1f}%`")

        st.divider()
        if st.button("🗑️ Clear session", use_container_width=True):
            for key in ["t1_path", "t2_path", "prob_map", "change_map", "metrics"]:
                st.session_state[key] = None if key != "metrics" else {}
            st.rerun()

        st.caption("UCDNet · IEEE TGRS 2022")
