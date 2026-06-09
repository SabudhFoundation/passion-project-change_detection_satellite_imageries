"""Reusable metrics card for precision, recall, F1, kappa, IoU."""

import streamlit as st


def render_metrics_card(metrics: dict, title: str = "Evaluation Metrics"):
    """Display a styled metrics card from compute_metrics() output."""
    if not metrics:
        st.info("No metrics available. Run prediction with a ground-truth label to evaluate.")
        return

    st.subheader(title)

    col1, col2, col3 = st.columns(3)
    col1.metric("F1 Score", f"{metrics.get('f1', 0)*100:.2f}%")
    col2.metric("Precision", f"{metrics.get('precision', 0)*100:.2f}%")
    col3.metric("Recall", f"{metrics.get('recall', 0)*100:.2f}%")

    col4, col5, col6 = st.columns(3)
    col4.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.2f}%")
    col5.metric("Kappa", f"{metrics.get('kappa', 0):.4f}")
    col6.metric("IoU (Jaccard)", f"{metrics.get('jaccard', 0)*100:.2f}%")

    with st.expander("Confusion matrix values"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("TP", f"{metrics.get('tp', 0):,}")
        c2.metric("TN", f"{metrics.get('tn', 0):,}")
        c3.metric("FP", f"{metrics.get('fp', 0):,}")
        c4.metric("FN", f"{metrics.get('fn', 0):,}")
