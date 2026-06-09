# UCDNet Streamlit UI

Interactive web interface for the UCDNet urban change detection pipeline.

## Quick start

```bash
# From project root
pip install -r streamlit_app/requirements_streamlit.txt
streamlit run streamlit_app/app.py
```

## Pages

| Page | File | Purpose |
|------|------|---------|
| Home | `app.py` | Session status overview |
| Dashboard | `pages/1_Dashboard.py` | Training curves, metrics CSV, model status |
| Upload Images | `pages/2_Upload_Images.py` | Set T1/T2 paths, preview 13-band imagery |
| Run Detection | `pages/3_Run_Detection.py` | Sliding-window inference with `predict_pair()` |
| Results | `pages/4_Results.py` | Change map, probability map, side-by-side comparison, downloads |
| Train Model | `pages/5_Train_Model.py` | Launch training with configurable hyperparameters |

## Components

- `components/sidebar.py` — global paths, session summary, clear button  
- `components/band_selector.py` — Sentinel-2 13-band / RGB composite picker  
- `components/metrics_card.py` — F1, precision, recall, kappa, IoU cards  
- `components/map_viewer.py` — folium interactive map or matplotlib fallback  

## Utils

- `utils/helpers.py` — `@st.cache_resource` model loader, image normalisation,
  band → display RGB converter, `ensure_src_on_path()` for project imports

## Notes

- The app adds `src/` to `sys.path` automatically, so all project imports work.
- Set `UCDNET_DATA_ROOT` and `UCDNET_OUTPUT_DIR` env vars or use the sidebar inputs.
- The **interactive map** requires `folium` and `streamlit-folium`; a static image
  fallback is used when they are absent.
