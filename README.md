# Optiver Vol Forecaster — Streamlit App

Regime-conditioned Graph Attention volatility forecasting dashboard.

## File structure

```
optiver_app/
├── app.py
├── requirements.txt
└── artifacts/
    ├── regime_0_best.pth
    ├── regime_1_best.pth
    ├── regime_2_best.pth
    ├── hmm_model.pkl
    ├── hmm_scaler.pkl
    ├── scaler.pkl
    ├── edge_index.pt
    ├── node_feats.pt
    ├── stock_to_idx.json
    ├── feature_columns.json
    └── regime_stats.csv
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repo
2. Go to share.streamlit.io → New app
3. Point to `app.py`
4. Streamlit Cloud will auto-install `requirements.txt`

> Note: Streamlit Community Cloud has no GPU. All inference runs on CPU.
> torch-geometric on CPU does not need CUDA — it works out of the box.

## Pages

| Page | Description |
|---|---|
| Market Overview | Live slider-based volatility forecast with gauge + feature bars |
| Regime Deep-Dive | HMM transition matrix, radar chart, regime statistics table |
| Cross-Asset Spillover | Stock graph (pyvis), neighbour bars, node feature heatmap |
| Model Benchmarking | RMSPE comparison vs GARCH, residual histograms, architecture card |
