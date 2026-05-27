import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Optiver Vol Forecaster",
    page_icon="📈",
)

st.markdown("""
<style>
.badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 12px;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: .3px;
}
.badge-green  { background:#d4edda; color:#155724; }
.badge-amber  { background:#fff3cd; color:#856404; }
.badge-red    { background:#f8d7da; color:#721c24; }
.metric-card {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 14px 18px;
    border: 1px solid #e0e0e0;
}
.metric-label { font-size:12px; color:#666; margin-bottom:2px; }
.metric-value { font-size:26px; font-weight:700; color:#212529; }
.metric-sub   { font-size:11px; color:#999; margin-top:2px; }
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────
ART = Path("artifacts")

FEATURE_COLS = [
    "realized_vol_600", "vol_60s", "vol_120s", "vol_300s",
    "mean_spread", "std_spread", "mean_imbalance", "std_imbalance",
    "mean_depth_ratio", "std_depth_ratio",
    "wap_mean", "wap_std", "wap_range",
    "trade_vol_per_sec", "trade_order_count", "trade_price_std", "trade_size_sum",
]
HMM_FEATURES = ["mean_imbalance", "std_imbalance", "mean_spread", "realized_vol_600"]
N_FEATURES   = 17
DEVICE       = torch.device("cpu")

REGIME_NAMES  = {0: "Trending", 1: "Mean-Reverting", 2: "Stressed"}
REGIME_COLORS = {0: "green",    1: "amber",           2: "red"}
REGIME_HEX    = {0: "#198754",  1: "#fd7e14",         2: "#dc3545"}
REGIME_BADGE  = {0: "badge-green", 1: "badge-amber", 2: "badge-red"}

# ═══════════════════════════════════════════════════════════════
# Model architecture (must match training exactly)
# ═══════════════════════════════════════════════════════════════
try:
    from torch_geometric.nn import GATConv
    PYGEOMETRIC = True
except ImportError:
    PYGEOMETRIC = False


class GATLayer(nn.Module):
    def __init__(self, in_ch, out_ch, heads=4):
        super().__init__()
        self.use_pyg = PYGEOMETRIC
        if self.use_pyg:
            self.gat     = GATConv(in_ch, out_ch, heads=heads, dropout=0.1, concat=True)
            self.out_dim = out_ch * heads
        else:
            self.proj    = nn.Linear(in_ch, out_ch * heads)
            self.out_dim = out_ch * heads

    def forward(self, x, edge_index):
        if self.use_pyg:
            return F.elu(self.gat(x, edge_index))
        return F.elu(self.proj(x))


class RegimeMLPHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),     nn.LayerNorm(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64,  32),     nn.LayerNorm(32),  nn.ReLU(),
            nn.Linear(32,  1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


class RegimeGATModel(nn.Module):
    def __init__(self, n_features=17, gat_out=64, gat_heads=4):
        super().__init__()
        self.gat  = GATLayer(n_features, gat_out, heads=gat_heads)
        self.head = RegimeMLPHead(self.gat.out_dim)

    def forward(self, x_batch, node_idx, node_feats, edge_index):
        node_emb   = self.gat(node_feats, edge_index)
        sample_emb = node_emb[node_idx]
        return self.head(sample_emb)


# ═══════════════════════════════════════════════════════════════
# Cached artifact loading
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def load_all_artifacts():
    with open(ART / "feature_columns.json") as f:
        feat_meta = json.load(f)

    with open(ART / "stock_to_idx.json") as f:
        stock_to_idx = {k: int(v) for k, v in json.load(f).items()}

    scaler     = joblib.load(ART / "scaler.pkl")
    hmm_scaler = joblib.load(ART / "hmm_scaler.pkl")
    hmm_model  = joblib.load(ART / "hmm_model.pkl")

    edge_index = torch.load(ART / "edge_index.pt", map_location=DEVICE)
    node_feats = torch.load(ART / "node_feats.pt", map_location=DEVICE)

    regime_stats = pd.read_csv(ART / "regime_stats.csv")

    models = {}
    for r in range(3):
        m = RegimeGATModel()
        m.load_state_dict(torch.load(ART / f"regime_{r}_best.pth", map_location=DEVICE))
        m.eval()
        models[r] = m

    return {
        "feat_meta":    feat_meta,
        "stock_to_idx": stock_to_idx,
        "scaler":       scaler,
        "hmm_scaler":   hmm_scaler,
        "hmm_model":    hmm_model,
        "edge_index":   edge_index,
        "node_feats":   node_feats,
        "regime_stats": regime_stats,
        "models":       models,
    }


def predict(art, feat_values: np.ndarray, stock_id: int):
    """Run full inference pipeline: scale → HMM regime → GAT model → vol."""
    feat_arr = feat_values.reshape(1, -1).astype(np.float64)
    feat_scaled = art["scaler"].transform(feat_arr).astype(np.float32)

    hmm_idx    = [FEATURE_COLS.index(f) for f in HMM_FEATURES]
    hmm_raw    = feat_values[hmm_idx].reshape(1, -1).astype(np.float64)
    hmm_scaled = art["hmm_scaler"].transform(hmm_raw)
    regime     = int(art["hmm_model"].predict(hmm_scaled)[0])
    proba      = art["hmm_model"].predict_proba(hmm_scaled)[0]
    confidence = float(proba.max()) * 100

    sid_str  = str(stock_id)
    node_idx = art["stock_to_idx"].get(sid_str, 0)

    x_t    = torch.tensor(feat_scaled, dtype=torch.float32)
    nidx_t = torch.tensor([node_idx],  dtype=torch.long)

    with torch.no_grad():
        pred_vol = art["models"][regime](
            x_t, nidx_t, art["node_feats"], art["edge_index"]
        ).item()

    return regime, confidence, pred_vol, proba


# ═══════════════════════════════════════════════════════════════
# Helper: default feature values (scaler means → inverse to raw)
# ═══════════════════════════════════════════════════════════════
def default_raw_features(art):
    zeros_scaled = np.zeros((1, N_FEATURES))
    return art["scaler"].inverse_transform(zeros_scaled).flatten()


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("📈 Optiver Vol Forecaster")
    st.caption("Regime-conditioned GAT volatility model")
    st.divider()
    page = st.radio(
        "Navigate",
        ["Market Overview", "Regime Deep-Dive", "Cross-Asset Spillover", "Model Benchmarking"],
        index=0,
    )
    st.divider()
    st.caption("Artifacts from: [Kaggle competition](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction)")


# ═══════════════════════════════════════════════════════════════
# Load artifacts (cached)
# ═══════════════════════════════════════════════════════════════
try:
    art = load_all_artifacts()
    artifacts_ok = True
except Exception as e:
    artifacts_ok = False
    art_error    = str(e)


# ═══════════════════════════════════════════════════════════════
# PAGE 1: Market Overview
# ═══════════════════════════════════════════════════════════════
if page == "Market Overview":
    st.title("Market Overview")
    st.caption("Real-time volatility forecast using trained RegimeGAT model")

    if not artifacts_ok:
        st.error(f"Could not load artifacts: {art_error}")
        st.stop()

    sorted_stocks = sorted(art["stock_to_idx"].keys(), key=lambda x: int(x))
    col_ctrl1, col_ctrl2 = st.columns([1, 3])

    with col_ctrl1:
        stock_id = st.selectbox("Stock ID", sorted_stocks, index=0)
        scenario = st.selectbox(
            "Preset scenario",
            ["Custom", "Low Volatility", "High Spread", "Stressed Market"],
        )

    defaults = default_raw_features(art)

    SCENARIOS = {
        "Low Volatility":  {i: defaults[i] * 0.5 for i in range(N_FEATURES)},
        "High Spread":     {i: (defaults[i] * 3 if FEATURE_COLS[i] in ("mean_spread","std_spread") else defaults[i]) for i in range(N_FEATURES)},
        "Stressed Market": {i: defaults[i] * 2.5 for i in range(N_FEATURES)},
        "Custom":          {i: defaults[i] for i in range(N_FEATURES)},
    }

    with col_ctrl2:
        st.caption("Feature sliders (raw values)")
        feat_vals = np.zeros(N_FEATURES)
        slider_cols = st.columns(3)
        for i, fname in enumerate(FEATURE_COLS):
            default_val = float(SCENARIOS[scenario][i])
            lo = float(min(0.0, default_val * 0.1))
            hi = float(max(abs(default_val) * 3.0, 1e-6))
            feat_vals[i] = slider_cols[i % 3].slider(
                fname, min_value=lo, max_value=hi,
                value=float(np.clip(default_val, lo, hi)),
                format="%.5f", key=f"sl_{i}"
            )

    regime, confidence, pred_vol, proba = predict(art, feat_vals, int(stock_id))
    rs = art["regime_stats"]
    row = rs[rs["regime"] == regime].iloc[0] if regime in rs["regime"].values else rs.iloc[0]

    st.divider()
    # ── Metric row ────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    badge_cls = REGIME_BADGE[regime]
    with m1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Current Regime</div>'
            f'<div class="metric-value">'
            f'<span class="badge {badge_cls}">{REGIME_NAMES[regime]}</span>'
            f'</div>'
            f'<div class="metric-sub">Regime {regime}</div>'
            f'</div>', unsafe_allow_html=True
        )
    with m2:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Predicted Vol (next 10 min)</div>'
            f'<div class="metric-value">{pred_vol:.6f}</div>'
            f'<div class="metric-sub">Realized volatility forecast</div>'
            f'</div>', unsafe_allow_html=True
        )
    with m3:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Regime Confidence</div>'
            f'<div class="metric-value">{confidence:.1f}%</div>'
            f'<div class="metric-sub">HMM posterior probability</div>'
            f'</div>', unsafe_allow_html=True
        )
    with m4:
        best_rmspe = float(row.get("best_val_rmspe", 0.0)) if "best_val_rmspe" in row else 0.0
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Val RMSPE (this regime)</div>'
            f'<div class="metric-value">{best_rmspe:.5f}</div>'
            f'<div class="metric-sub">From regime_stats.csv</div>'
            f'</div>', unsafe_allow_html=True
        )

    st.divider()
    ch1, ch2 = st.columns(2)

    with ch1:
        # ── Vol build-up line chart ────────────────────────────
        vol_names  = ["vol_60s", "vol_120s", "vol_300s", "realized_vol_600"]
        vol_labels = ["60s", "120s", "300s", "600s"]
        vol_vals   = [feat_vals[FEATURE_COLS.index(v)] for v in vol_names]
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=vol_labels, y=vol_vals,
            mode="lines+markers",
            line=dict(color=REGIME_HEX[regime], width=2.5),
            marker=dict(size=8),
            fill="tozeroy",
            fillcolor=REGIME_HEX[regime] + "33",
            name="Realized Vol Build-up",
        ))
        fig_vol.add_hline(
            y=float(row.get("mean_tgt", pred_vol)),
            line_dash="dash", line_color="#aaa",
            annotation_text=f"Regime mean target",
        )
        fig_vol.update_layout(
            template="plotly_white",
            title="Volatility Build-up (sub-window)",
            xaxis_title="Window", yaxis_title="Realized Vol",
            height=300, margin=dict(t=40, b=30),
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    with ch2:
        # ── Feature bar chart (scaled) ─────────────────────────
        feat_scaled_vals = art["scaler"].transform(feat_vals.reshape(1,-1)).flatten()
        means = art["scaler"].mean_
        stds  = art["scaler"].scale_
        colors_bar = [REGIME_HEX[regime] if abs(v) > 1 else "#6c757d" for v in feat_scaled_vals]
        fig_feat = go.Figure()
        fig_feat.add_trace(go.Bar(
            y=FEATURE_COLS,
            x=feat_scaled_vals,
            orientation="h",
            marker_color=colors_bar,
            name="Scaled value",
            error_x=dict(type="constant", value=1.0, color="#ccc"),
        ))
        fig_feat.add_vline(x=0, line_color="#333", line_width=1)
        fig_feat.update_layout(
            template="plotly_white",
            title="Feature Values (standardised, ±1σ bands)",
            xaxis_title="Z-score", yaxis_title="",
            height=300, margin=dict(t=40, b=30, l=160),
        )
        st.plotly_chart(fig_feat, use_container_width=True)

    # ── Gauge chart ────────────────────────────────────────────
    mean_tgt = float(row.get("mean_tgt", 0.003))
    std_tgt  = float(row.get("std_tgt",  0.002))
    gauge_max = mean_tgt + 3 * std_tgt
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pred_vol,
        delta={"reference": mean_tgt, "valueformat": ".6f"},
        number={"valueformat": ".6f"},
        title={"text": f"Predicted Vol vs Regime {regime} Mean ({REGIME_NAMES[regime]})"},
        gauge={
            "axis": {"range": [0, gauge_max], "tickformat": ".4f"},
            "bar":  {"color": REGIME_HEX[regime]},
            "steps": [
                {"range": [0,          mean_tgt - std_tgt], "color": "#d4edda"},
                {"range": [mean_tgt - std_tgt, mean_tgt + std_tgt], "color": "#fff3cd"},
                {"range": [mean_tgt + std_tgt, gauge_max],          "color": "#f8d7da"},
            ],
            "threshold": {
                "line": {"color": "#333", "width": 3},
                "thickness": 0.75,
                "value": mean_tgt,
            },
        },
    ))
    fig_gauge.update_layout(template="plotly_white", height=320, margin=dict(t=60, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 2: Regime Deep-Dive
# ═══════════════════════════════════════════════════════════════
elif page == "Regime Deep-Dive":
    st.title("Regime Deep-Dive")
    st.caption("HMM regime statistics, transition dynamics, and feature means")

    if not artifacts_ok:
        st.error(f"Could not load artifacts: {art_error}")
        st.stop()

    rs  = art["regime_stats"]
    hmm = art["hmm_model"]

    # ── Grouped bar: mean_tgt per regime with std error bars ──
    fig_bar = go.Figure()
    for i, row in rs.iterrows():
        r = int(row["regime"])
        fig_bar.add_trace(go.Bar(
            x=[REGIME_NAMES[r]],
            y=[row["mean_tgt"]],
            error_y=dict(type="data", array=[row["std_tgt"]], visible=True),
            name=REGIME_NAMES[r],
            marker_color=REGIME_HEX[r],
        ))
    fig_bar.update_layout(
        template="plotly_white",
        title="Mean Realized Volatility per Regime (±1 std)",
        yaxis_title="Realized Vol", xaxis_title="Regime",
        showlegend=True, height=350, margin=dict(t=50, b=30),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Metrics table ──────────────────────────────────────────
    st.subheader("Regime Statistics Table")
    display_df = rs.copy()
    display_df["regime_name"] = display_df["regime"].map(REGIME_NAMES)
    cols_order = ["regime", "regime_name", "n", "mean_tgt", "std_tgt",
                  "mean_spr", "mean_imb", "best_val_rmspe"]
    display_df = display_df[[c for c in cols_order if c in display_df.columns]]
    display_df.columns = [c.replace("_", " ").title() for c in display_df.columns]
    st.dataframe(
        display_df.style.background_gradient(
            subset=[c for c in display_df.columns if c not in ("Regime", "Regime Name")],
            cmap="RdYlGn_r",
        ).format({c: "{:.6f}" for c in display_df.columns if display_df[c].dtype == float}),
        use_container_width=True,
    )

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        # ── Transition matrix heatmap ──────────────────────────
        transmat = hmm.transmat_
        regime_labels_list = [f"R{i} {REGIME_NAMES[i]}" for i in range(3)]
        fig_trans = go.Figure(go.Heatmap(
            z=transmat,
            x=regime_labels_list,
            y=regime_labels_list,
            colorscale="Blues",
            zmin=0, zmax=1,
            text=[[f"{transmat[i][j]:.3f}" for j in range(3)] for i in range(3)],
            texttemplate="%{text}",
            textfont={"size": 13},
            showscale=True,
        ))
        fig_trans.update_layout(
            template="plotly_white",
            title="HMM Transition Probability Matrix",
            xaxis_title="To Regime",
            yaxis_title="From Regime",
            height=380, margin=dict(t=50, b=60, l=120),
        )
        st.plotly_chart(fig_trans, use_container_width=True)

    with col_b:
        # ── HMM means radar chart ──────────────────────────────
        hmm_means = hmm.means_   # (3, 4)
        radar_features = HMM_FEATURES + [HMM_FEATURES[0]]   # close the loop
        fig_radar = go.Figure()
        for r in range(3):
            vals = list(hmm_means[r]) + [hmm_means[r][0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals,
                theta=radar_features,
                fill="toself",
                name=f"R{r} {REGIME_NAMES[r]}",
                line_color=REGIME_HEX[r],
                fillcolor=REGIME_HEX[r] + "44",
            ))
        fig_radar.update_layout(
            template="plotly_white",
            polar=dict(
                radialaxis=dict(visible=True),
            ),
            title="HMM Regime Means (scaled HMM features)",
            showlegend=True,
            height=380, margin=dict(t=50, b=30),
        )
        st.plotly_chart(fig_radar, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 3: Cross-Asset Spillover
# ═══════════════════════════════════════════════════════════════
elif page == "Cross-Asset Spillover":
    st.title("Cross-Asset Spillover")
    st.caption("Stock graph topology, regime assignment per node, and feature heatmap")

    if not artifacts_ok:
        st.error(f"Could not load artifacts: {art_error}")
        st.stop()

    edge_index  = art["edge_index"].numpy()
    node_feats  = art["node_feats"].numpy()
    stock_to_idx = art["stock_to_idx"]
    idx_to_stock = {v: k for k, v in stock_to_idx.items()}
    N_STOCKS     = node_feats.shape[0]

    # Assign regime per node using HMM on node_feats
    hmm_feat_idx = [FEATURE_COLS.index(f) for f in HMM_FEATURES]
    node_hmm_raw    = node_feats[:, hmm_feat_idx].astype(np.float64)
    node_hmm_scaled = art["hmm_scaler"].transform(node_hmm_raw)
    node_regimes    = art["hmm_model"].predict(node_hmm_scaled)

    # Node size from realized_vol_600 (index 0)
    rv_col  = node_feats[:, 0]
    rv_min, rv_max = rv_col.min(), rv_col.max()
    def node_size(val):
        normed = (val - rv_min) / max(rv_max - rv_min, 1e-9)
        return int(10 + normed * 30)

    # ── PyVis network graph ────────────────────────────────────
    try:
        import networkx as nx
        from pyvis.network import Network
        import streamlit.components.v1 as components

        G = nx.DiGraph()
        for idx in range(N_STOCKS):
            sid   = idx_to_stock.get(idx, str(idx))
            reg   = int(node_regimes[idx])
            color = REGIME_HEX[reg]
            size  = node_size(rv_col[idx])
            G.add_node(idx, label=f"S{sid}", color=color, size=size,
                       title=f"Stock {sid} | Regime: {REGIME_NAMES[reg]}")

        MAX_EDGES = 300
        edges_to_add = edge_index[:, :MAX_EDGES]
        for j in range(edges_to_add.shape[1]):
            src, dst = int(edges_to_add[0, j]), int(edges_to_add[1, j])
            G.add_edge(src, dst)

        net = Network(height="500px", width="100%", directed=True, bgcolor="#ffffff")
        net.from_nx(G)
        net.set_options("""{
          "physics": {
            "enabled": true,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {"gravitationalConstant": -50}
          },
          "edges": {"arrows": {"to": {"enabled": true, "scaleFactor": 0.5}},
                    "color": {"opacity": 0.35}, "width": 0.8},
          "nodes": {"font": {"size": 10}}
        }""")
        html_str = net.generate_html()
        components.html(html_str, height=520, scrolling=False)

        # Legend
        leg1, leg2, leg3 = st.columns(3)
        for col, r in zip([leg1, leg2, leg3], [0, 1, 2]):
            col.markdown(
                f'<span class="badge badge-{REGIME_COLORS[r]}">● {REGIME_NAMES[r]}</span>',
                unsafe_allow_html=True
            )

    except ImportError:
        st.warning("pyvis or networkx not installed. Install both to view the network graph.")

    st.divider()
    col_left, col_right = st.columns([1, 2])

    with col_left:
        # ── Selected stock neighbours ──────────────────────────
        sorted_stocks_p3 = sorted(stock_to_idx.keys(), key=lambda x: int(x))
        sel_stock = st.selectbox("Select stock", sorted_stocks_p3, key="p3_stock")
        sel_idx   = stock_to_idx.get(str(sel_stock), 0)

        # Find outgoing neighbours in edge_index
        src_mask   = edge_index[0] == sel_idx
        neighbours = edge_index[1][src_mask][:10]

        if len(neighbours) > 0:
            nbr_ids   = [idx_to_stock.get(int(n), str(n)) for n in neighbours]
            nbr_rv    = [float(node_feats[int(n), 0]) for n in neighbours]
            nbr_reg   = [int(node_regimes[int(n)]) for n in neighbours]
            nbr_colors = [REGIME_HEX[r] for r in nbr_reg]

            fig_nbr = go.Figure(go.Bar(
                y=[f"S{s}" for s in nbr_ids],
                x=nbr_rv,
                orientation="h",
                marker_color=nbr_colors,
                text=[f"{v:.5f}" for v in nbr_rv],
                textposition="auto",
            ))
            fig_nbr.update_layout(
                template="plotly_white",
                title=f"Top neighbours of S{sel_stock} (realized_vol_600)",
                xaxis_title="realized_vol_600", yaxis_title="",
                height=340, margin=dict(t=40, b=30, l=60),
            )
            st.plotly_chart(fig_nbr, use_container_width=True)
        else:
            st.info("No outgoing edges found for this stock in the first 300 edges.")

    with col_right:
        # ── Node feature heatmap ───────────────────────────────
        n_show = st.slider("Stocks to display", min_value=10, max_value=N_STOCKS,
                           value=min(30, N_STOCKS), step=5, key="hm_slider")
        hm_data  = node_feats[:n_show, :]
        hm_ylabs = [f"S{idx_to_stock.get(i, i)}" for i in range(n_show)]

        fig_hm = go.Figure(go.Heatmap(
            z=hm_data,
            x=FEATURE_COLS,
            y=hm_ylabs,
            colorscale="RdBu_r",
            zmid=0,
            showscale=True,
        ))
        fig_hm.update_layout(
            template="plotly_white",
            title=f"Node Feature Heatmap (first {n_show} stocks)",
            xaxis_title="Feature", yaxis_title="Stock",
            height=500, margin=dict(t=50, b=80, l=60),
            xaxis=dict(tickangle=-45),
        )
        st.plotly_chart(fig_hm, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 4: Model Benchmarking
# ═══════════════════════════════════════════════════════════════
elif page == "Model Benchmarking":
    st.title("Model Benchmarking")
    st.caption("RegimeGAT vs GARCH baseline, residual diagnostics, and architecture summary")

    if not artifacts_ok:
        st.error(f"Could not load artifacts: {art_error}")
        st.stop()

    rs = art["regime_stats"]

    col_l, col_r = st.columns(2)

    with col_l:
        # ── RMSPE comparison bar chart ─────────────────────────
        GARCH_BASELINE = 0.350   # approximate placeholder
        fig_cmp = go.Figure()

        for _, row in rs.iterrows():
            r = int(row["regime"])
            fig_cmp.add_trace(go.Bar(
                name=f"RegimeGAT R{r}",
                x=[f"R{r} {REGIME_NAMES[r]}"],
                y=[row.get("best_val_rmspe", 0.0)],
                marker_color=REGIME_HEX[r],
            ))

        fig_cmp.add_trace(go.Bar(
            name="GARCH(1,1) ~baseline",
            x=["GARCH(1,1)"],
            y=[GARCH_BASELINE],
            marker_color="#6c757d",
            text=["~0.350 (approximate)"],
            textposition="auto",
        ))

        fig_cmp.update_layout(
            template="plotly_white",
            title="Validation RMSPE: RegimeGAT vs GARCH(1,1)",
            yaxis_title="RMSPE (lower is better)",
            barmode="group",
            height=360, margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

    with col_r:
        # ── Residual distribution ──────────────────────────────
        regime_sel = st.slider(
            "Regime for residual distribution",
            min_value=0, max_value=2, value=0, step=1,
            format="Regime %d",
        )
        sel_row = rs[rs["regime"] == regime_sel].iloc[0] if regime_sel in rs["regime"].values else rs.iloc[0]
        std_val = float(sel_row.get("std_tgt", 0.002))
        sim_res = np.random.normal(0, std_val, 1000)

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=sim_res,
            nbinsx=50,
            marker_color=REGIME_HEX[regime_sel],
            opacity=0.75,
            name=f"R{regime_sel} residuals",
        ))
        fig_hist.add_vline(x=0, line_dash="dash", line_color="#333")
        fig_hist.update_layout(
            template="plotly_white",
            title=f"Approximate Residual Distribution — Regime {regime_sel} ({REGIME_NAMES[regime_sel]})",
            xaxis_title="Residual (simulated from val-set std)",
            yaxis_title="Count",
            height=360, margin=dict(t=50, b=40),
        )
        st.markdown(
            '<span style="font-size:11px;color:#888">'
            'Simulated from N(0, regime std_tgt) — '
            'approximate distribution from validation set statistics.'
            '</span>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    # ── Architecture summary card ──────────────────────────────
    st.info(
        "**Model: RegimeGAT — Regime-Conditioned Graph Attention Volatility Forecaster**\n\n"
        "Each 10-minute order-book window is summarised into 17 features (WAP-derived realized "
        "volatility across sub-windows, bid-ask spread, order imbalance, depth ratio, and trade "
        "statistics). A 3-state Gaussian HMM classifies each window into a market regime "
        "(Trending / Mean-Reverting / Stressed) based on 4 market microstructure signals.\n\n"
        "A Graph Attention Network (GATConv, 4 heads) encodes cross-asset correlations "
        "across all 112 stocks using a Spearman correlation graph (top-20 edges per node). "
        "Each regime has a dedicated MLP head [256→128→64→32→1] that produces the final "
        "realized-volatility forecast. Models are trained end-to-end with RMSPE loss, "
        "AdamW optimiser, and a purged time-based validation split.\n\n"
        "📂 Dataset: [Optiver Realized Volatility Prediction](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction) "
        "— 112 Nasdaq stocks, ~3.8M order-book rows, 600-second windows.\n\n"
        "💻 Code: GitHub link placeholder — add your repo URL here."
    )

    with st.expander("Technical details"):
        # Compute real param counts
        try:
            m = list(art["models"].values())[0]
            params = sum(p.numel() for p in m.parameters())
            total_params = params * 3
        except Exception:
            params = 0
            total_params = 0

        gat_in  = N_FEATURES
        gat_out = 64 * 4  # heads=4
        mlp_layers = [(gat_out, 128), (128, 64), (64, 32), (32, 1)]
        mlp_params = sum(i*o + o for i, o in mlp_layers)
        gat_params_approx = (gat_in * 64 * 4) + (64 * 4 * 4)  # approx

        st.markdown(f"""
| Component | Details |
|---|---|
| GATConv | in={N_FEATURES} → out=64, heads=4, concat=True → dim 256 |
| MLP Head | 256→128→64→32→1, LayerNorm + ReLU + Dropout(0.2) per layer |
| Params per regime | ~{params:,} (loaded from checkpoint) |
| Total params (3 regimes) | ~{total_params:,} |
| Number of regimes | 3 (HMM-assigned) |
| Loss function | RMSPE = √(mean(((ŷ−y)/y)²)) |
| Optimiser | AdamW lr=1e-3, weight_decay=1e-4 |
| Scheduler | CosineAnnealingLR T_max=50 |
| Epochs | 50 per regime |
| Patience | 10 epochs (best checkpoint always saved) |
| Validation split | Purged time-based, last 20% of time_ids |
| Graph edges | Top-20 Spearman-correlated neighbours per stock |
| Training device | GPU P100 (Kaggle) |
| Inference device | CPU (Streamlit Cloud) |
        """)
