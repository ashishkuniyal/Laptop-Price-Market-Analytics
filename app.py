"""
Laptop Price & Market Analytics — Streamlit Web App
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (MUST be the first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Laptop Price & Market Analytics",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — dark premium theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #0d0b24 0%, #1a1a2e 60%, #16213e 100%);
    color: #e0e0e0;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12122a 0%, #16213e 100%);
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stMultiSelect span { background: #1e3a5f !important; }

/* ── KPI Cards ────────────────────────────────── */
.kpi-row { display:flex; gap:14px; margin-bottom:24px; }
.kpi-card {
    flex:1;
    background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%);
    border: 1px solid rgba(233,69,96,0.4);
    border-radius: 14px;
    padding: 22px 18px;
    text-align: center;
    box-shadow: 0 6px 24px rgba(233,69,96,0.12), 0 2px 8px rgba(0,0,0,0.3);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(233,69,96,0.25);
}
.kpi-icon { font-size: 1.4rem; margin-bottom: 6px; }
.kpi-value {
    font-size: 1.9rem;
    font-weight: 900;
    background: linear-gradient(90deg, #e94560, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.kpi-label {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 6px;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    font-weight: 600;
}

/* ── Section headers ──────────────────────────── */
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #f1f5f9;
    border-left: 3px solid #e94560;
    padding-left: 10px;
    margin: 0 0 12px 0;
}
.section-sub {
    font-size: 0.78rem;
    color: #64748b;
    margin: -8px 0 14px 13px;
}

/* ── Tabs ─────────────────────────────────────── */
[data-testid="stTabs"] {
    border-bottom: 1px solid #1e3a5f;
}
button[data-baseweb="tab"] {
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    color: #64748b !important;
    padding: 10px 22px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #e94560 !important;
    border-bottom: 2px solid #e94560 !important;
}

/* ── Prediction box ───────────────────────────── */
.pred-box {
    background: linear-gradient(135deg, #0d1b3e 0%, #1e1250 100%);
    border-radius: 16px;
    padding: 32px 24px;
    text-align: center;
    border: 1px solid rgba(233,69,96,0.5);
    box-shadow: 0 0 40px rgba(233,69,96,0.15), inset 0 1px 0 rgba(255,255,255,0.05);
}
.pred-label {
    font-size: 0.8rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 600;
    margin-bottom: 10px;
}
.pred-price {
    font-size: 3.2rem;
    font-weight: 900;
    background: linear-gradient(90deg, #e94560, #f59e0b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}
.pred-range {
    font-size: 0.85rem;
    color: #475569;
    margin-top: 6px;
}
.pred-tier-badge {
    display: inline-block;
    padding: 4px 16px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 700;
    margin-top: 12px;
    letter-spacing: 0.05em;
}
.insight-box {
    background: rgba(6,182,212,0.08);
    border: 1px solid rgba(6,182,212,0.25);
    border-radius: 10px;
    padding: 14px 16px;
    margin-top: 16px;
    font-size: 0.82rem;
    color: #94a3b8;
    line-height: 1.6;
}

/* ── Divider ──────────────────────────────────── */
hr { border-color: #1e3a5f !important; }

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers  — fixes the duplicate-margin bug
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = ["#e94560","#06b6d4","#f59e0b","#10b981","#8b5cf6",
           "#f97316","#ec4899","#84cc16","#3b82f6","#a78bfa"]

def dark_layout(fig, height=380, margin=None, **extra):
    """Apply consistent dark layout without duplicate-keyword errors."""
    m = margin if margin else dict(t=46, b=28, l=16, r=16)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,52,96,0.18)",
        font=dict(color="#e0e0e0", family="Inter"),
        height=height,
        margin=m,
        **extra
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("Laptop_Cleaned_Dataset.csv")
    df["storage"] = df["storage"].fillna(df["storage"].mode()[0])
    df["ram"]     = df["ram"].fillna(df["ram"].mode()[0])
    df["os"]      = df["os"].fillna("Unknown")
    df["extract_cpu_brand"]  = df["extract_cpu_brand"].fillna("Unknown")
    df["cpu_gen"]            = df["cpu_gen"].fillna("Unknown")
    df["intel_core_series"]  = df["intel_core_series"].fillna("Not Applicable")
    df["cpu_ryzen_series"]   = df["cpu_ryzen_series"].fillna("Not Applicable")
    for c in ["cpu_priority","gpu_priority","os_priority","ram_priority","storage_priority"]:
        df[c] = df[c].fillna(0)
    df["gpu"]   = df["gpu"].fillna("Unknown GPU")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.drop_duplicates().dropna(subset=["price"])
    return df


@st.cache_resource(show_spinner=False)
def train_model(df):
    FEATURES = ["cpu_priority","gpu_priority","ram_priority","os_priority","storage_priority"]
    ml = df[FEATURES + ["price"]].dropna()
    X, y = ml[FEATURES], ml["price"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    mdl = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    mdl.fit(X_tr, y_tr)
    r2 = r2_score(y_te, mdl.predict(X_te))
    return mdl, round(r2, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading dataset…"):
    df_full = load_data()

with st.spinner("Training price predictor…"):
    model, model_r2 = train_model(df_full)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown(
    "<h2 style='color:#e94560;margin-bottom:4px;'>🔍 Filters</h2>"
    "<p style='color:#475569;font-size:0.78rem;margin-top:0;'>Affects all charts</p>",
    unsafe_allow_html=True
)
st.sidebar.markdown("---")

all_brands = sorted(df_full["brand"].unique())
sel_brands = st.sidebar.multiselect("🏷 Brand", all_brands, default=all_brands)

all_os = sorted(df_full["os"].dropna().unique())
sel_os = st.sidebar.multiselect("💻 Operating System", all_os, default=all_os)

all_usage = sorted(df_full["usage_purpose"].dropna().unique())
sel_usage = st.sidebar.multiselect("🎯 Usage Purpose", all_usage, default=all_usage)

all_cpu = sorted(df_full["extract_cpu_brand"].dropna().unique())
sel_cpu = st.sidebar.multiselect("🔵 CPU Brand", all_cpu, default=all_cpu)

price_min = int(df_full["price"].min())
price_max = int(df_full["price"].max())
sel_price = st.sidebar.slider(
    "💰 Price Range (₺)", price_min, price_max,
    (price_min, price_max), step=1000, format="₺%d"
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<p style='color:#334155;font-size:0.75rem;text-align:center;'>"
    "💻 Laptop Price & Market Analytics<br>"
    "27,456 records · 10 brands · MIT License</p>",
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────────────────────
# Apply filters
# ─────────────────────────────────────────────────────────────────────────────
df = df_full.copy()
if sel_brands: df = df[df["brand"].isin(sel_brands)]
if sel_os:     df = df[df["os"].isin(sel_os)]
if sel_usage:  df = df[df["usage_purpose"].isin(sel_usage)]
if sel_cpu:    df = df[df["extract_cpu_brand"].isin(sel_cpu)]
df = df[df["price"].between(sel_price[0], sel_price[1])]

# Guard against empty filter result
if df.empty:
    st.warning("⚠️ No data matches the current filters. Please widen your selection.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:18px 0 8px;'>
<h1 style='font-size:2.8rem;font-weight:900;margin:0;
           background:linear-gradient(90deg,#e94560 0%,#8b5cf6 50%,#06b6d4 100%);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
    💻 Laptop Price &amp; Market Analytics
</h1>
<p style='color:#475569;margin-top:6px;font-size:0.95rem;'>
    Interactive market intelligence dashboard &nbsp;·&nbsp; ML price predictor &nbsp;·&nbsp;
    27,456 laptops &nbsp;·&nbsp; 10 brands
</p>
</div>
<hr style='margin:4px 0 20px;'>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# KPI row
# ─────────────────────────────────────────────────────────────────────────────
n   = len(df)
rev = df["price"].sum()
avg = df["price"].mean()
med = df["price"].median()
brs = df["brand"].nunique()
gpu_n = df["gpu"].nunique()

k1,k2,k3,k4,k5,k6 = st.columns(6)
for col, icon, val, lbl in [
    (k1, "📦", f"{n:,}",             "Total Listings"),
    (k2, "💰", f"₺{rev/1e9:.2f}bn", "Total Revenue"),
    (k3, "📊", f"₺{avg:,.0f}",       "Avg Price"),
    (k4, "〰️",  f"₺{med:,.0f}",      "Median Price"),
    (k5, "🏷",  str(brs),            "Brands"),
    (k6, "🎮", str(gpu_n),           "GPU Models"),
]:
    col.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-icon">{icon}</div>'
        f'<div class="kpi-value">{val}</div>'
        f'<div class="kpi-label">{lbl}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊  Market Overview",
    "🔬  Deep Analysis",
    "🤖  Price Predictor",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Market Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    # Row 1 — Total Sales | Price Distribution
    c1, c2 = st.columns(2, gap="medium")

    with c1:
        st.markdown('<div class="section-header">Total Sales Revenue by Brand</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Sorted by revenue · hover for exact figure</div>', unsafe_allow_html=True)
        bs = df.groupby("brand")["price"].sum().sort_values().reset_index()
        bs.columns = ["Brand", "Revenue"]
        fig = px.bar(bs, y="Brand", x="Revenue", orientation="h",
                     color="Revenue", color_continuous_scale="Plasma",
                     text=bs["Revenue"].apply(lambda v: f"₺{v/1e6:.1f}M"))
        dark_layout(fig, height=380, coloraxis_showscale=False,
                    xaxis_title="Total Sales (₺)", yaxis_title="")
        fig.update_traces(textposition="outside", textfont=dict(color="#e0e0e0", size=11))
        st.plotly_chart(fig, width="stretch")

    with c2:
        st.markdown('<div class="section-header">Price Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Distribution of all listings · mean & median lines</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x="price", nbins=70, color_discrete_sequence=["#e94560"],
                           opacity=0.85)
        fig.add_vline(x=med, line_dash="dash", line_color="#06b6d4", line_width=1.5,
                      annotation_text=f"Median ₺{med:,.0f}",
                      annotation_font=dict(color="#06b6d4", size=11))
        fig.add_vline(x=avg, line_dash="dash", line_color="#f59e0b", line_width=1.5,
                      annotation_text=f"Mean ₺{avg:,.0f}",
                      annotation_font=dict(color="#f59e0b", size=11),
                      annotation_position="bottom right")
        dark_layout(fig, height=380, xaxis_title="Price (₺)", yaxis_title="Count")
        st.plotly_chart(fig, width="stretch")

    # Row 2 — OS Treemap | Usage Revenue
    c3, c4 = st.columns(2, gap="medium")

    with c3:
        st.markdown('<div class="section-header">Operating System Market Share</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Area proportional to listing count</div>', unsafe_allow_html=True)
        os_c = df["os"].value_counts().reset_index()
        os_c.columns = ["OS", "Count"]
        # Dark-friendly colorscale — no bright yellows
        fig = px.treemap(os_c, path=["OS"], values="Count",
                         color="Count",
                         color_continuous_scale=[[0,"#0f3460"],[0.5,"#533483"],[1,"#e94560"]])
        fig.update_traces(
            textfont=dict(color="#ffffff", size=13),
            marker=dict(line=dict(color="#0d0b24", width=2))
        )
        dark_layout(fig, height=360, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

    with c4:
        st.markdown('<div class="section-header">Revenue & Volume by Usage Purpose</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Total revenue (left) · avg price (right)</div>', unsafe_allow_html=True)
        usage = df.groupby("usage_purpose")["price"].agg(["sum","mean","count"]).reset_index()
        usage.columns = ["Usage","Total Revenue","Avg Price","Count"]
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Total Revenue (₺)","Avg Price (₺)"],
                            horizontal_spacing=0.12)
        fig.add_trace(go.Bar(x=usage["Usage"], y=usage["Total Revenue"],
                             marker_color=PALETTE[:3], showlegend=False,
                             text=usage["Total Revenue"].apply(lambda v: f"₺{v/1e6:.0f}M"),
                             textposition="outside", textfont=dict(color="#e0e0e0")),
                      row=1, col=1)
        fig.add_trace(go.Bar(x=usage["Usage"], y=usage["Avg Price"],
                             marker_color=PALETTE[3:6], showlegend=False,
                             text=usage["Avg Price"].apply(lambda v: f"₺{v:,.0f}"),
                             textposition="outside", textfont=dict(color="#e0e0e0")),
                      row=1, col=2)
        dark_layout(fig, height=360)
        st.plotly_chart(fig, width="stretch")

    # Row 3 — Laptop count by brand bar
    st.markdown('<div class="section-header">Listing Count by Brand</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Total number of laptops listed per brand</div>', unsafe_allow_html=True)
    bc = df["brand"].value_counts().reset_index()
    bc.columns = ["Brand","Count"]
    fig = px.bar(bc, x="Brand", y="Count",
                 color="Count", color_continuous_scale="Teal",
                 text="Count")
    dark_layout(fig, height=320, coloraxis_showscale=False,
                xaxis_title="Brand", yaxis_title="Count")
    fig.update_traces(textposition="outside", textfont=dict(color="#e0e0e0", size=11))
    st.plotly_chart(fig, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Deep Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    # Row 1 — Boxplot | Scatter
    c1, c2 = st.columns(2, gap="medium")

    with c1:
        st.markdown('<div class="section-header">Price Range per Brand</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Box = IQR · line = median · dots = outliers</div>', unsafe_allow_html=True)
        brand_order = df.groupby("brand")["price"].median().sort_values(ascending=False).index.tolist()
        fig = px.box(df, x="brand", y="price", color="brand",
                     category_orders={"brand": brand_order},
                     color_discrete_sequence=PALETTE, points="outliers",
                     hover_data=["name"] if "name" in df.columns else None)
        dark_layout(fig, height=420, showlegend=False,
                    xaxis_title="Brand", yaxis_title="Price (₺)")
        fig.update_traces(marker=dict(size=3, opacity=0.4))
        st.plotly_chart(fig, width="stretch")

    with c2:
        st.markdown('<div class="section-header">RAM Tier vs Price — by Usage</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Hover for brand · CPU · GPU details</div>', unsafe_allow_html=True)
        sc = df.dropna(subset=["ram_priority","price","usage_purpose"])
        fig = px.scatter(sc.sample(min(len(sc), 5000), random_state=42),
                         x="ram_priority", y="price",
                         color="usage_purpose",
                         hover_data=["brand","cpu","gpu"],
                         color_discrete_sequence=PALETTE, opacity=0.55,
                         labels={"ram_priority":"RAM Tier","price":"Price (₺)",
                                 "usage_purpose":"Usage"})
        dark_layout(fig, height=420)
        st.plotly_chart(fig, width="stretch")

    # Row 2 — GPU sunburst | CPU brand pie
    c3, c4 = st.columns(2, gap="medium")

    with c3:
        st.markdown('<div class="section-header">Top 12 GPU Models</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">By listing count · hover for details</div>', unsafe_allow_html=True)
        gpu_top = df["gpu"].value_counts().head(12).reset_index()
        gpu_top.columns = ["GPU","Count"]
        fig = px.bar(gpu_top.sort_values("Count"), y="GPU", x="Count",
                     orientation="h",
                     color="Count", color_continuous_scale="Magma",
                     text="Count")
        dark_layout(fig, height=420, coloraxis_showscale=False,
                    yaxis_title="", xaxis_title="Count")
        fig.update_traces(textposition="outside", textfont=dict(color="#e0e0e0", size=10))
        st.plotly_chart(fig, width="stretch")

    with c4:
        st.markdown('<div class="section-header">CPU Brand Share</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Intel vs AMD vs other</div>', unsafe_allow_html=True)
        cpu_share = df["extract_cpu_brand"].value_counts().reset_index()
        cpu_share.columns = ["CPU Brand","Count"]
        fig = px.pie(cpu_share, names="CPU Brand", values="Count",
                     color_discrete_sequence=PALETTE,
                     hole=0.52)
        fig.update_traces(
            textinfo="percent+label",
            textfont=dict(color="#e0e0e0", size=12),
            marker=dict(line=dict(color="#0d0b24", width=2))
        )
        dark_layout(fig, height=420)
        fig.update_layout(legend=dict(orientation="v", x=1.02, y=0.5))
        st.plotly_chart(fig, width="stretch")

    # Row 3 — Average Price per Brand (horizontal)
    c5, c6 = st.columns(2, gap="medium")

    with c5:
        st.markdown('<div class="section-header">Average Price per Brand</div>', unsafe_allow_html=True)
        avg_b = df.groupby("brand")["price"].mean().sort_values().reset_index()
        avg_b.columns = ["Brand","Avg Price"]
        fig = px.bar(avg_b, y="Brand", x="Avg Price",
                     orientation="h",
                     color="Avg Price", color_continuous_scale="Turbo",
                     text=avg_b["Avg Price"].apply(lambda v: f"₺{v:,.0f}"))
        dark_layout(fig, height=380, coloraxis_showscale=False,
                    yaxis_title="", xaxis_title="Avg Price (₺)")
        fig.update_traces(textposition="outside", textfont=dict(color="#e0e0e0", size=10))
        st.plotly_chart(fig, width="stretch")

    with c6:
        st.markdown('<div class="section-header">Price Tier Distribution by Brand</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Share of Budget / Mid / Premium listings</div>', unsafe_allow_html=True)
        def price_tier(p):
            if p < 30000: return "Budget"
            elif p < 80000: return "Mid-Range"
            else: return "Premium"
        df_tier = df.copy()
        df_tier["tier"] = df_tier["price"].apply(price_tier)
        tier_brand = df_tier.groupby(["brand","tier"]).size().reset_index(name="count")
        tier_totals = tier_brand.groupby("brand")["count"].transform("sum")
        tier_brand["pct"] = tier_brand["count"] / tier_totals * 100
        fig = px.bar(tier_brand, x="brand", y="pct", color="tier",
                     color_discrete_map={"Budget":"#10b981","Mid-Range":"#f59e0b","Premium":"#e94560"},
                     barmode="stack",
                     labels={"pct":"Share (%)", "brand":"Brand", "tier":"Price Tier"})
        dark_layout(fig, height=380, xaxis_title="Brand", yaxis_title="Share (%)")
        fig.update_layout(legend=dict(orientation="h", y=1.06, x=0))
        st.plotly_chart(fig, width="stretch")

    # Full-width heatmap
    st.markdown('<div class="section-header">Feature Correlation Heatmap</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">How strongly each feature relates to price and other specs</div>', unsafe_allow_html=True)
    num_cols = ["price","cpu_priority","gpu_priority","ram_priority","os_priority","storage_priority"]
    corr = df[num_cols].corr().round(2)
    fig = px.imshow(corr, text_auto=True,
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="auto",
                    labels={"color":"Correlation"})
    dark_layout(fig, height=380)
    st.plotly_chart(fig, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Price Predictor
# ══════════════════════════════════════════════════════════════════════════════
with tab3:

    st.markdown(
        f"<p style='color:#64748b;font-size:0.9rem;'>"
        f"Configure laptop specs using the sliders below. Our "
        f"<strong style='color:#e94560;'>Random Forest model</strong> "
        f"(R² = <strong style='color:#06b6d4;'>{model_r2}</strong>, "
        f"trained on 24,610 real listings) will predict the market price instantly."
        f"</p>",
        unsafe_allow_html=True
    )

    col_s, col_r = st.columns([1.1, 1], gap="large")

    # Spec labels
    cpu_labels  = {0:"Unknown", 1:"Celeron / Pentium", 2:"Core i3 / Ryzen 3",
                   3:"Core i5 / Ryzen 5", 4:"Core i7 / Ryzen 7", 5:"Core i9 / Ryzen 9"}
    gpu_labels  = {0:"None / Unknown", 1:"Intel Integrated", 2:"AMD Integrated",
                   3:"Entry Discrete (MX / RX 6500)", 4:"Mid-Range (RTX 3060 / RX 6700)",
                   5:"High-End (RTX 4080 / RX 7900)"}
    ram_labels  = {0:"Unknown", 1:"4 GB", 2:"8 GB", 3:"12 GB",
                   4:"16 GB", 5:"32 GB", 6:"64 GB", 7:"96 GB"}
    os_labels   = {0:"Unknown", 1:"FreeDOS", 2:"Linux", 3:"Windows 11 Home",
                   4:"Windows 11 Pro", 5:"macOS"}
    stor_labels = {0:"Unknown", 1:"128 GB SSD", 2:"256 GB SSD", 3:"512 GB SSD",
                   4:"1 TB SSD", 5:"1 TB HDD", 6:"2 TB", 7:"4 TB+"}

    with col_s:
        st.markdown('<div class="section-header">🔧 Configure Specs</div>', unsafe_allow_html=True)

        def make_slider(label, opts, key, default):
            v = st.slider(label, min_value=min(opts), max_value=max(opts),
                          value=default, key=key)
            st.markdown(
                f"<span style='font-size:0.78rem;color:#06b6d4;margin-left:2px;'>"
                f"→ {opts[v]}</span>",
                unsafe_allow_html=True
            )
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            return float(v)

        cpu_v  = make_slider("🔵 CPU Tier",     cpu_labels,  "cpu_s",  3)
        gpu_v  = make_slider("🟣 GPU Tier",     gpu_labels,  "gpu_s",  3)
        ram_v  = make_slider("🟢 RAM Tier",     ram_labels,  "ram_s",  4)
        os_v   = make_slider("🟡 OS Tier",      os_labels,   "os_s",   3)
        stor_v = make_slider("🟠 Storage Tier", stor_labels, "stor_s", 3)

    with col_r:
        st.markdown('<div class="section-header">💡 Predicted Price</div>', unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Predict with all trees for confidence interval
        X_in = np.array([[cpu_v, gpu_v, ram_v, os_v, stor_v]])
        predicted = model.predict(X_in)[0]

        # Estimate range from individual tree predictions
        tree_preds = np.array([t.predict(X_in)[0] for t in model.estimators_])
        p10, p90   = np.percentile(tree_preds, 10), np.percentile(tree_preds, 90)

        # Tier
        if predicted < 20_000:
            tier, tc, tbg = "Budget",    "#10b981", "rgba(16,185,129,0.15)"
        elif predicted < 60_000:
            tier, tc, tbg = "Mid-Range", "#f59e0b", "rgba(245,158,11,0.15)"
        elif predicted < 100_000:
            tier, tc, tbg = "Upper-Mid", "#f97316", "rgba(249,115,22,0.15)"
        else:
            tier, tc, tbg = "Premium",   "#e94560", "rgba(233,69,96,0.15)"

        st.markdown(f"""
        <div class="pred-box">
            <div class="pred-label">Estimated Market Price</div>
            <div class="pred-price">&#8378;{predicted:,.0f}</div>
            <div class="pred-range">Likely range: ₺{p10:,.0f} – ₺{p90:,.0f}</div>
            <div class="pred-tier-badge" style="background:{tbg};color:{tc};border:1px solid {tc};">
                {tier}
            </div>
            <hr style="border-color:#1e3a5f;margin:18px 0 12px;">
            <div style="font-size:0.78rem;color:#475569;">
                Model accuracy (R²) = <strong style="color:#06b6d4;">{model_r2}</strong>
                &nbsp;·&nbsp; 300 decision trees &nbsp;·&nbsp; 24,610 training samples
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        # Feature importance — sorted, fixed margin
        imp_pairs = sorted(
            zip(["CPU","GPU","RAM","OS","Storage"], model.feature_importances_),
            key=lambda x: x[1]
        )
        feat_names  = [x[0] for x in imp_pairs]
        feat_scores = [x[1]*100 for x in imp_pairs]
        feat_colors = [PALETTE[i % len(PALETTE)] for i in range(len(feat_names))]

        fig = go.Figure(go.Bar(
            x=feat_scores, y=feat_names,
            orientation="h",
            marker=dict(color=feat_colors),
            text=[f"{s:.1f}%" for s in feat_scores],
            textposition="outside",
            textfont=dict(color="#e0e0e0", size=11)
        ))
        # NOTE: margin passed ONLY here — never in **DARK to avoid duplicate-kwarg error
        dark_layout(fig, height=230, margin=dict(t=8, b=8, l=10, r=50),
                    xaxis_title="Importance (%)", yaxis_title="",
                    title_text="Feature Importance")
        st.plotly_chart(fig, width="stretch")

        # Insight box
        top_feat = feat_names[-1]
        st.markdown(f"""
        <div class="insight-box">
            💡 <strong style='color:#e0e0e0;'>{top_feat} tier</strong> is the single strongest
            price driver in your dataset, contributing
            <strong style='color:#06b6d4;'>{feat_scores[-1]:.1f}%</strong> of the model's
            decision weight. Upgrading the {top_feat} tier by 1 step will have the largest impact
            on the estimated price.
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#334155;font-size:0.78rem;'>"
    "Laptop Price &amp; Market Analytics &nbsp;·&nbsp; "
    "Built with Python · Streamlit · Plotly · scikit-learn &nbsp;·&nbsp; MIT License"
    "</p>",
    unsafe_allow_html=True
)
