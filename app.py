"""
NSW Property Price Estimator
Run with: streamlit run app.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NSW Property Price Estimator",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODELS_DIR = Path(__file__).parent / "models"

# ── Startup check ─────────────────────────────────────────────────────────────

if not (MODELS_DIR / "xgb_model.json").exists():
    st.error(
        "Model files not found. "
        "Run `python scripts/build_model.py` first, then restart the app."
    )
    st.stop()

# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    m = xgb.XGBRegressor()
    m.load_model(str(MODELS_DIR / "xgb_model.json"))
    return m


@st.cache_resource
def load_pipeline():
    with open(MODELS_DIR / "pipeline.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_data():
    encoding = pd.read_parquet(MODELS_DIR / "suburb_encoding.parquet")[
        "suburb_mean_log_price"
    ]
    annual   = pd.read_parquet(MODELS_DIR / "suburb_annual_stats.parquet")
    nsw      = pd.read_parquet(MODELS_DIR / "nsw_annual_stats.parquet")
    summary  = pd.read_parquet(MODELS_DIR / "suburb_summary.parquet")
    return encoding, annual, nsw, summary


model    = load_model()
pipeline = load_pipeline()
suburb_encoding, suburb_annual, nsw_annual, suburb_summary = load_data()

suburb_list   = sorted(suburb_encoding.index.tolist())
enc_mean      = float(suburb_encoding.mean())
enc_std       = float(suburb_encoding.std())
FEATURE_NAMES = model.get_booster().feature_names

# ── Prediction ────────────────────────────────────────────────────────────────

def predict_price(suburb: str, area_m2: float, year: int, quarter: int) -> float:
    enc = float(suburb_encoding.get(suburb, enc_mean))
    enc = float(np.clip(enc, enc_mean - 3 * enc_std, enc_mean + 3 * enc_std))
    X_raw = pd.DataFrame({
        "log_area":       [np.log1p(area_m2)],
        "year":           [float(year)],
        "quarter":        [float(quarter)],
        "suburb_encoded": [enc],
    })
    X_scaled = pipeline.transform(X_raw)
    X_df     = pd.DataFrame(X_scaled, columns=FEATURE_NAMES)
    return float(np.expm1(model.predict(X_df)[0]))


# ── Header ────────────────────────────────────────────────────────────────────

st.title("NSW Property Price Estimator")
st.caption(
    "Based on 1.88 million NSW residential sales, 2010–2026.  "
    "Source: NSW Valuer General / Land Registry Services (CC BY 4.0)."
)

tab_est, tab_trends, tab_explore = st.tabs(
    ["Price Estimator", "Suburb Trends", "Suburb Explorer"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Price Estimator
# ─────────────────────────────────────────────────────────────────────────────

with tab_est:
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("Property details")

        suburb = st.selectbox(
            "Suburb",
            options=suburb_list,
            index=suburb_list.index("MOSMAN") if "MOSMAN" in suburb_list else 0,
            format_func=lambda s: s.title(),
            help="Select from 4,000+ NSW suburbs in the Land Registry dataset.",
        )
        area = st.number_input(
            "Land area (m²)",
            min_value=50,
            max_value=10_000,
            value=500,
            step=50,
        )
        year    = st.slider("Year of sale", 2010, 2026, 2026)
        quarter = st.selectbox(
            "Quarter",
            options=[1, 2, 3, 4],
            index=1,
            format_func=lambda q: f"Q{q}  ({['Jan–Mar','Apr–Jun','Jul–Sep','Oct–Dec'][q-1]})",
        )

        st.divider()
        st.caption(
            "The model uses suburb, land area, year, and quarter only.  "
            "It does not know bedrooms, bathrooms, or property condition."
        )

    with right:
        price = predict_price(suburb, area, year, quarter)
        low   = price * 0.78
        high  = price * 1.28

        st.subheader(f"Estimated sale price — {suburb.title()}")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Estimate",    f"${price / 1e6:.2f}M")
        col_b.metric("Lower bound", f"${low   / 1e6:.2f}M")
        col_c.metric("Upper bound", f"${high  / 1e6:.2f}M")

        st.caption(
            "Indicative range: about half of all XGBoost predictions fall within "
            "±20% of the actual sale price on the test set (375,514 properties)."
        )

        st.divider()

        # Historical price trend for this suburb
        sub_hist = suburb_annual[suburb_annual["suburb"] == suburb].sort_values("year")

        if not sub_hist.empty:
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=nsw_annual["year"],
                y=nsw_annual["median_price"] / 1e6,
                mode="lines",
                name="NSW median",
                line=dict(color="#cccccc", dash="dot", width=1.5),
                hovertemplate="NSW median — %{x}: $%{y:.2f}M<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=sub_hist["year"],
                y=sub_hist["median_price"] / 1e6,
                mode="lines+markers",
                name=suburb.title(),
                line=dict(color="#1f77b4", width=2.5),
                marker=dict(size=6),
                customdata=sub_hist["n_sales"],
                hovertemplate=(
                    f"{suburb.title()} — %{{x}}: $%{{y:.2f}}M<br>"
                    "Sales: %{customdata:,}<extra></extra>"
                ),
            ))
            # Star for the user's estimate
            fig.add_trace(go.Scatter(
                x=[year],
                y=[price / 1e6],
                mode="markers",
                name="Your estimate",
                marker=dict(color="#d62728", size=13, symbol="star"),
                hovertemplate=f"Estimate ({year}): ${price / 1e6:.2f}M<extra></extra>",
            ))

            fig.update_layout(
                title=f"{suburb.title()} — median sale price vs NSW",
                xaxis_title="Year",
                yaxis_title="Median price (AUD millions)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                hovermode="x",
                margin=dict(t=60, b=40, l=0, r=0),
                height=370,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No historical data available for {suburb.title()}.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Suburb Trends
# ─────────────────────────────────────────────────────────────────────────────

with tab_trends:
    st.subheader("Suburb price trends")
    st.caption(
        "Median sale price per year for selected suburbs. "
        "NSW median shown as a dotted reference line."
    )

    col_sel, col_yr = st.columns([3, 1])
    with col_sel:
        defaults = [s for s in ["MOSMAN", "PARRAMATTA", "BLACKTOWN", "PENRITH", "DUBBO"]
                    if s in suburb_list]
        selected = st.multiselect(
            "Select suburbs to compare",
            options=suburb_list,
            default=defaults,
            format_func=lambda s: s.title(),
        )
    with col_yr:
        yr_range = st.slider(
            "Year range", 2010, 2026, (2010, 2026), key="yr_trends"
        )

    if selected:
        plot_data = suburb_annual[
            suburb_annual["suburb"].isin(selected) &
            suburb_annual["year"].between(*yr_range)
        ].copy()
        nsw_filt = nsw_annual[nsw_annual["year"].between(*yr_range)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=nsw_filt["year"],
            y=nsw_filt["median_price"] / 1e6,
            mode="lines",
            name="NSW median",
            line=dict(color="#cccccc", dash="dot", width=1.5),
            hovertemplate="NSW median: $%{y:.2f}M<extra></extra>",
        ))
        for sub in selected:
            d = plot_data[plot_data["suburb"] == sub].sort_values("year")
            if d.empty:
                continue
            fig.add_trace(go.Scatter(
                x=d["year"],
                y=d["median_price"] / 1e6,
                mode="lines+markers",
                name=sub.title(),
                customdata=d["n_sales"],
                hovertemplate=(
                    f"<b>{sub.title()}</b><br>"
                    "%{x}: $%{y:.2f}M<br>"
                    "Sales: %{customdata:,}<extra></extra>"
                ),
            ))

        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Median sale price (AUD millions)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            height=480,
            margin=dict(t=20, b=40, l=0, r=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary table
        rows = []
        for sub in selected:
            d = suburb_annual[suburb_annual["suburb"] == sub]
            if d.empty:
                continue
            first = d.loc[d["year"].idxmin()]
            last  = d.loc[d["year"].idxmax()]
            change_pct = (last["median_price"] - first["median_price"]) / first["median_price"] * 100
            rows.append({
                "Suburb":          sub.title(),
                f"Median ({int(last['year'])})": f"${last['median_price']:,.0f}",
                f"Median ({int(first['year'])})": f"${first['median_price']:,.0f}",
                "Change":          f"+{change_pct:.0f}%" if change_pct >= 0 else f"{change_pct:.0f}%",
                "Total sales":     f"{d['n_sales'].sum():,}",
            })
        if rows:
            st.dataframe(
                pd.DataFrame(rows), use_container_width=True, hide_index=True
            )
    else:
        st.info("Select at least one suburb above to see the chart.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Suburb Explorer
# ─────────────────────────────────────────────────────────────────────────────

with tab_explore:
    st.subheader("Suburb Explorer")
    st.caption(
        "NSW residential property — median sale price across all years (2010–2026)."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        show_mode = st.radio(
            "Show", ["Most expensive", "Most affordable"], horizontal=True
        )
    with c2:
        min_sales = st.select_slider(
            "Min. sales (filters out low-data suburbs)",
            options=[50, 100, 200, 500, 1_000],
            value=200,
        )
    with c3:
        n_show = st.slider("Number of suburbs", 10, 40, 20)

    eligible = suburb_summary[suburb_summary["n_sales"] >= min_sales].copy()

    if show_mode == "Most expensive":
        display     = eligible.nlargest(n_show, "median_price").sort_values("median_price")
        color_scale = "Blues"
        chart_title = f"Top {n_show} most expensive suburbs (min. {min_sales:,} sales)"
    else:
        display     = eligible.nsmallest(n_show, "median_price").sort_values("median_price", ascending=False)
        color_scale = "Reds_r"
        chart_title = f"Top {n_show} most affordable suburbs (min. {min_sales:,} sales)"

    display = display.copy()
    display["suburb_label"] = display["suburb"].str.title()

    fig = px.bar(
        display,
        x="median_price",
        y="suburb_label",
        orientation="h",
        color="median_price",
        color_continuous_scale=color_scale,
        labels={"median_price": "Median sale price (AUD)", "suburb_label": ""},
        custom_data=["n_sales"],
        title=chart_title,
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Median: $%{x:,.0f}<br>"
            "Total sales: %{customdata[0]:,}<extra></extra>"
        )
    )
    fig.update_coloraxes(showscale=False)
    fig.update_layout(
        height=max(420, n_show * 22),
        margin=dict(t=50, b=20, l=10, r=30),
        xaxis=dict(tickformat="$,.0f"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Search all suburbs")

    col_s, col_o = st.columns([2, 1])
    with col_s:
        search = st.text_input("Filter by suburb name", placeholder="e.g. Bondi")
    with col_o:
        sort_by = st.selectbox(
            "Sort by", ["Median price", "Total sales", "Suburb name"]
        )

    table = suburb_summary.copy()
    if search:
        table = table[table["suburb"].str.contains(search.upper(), na=False)]

    sort_map = {
        "Median price": ("median_price", False),
        "Total sales":  ("n_sales", False),
        "Suburb name":  ("suburb", True),
    }
    sk, sa = sort_map[sort_by]
    table = table.sort_values(sk, ascending=sa).reset_index(drop=True)

    st.dataframe(
        pd.DataFrame({
            "Suburb":       table["suburb"].str.title(),
            "Median price": table["median_price"].apply(lambda x: f"${x:,.0f}"),
            "Total sales":  table["n_sales"].apply(lambda x: f"{x:,}"),
        }),
        use_container_width=True,
        height=420,
        hide_index=True,
    )


# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Data: NSW Government open data, CC BY 4.0.  "
    "Predictions are indicative only — not financial advice.  "
    "Model: XGBoost trained on 1.5M properties (R² = 0.57, RMSE ~$1.05M for 200-round model).  "
    "Source: [github.com/nancy-vn-le/nsw-property-price-prediction]"
    "(https://github.com/nancy-vn-le/nsw-property-price-prediction)"
)
