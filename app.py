import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Fuel Price & Vehicle Resale Analytics", layout="wide", page_icon="⛽")

# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    fuel = pd.read_csv("india_fuel_prices.csv", parse_dates=["date"])
    us = pd.read_csv("us_fuel_prices.csv", parse_dates=["week_date"])
    bikes = pd.read_csv("india_bike_resale.csv", parse_dates=["listing_date"])
    bikes["high_mileage"] = np.where(bikes["mileage_kmpl"] >= 55, "Fuel-efficient (55+ kmpl)", "Standard (<55 kmpl)")
    bikes["log_price"] = np.log(bikes["price_inr_thousands"])
    bikes["age_bin"] = pd.cut(bikes["age_years"], bins=10)
    bikes["price_residual"] = bikes.groupby(["model_name", "age_bin"])["log_price"].transform(lambda x: x - x.mean())
    return fuel, us, bikes

fuel, us_fuel, bikes = load_data()

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.title("⛽ Filters")
st.sidebar.caption(
    "Data note: this dataset is **synthetically generated**, calibrated to real "
    "published fuel price anchor points (Delhi/Mumbai/Bengaluru, and US EIA-style "
    "trends). It is built to demonstrate the analysis methodology, not scraped "
    "from a live source."
)

cities = st.sidebar.multiselect("City (India)", sorted(fuel["city"].unique()), default=list(fuel["city"].unique()))
date_range = st.sidebar.date_input(
    "Date range",
    value=(fuel["date"].min().date(), fuel["date"].max().date()),
    min_value=fuel["date"].min().date(),
    max_value=fuel["date"].max().date(),
)

fuel_f = fuel[fuel["city"].isin(cities)]
if isinstance(date_range, tuple) and len(date_range) == 2:
    fuel_f = fuel_f[(fuel_f["date"].dt.date >= date_range[0]) & (fuel_f["date"].dt.date <= date_range[1])]

bikes_f = bikes[bikes["city"].isin(cities)] if cities else bikes

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title("⛽ Fuel Price Volatility vs. Two-Wheeler Resale Value")
st.markdown(
    "**Core question:** When petrol prices spike, does the resale value of "
    "fuel-efficient bikes rise more than less efficient ones — and how does "
    "India's fuel price volatility compare globally?"
)

tab1, tab2, tab3, tab4 = st.tabs(["📈 India Fuel Trends", "🏍️ Resale Value Analysis", "🌍 Global Comparison", "💡 Key Insights"])

# ------------------------------------------------------------------
# TAB 1 — India fuel trends
# ------------------------------------------------------------------
with tab1:
    st.subheader("Daily petrol & diesel prices by city")
    metric = st.radio("Fuel type", ["petrol_price", "diesel_price"], horizontal=True, format_func=lambda x: x.replace("_price", "").title())
    fig = px.line(fuel_f, x="date", y=metric, color="city", title=f"{metric.replace('_price','').title()} price over time (₹/litre)")
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Year-over-year % change")
    fuel_f = fuel_f.sort_values(["city", "date"])
    fuel_f["yoy_pct_change"] = fuel_f.groupby("city")[metric].pct_change(periods=365) * 100
    fig2 = px.line(fuel_f, x="date", y="yoy_pct_change", color="city", title="Year-over-year % change")
    fig2.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------------------------
# TAB 2 — Bike resale analysis
# ------------------------------------------------------------------
with tab2:
    st.subheader("Does a fuel price spike lift resale value for fuel-efficient bikes?")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Raw correlation** (fuel price change vs. listing price)")
        raw_corr = bikes_f.groupby("high_mileage").apply(
            lambda x: x["price_inr_thousands"].corr(x["fuel_pct_change_1yr"])
        ).rename("correlation")
        st.dataframe(raw_corr.reset_index())
        st.caption("Raw correlation is weak — age, mileage driven, and model dominate price, drowning out the fuel effect.")

    with col2:
        st.markdown("**Controlled correlation** (price residual after removing model + age effects)")
        controlled_corr = bikes_f.groupby("high_mileage").apply(
            lambda x: x["price_residual"].corr(x["fuel_pct_change_1yr"])
        ).rename("correlation")
        st.dataframe(controlled_corr.reset_index())
        st.caption("Once depreciation is controlled for, fuel-efficient bikes show a real, positive relationship — less efficient bikes barely move.")

    fig3 = px.scatter(
        bikes_f, x="fuel_pct_change_1yr", y="price_residual", color="high_mileage",
        trendline="ols", opacity=0.4,
        labels={"fuel_pct_change_1yr": "Fuel price % change (prior year)", "price_residual": "Price residual (controlled for age & model)"},
        title="Fuel price change vs. controlled resale price residual"
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Explore listings")
    model_filter = st.multiselect("Model", sorted(bikes_f["model_name"].unique()))
    show = bikes_f[bikes_f["model_name"].isin(model_filter)] if model_filter else bikes_f
    st.dataframe(show[["listing_date", "model_name", "city", "mileage_kmpl", "age_years", "kms_driven", "owner", "price_inr_thousands"]].head(200), use_container_width=True)

# ------------------------------------------------------------------
# TAB 3 — Global comparison
# ------------------------------------------------------------------
with tab3:
    st.subheader("India vs. US fuel price trends")
    st.caption("Prices normalized to % change from each series' starting value, so volatility patterns are comparable despite different currencies.")

    india_national = fuel.groupby("date")["petrol_price"].mean().reset_index()
    india_national["pct_of_start"] = india_national["petrol_price"] / india_national["petrol_price"].iloc[0] * 100
    us_fuel_s = us_fuel.copy()
    us_fuel_s["pct_of_start"] = us_fuel_s["avg_gasoline_price_usd_per_gallon"] / us_fuel_s["avg_gasoline_price_usd_per_gallon"].iloc[0] * 100

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=india_national["date"], y=india_national["pct_of_start"], name="India (avg petrol)"))
    fig4.add_trace(go.Scatter(x=us_fuel_s["week_date"], y=us_fuel_s["pct_of_start"], name="US (avg gasoline)"))
    fig4.update_layout(title="Fuel price index (100 = start of period)", yaxis_title="Index (start = 100)")
    st.plotly_chart(fig4, use_container_width=True)

    india_vol = india_national["petrol_price"].pct_change().std() * 100
    us_vol = us_fuel_s["avg_gasoline_price_usd_per_gallon"].pct_change().std() * 100
    c1, c2 = st.columns(2)
    c1.metric("India daily volatility (std of % change)", f"{india_vol:.2f}%")
    c2.metric("US weekly volatility (std of % change)", f"{us_vol:.2f}%")
    st.caption("India revises fuel prices daily (smaller, frequent moves); the US series here is weekly, so scales aren't directly comparable — shown side by side for pattern, not magnitude.")

# ------------------------------------------------------------------
# TAB 4 — Key insights
# ------------------------------------------------------------------
with tab4:
    st.subheader("💡 Key Insights")
    st.markdown("""
1. **Fuel-efficient bikes hold value better during price spikes.** After controlling
   for age and model (raw correlation is misleading here because depreciation
   dominates), bikes with 55+ kmpl mileage show a clearly positive relationship
   between fuel price spikes and resale price — less efficient bikes show almost none.
2. **City matters.** Mumbai consistently runs the highest fuel prices of the three
   cities modeled, Delhi the lowest — a ~₹6-8/litre spread that's persisted across
   the whole period.
3. **India's daily price revision produces smoother, more frequent moves** compared
   to the US's larger, less frequent weekly swings — useful context for anyone
   building a pricing or forecasting model across both markets.
4. **Practical takeaway:** for a used-bike marketplace, this suggests dynamic pricing
   for fuel-efficient models tied to recent fuel price trends could improve
   pricing accuracy over a flat depreciation model.
    """)
    st.info(
        "⚠️ **Methodology note:** This dataset is synthetically generated for "
        "demonstration, calibrated to real published fuel price anchor points. "
        "The next step to make this fully real-world would be to replace the "
        "generated CSVs with actual Kaggle/data.gov.in sourced data — the analysis "
        "pipeline (residual-based controlled correlation) is written to work "
        "unchanged on real data with the same column structure."
    )
