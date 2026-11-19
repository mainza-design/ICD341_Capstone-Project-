"""Interactive dashboard: Maize Production in Zambia.

Run from the project root:   streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import analysis as an  # noqa: E402

st.set_page_config(page_title="Zambia Maize Dashboard", layout="wide")


@st.cache_data
def get_data():
    return an.clean_data(an.load_data())


df_all = get_data()

st.title("Zambia Maize Production Dashboard")
st.caption("ICD341 Capstone | NOTE: data are simulated for learning purposes.")

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Filters")
districts = st.sidebar.multiselect("Districts", sorted(df_all["District"].unique()),
                                   default=sorted(df_all["District"].unique()))
y_min, y_max = int(df_all["Year"].min()), int(df_all["Year"].max())
years = st.sidebar.slider("Years", y_min, y_max, (y_min, y_max))

df = df_all[df_all["District"].isin(districts) & df_all["Year"].between(*years)]
if df.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# ---------------------------------------------------------------- metrics
head = an.headline_numbers(df)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total production", f"{head['total_production_t']:,} t")
c2.metric("Mean yield", f"{head['mean_yield_t_ha']:.2f} t/ha")
c3.metric("Highest district (mean)", head["highest_district"])
c4.metric("Lowest district (mean)", head["lowest_district"])

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "District explorer", "Model", "Risk simulation"])

# --------------------------------------------------------------- overview
with tab1:
    left, right = st.columns(2)
    with left:
        st.subheader("Mean production by district")
        m = df.groupby("District")["Production_t"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots()
        ax.bar(m.index, m.values, color="#2a7f62")
        ax.set_ylabel("Tonnes")
        plt.xticks(rotation=40)
        st.pyplot(fig)
    with right:
        st.subheader("Production trend")
        trend = df.pivot_table(index="Year", columns="District", values="Production_t")
        st.line_chart(trend)
    st.subheader("Summary statistics by district")
    st.dataframe(an.district_summary(df))
    st.download_button("Download filtered data (CSV)", df.to_csv(index=False),
                       file_name="maize_filtered.csv", mime="text/csv")

# ------------------------------------------------------- district explorer
with tab2:
    chosen = st.selectbox("Choose a district", sorted(df["District"].unique()))
    sub = df[df["District"] == chosen]
    st.write(f"**{chosen}**: mean production {sub['Production_t'].mean():,.0f} t, "
             f"mean yield {sub['Yield_t_ha'].mean():.2f} t/ha, "
             f"mean rainfall {sub['Rainfall_mm'].mean():.0f} mm")
    st.line_chart(sub.set_index("Year")[["Production_t"]])
    st.dataframe(sub)

# ------------------------------------------------------------------ model
with tab3:
    if len(df) > 10:
        ols = an.fit_ols(df)
        st.write(f"Yield ~ Rainfall + Fertilizer + Year trend  (n = {ols['n']}, "
                 f"R² = {ols['r2']:.3f})")
        st.dataframe(ols["table"])
        fig, ax = plt.subplots()
        ax.scatter(df["Rainfall_mm"], df["Yield_t_ha"], alpha=.6)
        ax.set_xlabel("Rainfall (mm)")
        ax.set_ylabel("Yield (t/ha)")
        st.pyplot(fig)
    else:
        st.info("Select more districts or years to fit the model (need > 10 rows).")

# ------------------------------------------------------------- simulation
with tab4:
    if len(df) > 10:
        ols = an.fit_ols(df)
        cut = st.slider("Rainfall as % of normal", 60, 100, 85) / 100
        thr = st.slider("Low-yield threshold (t/ha)", 1.0, 3.0, 2.0, 0.1)
        risk = an.drought_risk(df, ols, rain_cut=cut, yield_threshold=thr)
        a, b = st.columns(2)
        a.metric("P(low yield), normal rainfall", f"{risk['normal']['p_low_yield']:.1%}")
        b.metric("P(low yield), reduced rainfall", f"{risk['drought']['p_low_yield']:.1%}")
        fig, ax = plt.subplots()
        ax.hist(risk["normal"]["samples"], bins=40, alpha=.6, label="Normal")
        ax.hist(risk["drought"]["samples"], bins=40, alpha=.6, label="Reduced rainfall")
        ax.axvline(thr, color="black", ls="--")
        ax.set_xlabel("Simulated yield (t/ha)")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("Select more districts or years to run the simulation.")
