"""
ICD341 Capstone: Statistical Analysis and Visualization of Maize Production
in Zambia Using Python.

Reusable analysis functions used by the notebook and the Streamlit dashboard.
Run the whole pipeline from the project root with:

    python src/analysis.py

NOTE ON DATA: the dataset is SIMULATED (reproducible via a fixed random seed)
so that the project runs anywhere. To use real data (e.g. Zambia Statistics
Agency crop forecast surveys, FAOSTAT), replace data/maize_production.csv
with a file that has the same column names.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "maize_production.csv"
OUT_DIR = ROOT / "outputs"
SEED = 123

# district: (province, base area ha, mean annual rainfall mm, soil effect t/ha)
DISTRICTS = {
    "Lusaka":    ("Lusaka",   30000,  850,  0.10),
    "Chongwe":   ("Lusaka",   42000,  880, -0.10),
    "Kabwe":     ("Central",  55000,  900,  0.15),
    "Mkushi":    ("Central",  70000,  980,  0.30),
    "Mumbwa":    ("Central",  48000,  870, -0.20),
    "Chipata":   ("Eastern",  85000, 1000,  0.20),
    "Petauke":   ("Eastern",  60000,  930, -0.05),
    "Choma":     ("Southern", 52000,  800,  0.05),
    "Mazabuka":  ("Southern", 45000,  790,  0.35),
    "Kasama":    ("Northern", 38000, 1250, -0.30),
}
YEARS = range(2010, 2024)


# ----------------------------------------------------------------- data
def generate_data(seed=SEED, path=DATA_PATH):
    """Create a reproducible simulated district-level maize dataset."""
    rng = np.random.default_rng(seed)
    rows = []
    for d, (prov, area0, rain0, soil) in DISTRICTS.items():
        for y in YEARS:
            t = y - 2010
            rain = rng.normal(rain0, 140)
            fert = max(rng.normal(150 + 5 * t, 25), 60)
            area = area0 * (1 + 0.02 * t) * rng.normal(1, 0.05)
            yld = (-0.5 + soil + 0.0015 * rain + 0.008 * fert
                   + 0.01 * t + rng.normal(0, 0.25))
            yld = max(yld, 0.8)
            rows.append([d, prov, y, round(area), round(rain, 1),
                         round(fert, 1), round(yld, 3), round(area * yld)])
    df = pd.DataFrame(rows, columns=["District", "Province", "Year", "Area_ha",
                                     "Rainfall_mm", "Fertilizer_kg_ha",
                                     "Yield_t_ha", "Production_t"])
    # Make the raw file realistically messy: ~4% missing rainfall values
    miss = rng.choice(df.index, size=int(0.04 * len(df)), replace=False)
    df.loc[miss, "Rainfall_mm"] = np.nan
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_data(path=DATA_PATH):
    if not Path(path).exists():
        generate_data(path=Path(path))
    return pd.read_csv(path)


def clean_data(df):
    """Remove duplicates, fix types, impute missing values by district median."""
    df = df.drop_duplicates().copy()
    for col in ["Rainfall_mm", "Fertilizer_kg_ha"]:
        df[col] = df.groupby("District")[col].transform(lambda s: s.fillna(s.median()))
    df["Yield_t_ha"] = df["Yield_t_ha"].fillna(df["Production_t"] / df["Area_ha"])
    df["Year"] = df["Year"].astype(int)
    assert df.isna().sum().sum() == 0, "Missing values remain after cleaning"
    return df.reset_index(drop=True)


# ----------------------------------------------------- descriptive stats
def district_summary(df):
    g = df.groupby("District").agg(
        Mean_Production=("Production_t", "mean"),
        SD_Production=("Production_t", "std"),
        Mean_Yield=("Yield_t_ha", "mean"),
        Mean_Rainfall=("Rainfall_mm", "mean"),
    ).round(2)
    g["CV_%"] = (100 * g["SD_Production"] / g["Mean_Production"]).round(1)
    return g.sort_values("Mean_Production", ascending=False)


def headline_numbers(df):
    by_d = df.groupby("District")["Production_t"].mean()
    return {
        "total_production_t": int(df["Production_t"].sum()),
        "mean_yield_t_ha": round(float(df["Yield_t_ha"].mean()), 3),
        "highest_district": by_d.idxmax(),
        "lowest_district": by_d.idxmin(),
    }


def national_trend(df):
    t = df.groupby("Year").agg(Production_t=("Production_t", "sum"),
                               Mean_Yield=("Yield_t_ha", "mean")).reset_index()
    res = stats.linregress(t["Year"], t["Mean_Yield"])
    return t, {"slope": res.slope, "p_value": res.pvalue, "r2": res.rvalue ** 2}


# ----------------------------------------------------------- regression
def _design(df):
    X = np.column_stack([np.ones(len(df)), df["Rainfall_mm"],
                         df["Fertilizer_kg_ha"], df["Year"] - 2010])
    return X, df["Yield_t_ha"].to_numpy()


def fit_ols(df):
    """OLS: Yield ~ Rainfall + Fertilizer + Trend (implemented with NumPy)."""
    X, y = _design(df)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    sigma2 = resid @ resid / (n - k)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), df=n - k)
    r2 = 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))
    table = pd.DataFrame({"Coefficient": beta, "Std_Error": se, "t": t, "p_value": p},
                         index=["Intercept", "Rainfall_mm", "Fertilizer_kg_ha", "Year_trend"])
    return {"table": table.round(5), "r2": r2, "sigma": np.sqrt(sigma2),
            "beta": beta, "n": n}


def bootstrap_rain_slope(df, n_boot=2000, seed=SEED):
    """95% bootstrap CI for the rainfall coefficient."""
    rng = np.random.default_rng(seed)
    X, y = _design(df)
    slopes = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        beta, *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
        slopes[b] = beta[1]
    return np.percentile(slopes, [2.5, 97.5]), slopes


# ---------------------------------------------------------- Monte Carlo
def drought_risk(df, ols, n_sim=20000, rain_cut=0.85, yield_threshold=2.0, seed=SEED):
    """Monte Carlo: how likely is a low-yield season if rainfall falls short?

    Rainfall ~ Normal(mean, sd) of observed data; yield = fitted model + residual noise.
    Compares normal rainfall with a drought scenario (mean rainfall x `rain_cut`).
    """
    rng = np.random.default_rng(seed)
    b, sigma = ols["beta"], ols["sigma"]
    fert, trend = df["Fertilizer_kg_ha"].mean(), df["Year"].max() - 2010
    out = {}
    for name, factor in [("normal", 1.0), ("drought", rain_cut)]:
        rain = rng.normal(df["Rainfall_mm"].mean() * factor, df["Rainfall_mm"].std(), n_sim)
        yld = b[0] + b[1] * rain + b[2] * fert + b[3] * trend + rng.normal(0, sigma, n_sim)
        out[name] = {"p_low_yield": float((yld < yield_threshold).mean()),
                     "mean_yield": float(yld.mean()), "samples": yld}
    return out


# -------------------------------------------------------------- graphs
def make_graphs(df, ols, path=OUT_DIR / "graphs.png"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    m = df.groupby("District")["Production_t"].mean().sort_values(ascending=False)
    ax[0, 0].bar(m.index, m.values, color="#2a7f62")
    ax[0, 0].set(title="Mean annual maize production by district", ylabel="Tonnes")
    ax[0, 0].tick_params(axis="x", rotation=45)
    t, _ = national_trend(df)
    ax[0, 1].plot(t["Year"], t["Production_t"] / 1000, marker="o", color="#c0562b")
    ax[0, 1].set(title="Total production, 10 districts", xlabel="Year", ylabel="'000 tonnes")
    ax[1, 0].scatter(df["Rainfall_mm"], df["Yield_t_ha"], alpha=.6, color="#35608d")
    b = ols["beta"]
    xs = np.linspace(df["Rainfall_mm"].min(), df["Rainfall_mm"].max(), 50)
    ys = b[0] + b[1] * xs + b[2] * df["Fertilizer_kg_ha"].mean() + b[3] * 6.5
    ax[1, 0].plot(xs, ys, color="black", label="Fitted (other variables at mean)")
    ax[1, 0].set(title="Rainfall vs yield", xlabel="Rainfall (mm)", ylabel="Yield (t/ha)")
    ax[1, 0].legend()
    ax[1, 1].hist(df["Yield_t_ha"], bins=20, edgecolor="black", color="#8c6bb1")
    ax[1, 1].set(title="Distribution of yield", xlabel="Yield (t/ha)", ylabel="Frequency")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    raw = load_data()
    print("Raw rows:", len(raw), "| missing cells:", int(raw.isna().sum().sum()))
    df = clean_data(raw)
    print("\nHeadline:", headline_numbers(df))
    print("\nDistrict summary:\n", district_summary(df))
    ols = fit_ols(df)
    print("\nOLS (R2 = %.3f):\n" % ols["r2"], ols["table"])
    ci, _ = bootstrap_rain_slope(df, 1000)
    print("\nBootstrap 95%% CI for rainfall slope: [%.5f, %.5f]" % tuple(ci))
    risk = drought_risk(df, ols)
    print("P(yield < 2 t/ha): normal=%.3f  drought=%.3f" %
          (risk["normal"]["p_low_yield"], risk["drought"]["p_low_yield"]))
    print("Graphs saved to", make_graphs(df, ols))


if __name__ == "__main__":
    main()
