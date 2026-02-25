import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Settings
# -----------------------------
USE_MIN_ABC = True   # If True: HC = min(PA,PB,PC); if False: choose a single column below
SINGLE_COL = "PA"    # Used only if USE_MIN_ABC is False


# -----------------------------
# IO helpers
# -----------------------------
def read_all_capmaps(root: Path) -> pd.DataFrame:
    records = []
    station_dirs = sorted([p for p in root.iterdir() if p.is_dir()])

    for st_dir in station_dirs:
        station = st_dir.name
        cap_path = st_dir / "out_6_3" / "CapMap_chk.txt"
        if not cap_path.exists():
            continue

        try:
            df = pd.read_csv(cap_path, skipinitialspace=True)
            df.columns = [c.strip() for c in df.columns]
            df["station"] = station
            records.append(df)
        except Exception as e:
            print(f"[WARN] Failed reading {cap_path}: {e}")

    if not records:
        raise RuntimeError(f"No CapMap_chk.txt found under: {root}")

    return pd.concat(records, ignore_index=True)


def split_modes(df: pd.DataFrame):
    demand = df[(df["F"] >= 0) & (df["F"] <= 23)].copy()
    demand["hour"] = demand["F"].astype(int)
    demand["mode"] = "Demand"

    gen = df[(df["F"] >= 24) & (df["F"] <= 47)].copy()
    gen["hour"] = (gen["F"] - 24).astype(int)
    gen["mode"] = "Generation"

    return demand, gen


def compute_hc_metric(df: pd.DataFrame) -> pd.Series:
    for c in ["PA", "PB", "PC", "P3"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if USE_MIN_ABC:
        return df[["PA", "PB", "PC"]].min(axis=1)
    else:
        if SINGLE_COL not in df.columns:
            raise ValueError(f"Column {SINGLE_COL} not found.")
        return df[SINGLE_COL]


# -----------------------------
# Firm / non-firm metrics
# -----------------------------
def firm_nonfirm_by_bus(df_mode: pd.DataFrame, q_list=(0.90, 0.95)) -> pd.DataFrame:
    tmp = df_mode.copy()
    tmp["HC"] = compute_hc_metric(tmp)

    # Group by station + BusID (CGP buses)
    tmp = tmp.rename(columns={"BusId": "BusID"})
    g = tmp.groupby(["station", "BusID"])["HC"]

    out = pd.DataFrame({
        "HC_firm_min": g.min(),
        "HC_median": g.quantile(0.50),
        "HC_mean": g.mean(),
    })

    for q in q_list:
        out[f"HC_p{int(q*100)}"] = g.quantile(q)

    out = out.reset_index()

    # Gains and ratios (using P95 if available, else highest quantile)
    qmax = max(q_list)
    qcol = f"HC_p{int(qmax*100)}"
    out["HC_nonfirm"] = out[qcol]
    out["gain_nonfirm_minus_firm"] = out["HC_nonfirm"] - out["HC_firm_min"]

    # Ratio (avoid division by zero; if firm=0, set NaN)
    out["ratio_nonfirm_to_firm"] = np.where(
        out["HC_firm_min"] > 0,
        out["HC_nonfirm"] / out["HC_firm_min"],
        np.nan
    )

    return out


# -----------------------------
# Plotting
# -----------------------------
def plot_hist(series: pd.Series, title: str, xlabel: str, outpath: Path, bins=50, logy=False):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = series.dropna().values
    ax.hist(x, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    if logy:
        ax.set_yscale("log")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_scatter(x: pd.Series, y: pd.Series, title: str, xlabel: str, ylabel: str, outpath: Path):
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(x, y, s=8)

    # 45-degree line
    xy = pd.concat([x, y], axis=1).dropna()
    if not xy.empty:
        m = max(xy.max().max(), 1e-6)
        ax.plot([0, m], [0, m], linestyle="--", linewidth=1.0)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def summarize_global(df_bus: pd.DataFrame) -> pd.DataFrame:
    cols = ["HC_firm_min", "HC_nonfirm", "gain_nonfirm_minus_firm", "ratio_nonfirm_to_firm"]
    summ = {}
    for c in cols:
        s = df_bus[c].dropna()
        summ[c + "_min"] = s.min() if len(s) else np.nan
        summ[c + "_p05"] = s.quantile(0.05) if len(s) else np.nan
        summ[c + "_p50"] = s.quantile(0.50) if len(s) else np.nan
        summ[c + "_p95"] = s.quantile(0.95) if len(s) else np.nan
        summ[c + "_max"] = s.max() if len(s) else np.nan
        summ[c + "_mean"] = s.mean() if len(s) else np.nan
    return pd.DataFrame([summ])


# -----------------------------
# Main (no argparse)
# -----------------------------
def main():
    wd = Path(os.getcwd())
    root = wd / "pola_results"
    if not root.exists():
        raise FileNotFoundError(f"Expected folder not found: {root}")

    outdir = wd / "ch4_figures_all"
    outdir.mkdir(parents=True, exist_ok=True)

    df_all = read_all_capmaps(root)
    demand, gen = split_modes(df_all)

    # Compute firm/non-firm per bus
    df_dem_bus = firm_nonfirm_by_bus(demand, q_list=(0.90, 0.95))
    df_gen_bus = firm_nonfirm_by_bus(gen, q_list=(0.90, 0.95))

    # Save tables
    df_dem_bus.to_csv(outdir / "firm_nonfirm_byBus_demand.csv", index=False)
    df_gen_bus.to_csv(outdir / "firm_nonfirm_byBus_generation.csv", index=False)

    summarize_global(df_dem_bus).to_csv(outdir / "firm_nonfirm_summary_demand.csv", index=False)
    summarize_global(df_gen_bus).to_csv(outdir / "firm_nonfirm_summary_generation.csv", index=False)

    # Figures: firm vs non-firm distributions
    plot_hist(df_dem_bus["HC_firm_min"],
              "Demand: Firm hosting capacity (min over time)",
              "HC_firm [kW]", outdir / "hist_firm_demand.png")

    plot_hist(df_dem_bus["HC_nonfirm"],
              "Demand: Non-firm hosting capacity (P95 over time)",
              "HC_nonfirm,P95 [kW]", outdir / "hist_nonfirm_p95_demand.png")

    plot_hist(df_dem_bus["gain_nonfirm_minus_firm"],
              "Demand: Non-firm gain (P95 - Firm)",
              "Gain [kW]", outdir / "hist_gain_p95_minus_firm_demand.png")

    plot_hist(df_dem_bus["ratio_nonfirm_to_firm"],
              "Demand: Non-firm ratio (P95 / Firm)",
              "Ratio [-]", outdir / "hist_ratio_p95_over_firm_demand.png", logy=True)

    plot_scatter(df_dem_bus["HC_firm_min"], df_dem_bus["HC_nonfirm"],
                 "Demand: Firm vs Non-firm (P95) hosting capacity",
                 "Firm HC (min over time) [kW]",
                 "Non-firm HC (P95 over time) [kW]",
                 outdir / "scatter_firm_vs_nonfirm_p95_demand.png")

    # Same for generation
    plot_hist(df_gen_bus["HC_firm_min"],
              "Generation: Firm hosting capacity (min over time)",
              "HC_firm [kW]", outdir / "hist_firm_generation.png")

    plot_hist(df_gen_bus["HC_nonfirm"],
              "Generation: Non-firm hosting capacity (P95 over time)",
              "HC_nonfirm,P95 [kW]", outdir / "hist_nonfirm_p95_generation.png")

    plot_hist(df_gen_bus["gain_nonfirm_minus_firm"],
              "Generation: Non-firm gain (P95 - Firm)",
              "Gain [kW]", outdir / "hist_gain_p95_minus_firm_generation.png")

    plot_hist(df_gen_bus["ratio_nonfirm_to_firm"],
              "Generation: Non-firm ratio (P95 / Firm)",
              "Ratio [-]", outdir / "hist_ratio_p95_over_firm_generation.png", logy=True)

    plot_scatter(df_gen_bus["HC_firm_min"], df_gen_bus["HC_nonfirm"],
                 "Generation: Firm vs Non-firm (P95) hosting capacity",
                 "Firm HC (min over time) [kW]",
                 "Non-firm HC (P95 over time) [kW]",
                 outdir / "scatter_firm_vs_nonfirm_p95_generation.png")

    print("[OK] Wrote firm/non-firm figures + tables to:", outdir.resolve())


if __name__ == "__main__":
    main()
