import os
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- Settings ----------
OUT_W_MM = 175   # width in mm
OUT_H_MM = 132   # height in mm
INCH_PER_MM = 1 / 25.4

CRITICAL_HOUR_DEMAND = 12
PLOT_LINES = True  # Now enabled (black lines)


def read_capmap(cap_path: Path) -> pd.DataFrame:
    df = pd.read_csv(cap_path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    return df


def read_db_tables(db_path: Path):
    con = sqlite3.connect(db_path)
    bus = pd.read_sql_query("SELECT * FROM Bus;", con)
    cgp = pd.read_sql_query("SELECT * FROM CGP;", con)
    try:
        line = pd.read_sql_query("SELECT * FROM Line;", con)
    except Exception:
        line = pd.DataFrame()
    con.close()
    return bus, cgp, line


def cgp_coordinates(bus: pd.DataFrame, cgp: pd.DataFrame) -> pd.DataFrame:
    bus = bus.rename(columns={"ID": "BusId"})
    if "Bus" in cgp.columns and "BusId" not in cgp.columns:
        cgp = cgp.rename(columns={"Bus": "BusId"})
    out = cgp.merge(bus[["BusId", "X", "Y"]], on="BusId", how="left")
    out = out.drop_duplicates(subset=["BusId"])
    return out[["BusId", "X", "Y"]]


def line_segments(bus: pd.DataFrame, line: pd.DataFrame) -> pd.DataFrame:
    if line.empty:
        return pd.DataFrame()
    bus = bus.rename(columns={"ID": "BusId"})
    b1 = bus[["BusId", "X", "Y"]].rename(columns={"BusId": "Bus1", "X": "X1", "Y": "Y1"})
    b2 = bus[["BusId", "X", "Y"]].rename(columns={"BusId": "Bus2", "X": "X2", "Y": "Y2"})
    seg = line.merge(b1, on="Bus1", how="left").merge(b2, on="Bus2", how="left")
    return seg[["X1", "Y1", "X2", "Y2"]]


def main():
    wd = Path(os.getcwd())
    root = wd / "pola_results"
    outdir = wd / "ch4_figures_all"
    outdir.mkdir(parents=True, exist_ok=True)

    pts_all = []
    seg_all = []

    station_dirs = sorted([p for p in root.iterdir() if p.is_dir()])

    for st_dir in station_dirs:
        cap_path = st_dir / "out_6_3" / "CapMap_chk.txt"
        db_path = st_dir / "net.db"
        if not cap_path.exists() or not db_path.exists():
            continue

        cap = read_capmap(cap_path)

        dem = cap[(cap["F"] >= 0) & (cap["F"] <= 23)].copy()
        dem["hour"] = dem["F"].astype(int)
        dem = dem[dem["hour"] == CRITICAL_HOUR_DEMAND].copy()

        for c in ["PA", "PB", "PC"]:
            dem[c] = pd.to_numeric(dem[c], errors="coerce")

        dem["HC"] = dem[["PA", "PB", "PC"]].min(axis=1)

        bus, cgp, line = read_db_tables(db_path)
        cgp_xy = cgp_coordinates(bus, cgp)

        dem = dem.merge(cgp_xy, on="BusId", how="left")
        pts_all.append(dem[["X", "Y", "HC"]])

        if PLOT_LINES:
            seg = line_segments(bus, line)
            if not seg.empty:
                seg_all.append(seg)

    pts = pd.concat(pts_all, ignore_index=True).dropna(subset=["X", "Y", "HC"])
    seg = pd.concat(seg_all, ignore_index=True) if seg_all else pd.DataFrame()

    # --------- Plot ---------
    fig_w_in = OUT_W_MM * INCH_PER_MM
    fig_h_in = OUT_H_MM * INCH_PER_MM

    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))

    if not seg.empty:
        seg = seg.dropna()
        for _, r in seg.iterrows():
            ax.plot(
                [r["X1"], r["X2"]],
                [r["Y1"], r["Y2"]],
                linewidth=0.3,
                color="black",
                alpha=1.0
            )

    sc = ax.scatter(
        pts["X"],
        pts["Y"],
        c=pts["HC"],
        s=6,
        cmap="viridis"
    )

    cb = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Demand hosting capacity [kW] (min across phases)")

    ax.set_title(
        f"Demand nodal hosting capacity at critical hour (t={CRITICAL_HOUR_DEMAND:02d})"
    )

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()

    outpath = outdir / f"map_demand_global_hour{CRITICAL_HOUR_DEMAND:02d}_175x132mm.png"
    fig.savefig(outpath)
    plt.close(fig)

    print("[OK] Saved:", outpath.resolve())


if __name__ == "__main__":
    main()
