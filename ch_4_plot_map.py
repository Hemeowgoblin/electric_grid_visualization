import os
from pathlib import Path
import sqlite3
import platform
import subprocess
import time
import psutil
import ctypes
import ctypes.wintypes as wintypes
import uuid

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


# ---------- Settings ----------
OUT_W_MM = 175   # width in mm
OUT_H_MM = 132   # height in mm
INCH_PER_MM = 1 / 25.4

CRITICAL_HOUR_DEMAND = 12

# ---------- Line Settings ----------
PLOT_LINES = False  # Now enabled (black lines)

LINE_WIDTH = 0.6  # Width of the line in points. Range: 0 (no line) to ~2+ (thick line). Typical range: 0.1-1.0.
LINE_ALPHA = 0.8  # Transparency of the line. Range: 0 (invisible) to 1 (fully opaque).
LINE_COLOR = "black"  # Color of the line. Use any matplotlib color name (e.g., "black", "white", "red") or hex code (e.g., "#000000").
LINE_DYNAMIC_COLOR = True  # If True, color lines by average HC of connected buses (overrides LINE_COLOR)
LINE_GROUP_CONNECTIVITY = True  # If True, use connected line groups for color (all lines in a connected group share the same color)
LINE_GROUP_CONNECTIVITY_WITH_SWITCHES_AND_TRANSFORMERS = False  # If True, include switches and transformers when determining connected line groups

# ---------- Proper Line (Line Segments) with Segments Settings ----------
PLOT_PROPER_LINES = True  # Toggle proper segment-based lines on/off

PROPER_LINE_WIDTH = 0.6   # Width of proper lines
PROPER_LINE_ALPHA = 0.8   # Transparency of proper lines
PROPER_LINE_COLOR = "blue"  # Color of proper lines (blue to distinguish from black direct lines)
PROPER_LINE_DYNAMIC_COLOR = True  # If True, color lines by average HC of connected buses (overrides PROPER_LINE_COLOR)
PROPER_LINE_GROUP_CONNECTIVITY = True  # If True, use connected line groups for color (all lines in a connected group share the same color)
PROPER_LINE_GROUP_CONNECTIVITY_WITH_SWITCHES_AND_TRANSFORMERS = True  # If True, include switches and transformers when determining connected line groups

# ---------- Circle Settings ----------
PLOT_CIRCLES = True  # Toggle circle points on/off (master toggle)

# Circle fill (inside)
PLOT_CIRCLE_FILL = True  # Toggle circle fill color on/off. When True, circles are colored by HC values using CIRCLE_CMAP. When False, circles appear in a plain gray color.
CIRCLE_SIZE = 5  # Size of circles in points. Larger values make circles more visible. Typical range: 5-20.
CIRCLE_CMAP = "RdYlGn"  # Color map for circle fill. "RdYlGn" maps red (low HC) → yellow (medium) → green (high HC). Other options: "viridis", "plasma", "cool", "hot". Only used if PLOT_CIRCLE_FILL is True.
CIRCLE_ALPHA = 1.0  # Transparency of circle fill. Range: 0 (invisible) to 1 (fully opaque). Lower values make overlapping circles more visible.

# Circle border (outline)
PLOT_CIRCLE_EDGE = True  # Toggle circle border/outline on/off. When True, circles have a visible edge. When False, circles have no outline.
CIRCLE_EDGE_COLOR = "black"  # Color of the circle outline. Use any matplotlib color name (e.g., "black", "white", "red") or hex code (e.g., "#000000").
CIRCLE_EDGE_WIDTH = 0.5  # Thickness of the circle outline in points. Range: 0 (no outline) to ~2+ (thick outline). Typical range: 0.1-1.0.

# ---------- Background Settings ----------
BACKGROUND_COLOR = "gray"  # Background color of the plot. Use any matplotlib color name or hex code. "lightgray" provides good contrast for both dark and light elements.

# ---------- Output Settings ----------
SAVE_HIGH_RES = True   # Save additional high-resolution PNG version
HIGH_RES_DPI = 900    # DPI for high-resolution version (default is ~100)
SAVE_SVG = True       # Save vector SVG version


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
    switch = pd.read_sql_query("SELECT * FROM Switch;", con)
    transformer = pd.read_sql_query("SELECT * FROM Transformer;", con)
    con.close()
    return bus, cgp, line, switch, transformer


def cgp_coordinates(bus: pd.DataFrame, cgp: pd.DataFrame) -> pd.DataFrame:
    bus = bus.rename(columns={"ID": "BusId"})
    if "Bus" in cgp.columns and "BusId" not in cgp.columns:
        cgp = cgp.rename(columns={"Bus": "BusId"})
    out = cgp.merge(bus[["BusId", "X", "Y"]], on="BusId", how="left")
    out = out.drop_duplicates(subset=["BusId"])
    return out[["BusId", "X", "Y"]]


def line_segments(bus: pd.DataFrame, line: pd.DataFrame, hc_data: pd.DataFrame = None, group_hc_df: pd.DataFrame = None, line_group_df: pd.DataFrame = None) -> pd.DataFrame:
    if line.empty:
        return pd.DataFrame()
    bus = bus.rename(columns={"ID": "BusId"})
    b1 = bus[["BusId", "X", "Y"]].rename(columns={"BusId": "Bus1", "X": "X1", "Y": "Y1"})
    b2 = bus[["BusId", "X", "Y"]].rename(columns={"BusId": "Bus2", "X": "X2", "Y": "Y2"})
    seg = line.merge(b1, on="Bus1", how="left").merge(b2, on="Bus2", how="left")
    
    hc_lookup = {}
    if hc_data is not None and not hc_data.empty:
        hc_lookup = hc_data.set_index("BusId")["HC"].to_dict()
    
    group_hc_lookup = {}
    line_to_group = {}
    if group_hc_df is not None and not group_hc_df.empty and line_group_df is not None and not line_group_df.empty:
        group_hc_lookup = group_hc_df.set_index("GroupID")["AvgHC"].to_dict()
        line_to_group = line_group_df.set_index("LineID")["GroupID"].to_dict()
    
    results = []
    for _, row in seg.iterrows():
        line_id = row.get("ID")
        bus1, bus2 = row.get("Bus1"), row.get("Bus2")
        x1, y1 = row.get("X1"), row.get("Y1")
        x2, y2 = row.get("X2"), row.get("Y2")
        
        avg_hc = None
        
        if group_hc_lookup and line_id in line_to_group:
            group_id = line_to_group[line_id]
            if group_id in group_hc_lookup:
                avg_hc = group_hc_lookup[group_id]
        elif hc_lookup:
            hc1 = hc_lookup.get(bus1)
            hc2 = hc_lookup.get(bus2)
            
            if hc1 is not None and hc2 is not None:
                avg_hc = (hc1 + hc2) / 2
            elif hc1 is not None:
                avg_hc = hc1
            elif hc2 is not None:
                avg_hc = hc2
            else:
                pass
        
        results.append({"X1": x1, "Y1": y1, "X2": x2, "Y2": y2, "HC": avg_hc})
    
    return pd.DataFrame(results)


def proper_line_segments(bus: pd.DataFrame, line: pd.DataFrame, db_path: Path, hc_data: pd.DataFrame = None, group_hc_df: pd.DataFrame = None, line_group_df: pd.DataFrame = None) -> pd.DataFrame:
    if line.empty:
        return pd.DataFrame()
    
    con = sqlite3.connect(db_path)
    seg = pd.read_sql_query("SELECT Line, ID, X, Y FROM Seg ORDER BY Line, ID", con)
    con.close()
    
    if seg.empty:
        return pd.DataFrame()
    
    bus = bus.rename(columns={"ID": "BusId"})
    bus_xy = bus.set_index("BusId")[["X", "Y"]].to_dict("index")
    
    hc_lookup = {}
    if hc_data is not None and not hc_data.empty:
        hc_lookup = hc_data.set_index("BusId")["HC"].to_dict()
    
    group_hc_lookup = {}
    line_to_group = {}
    if group_hc_df is not None and not group_hc_df.empty and line_group_df is not None and not line_group_df.empty:
        group_hc_lookup = group_hc_df.set_index("GroupID")["AvgHC"].to_dict()
        line_to_group = line_group_df.set_index("LineID")["GroupID"].to_dict()
    
    results = []
    for _, row in line.iterrows():
        line_id = row["ID"]
        bus1, bus2 = row.get("Bus1"), row.get("Bus2")
        seg1, seg2 = row.get("Seg1"), row.get("Seg2")
        
        if pd.notna(seg1) and pd.notna(seg2):
            mask = (seg["Line"] == row["ID"]) & (seg["ID"] >= seg1) & (seg["ID"] <= seg2)
            pts = seg.loc[mask, ["X", "Y"]].values.tolist()
        else:
            pts = []
            if bus1 in bus_xy:
                pts.append([bus_xy[bus1]["X"], bus_xy[bus1]["Y"]])
            if bus2 in bus_xy:
                pts.append([bus_xy[bus2]["X"], bus_xy[bus2]["Y"]])
        
        avg_hc = None
        
        if group_hc_lookup and line_id in line_to_group:
            group_id = line_to_group[line_id]
            if group_id in group_hc_lookup:
                avg_hc = group_hc_lookup[group_id]
        elif hc_lookup:
            hc1 = hc_lookup.get(bus1)
            hc2 = hc_lookup.get(bus2)
            
            if hc1 is not None and hc2 is not None:
                avg_hc = (hc1 + hc2) / 2
            elif hc1 is not None:
                avg_hc = hc1
                # print(f"[INFO] Proper Line {row.get('ID')}: Bus1 has HC, Bus2 does not. Using Bus1 HC={hc1}")
            elif hc2 is not None:
                avg_hc = hc2
                # print(f"[INFO] Proper Line {row.get('ID')}: Bus2 has HC, Bus1 does not. Using Bus2 HC={hc2}")
            else:
                pass
                # print(f"[WARN] Proper Line {row.get('ID')}: No HC data for Bus1={bus1} or Bus2={bus2}")
        
        if len(pts) >= 2:
            results.append({"X": [p[0] for p in pts], "Y": [p[1] for p in pts], "HC": avg_hc})
    
    return pd.DataFrame(results)


def find_connected_line_groups(line: pd.DataFrame) -> pd.DataFrame:
    """
    Find groups of electrically connected lines using Union-Find.
    Lines are connected if they share a common bus.
    
    Returns: DataFrame with columns [LineID, GroupID]
    """
    from collections import defaultdict
    
    if line.empty:
        return pd.DataFrame(columns=["LineID", "GroupID"])
    
    bus_to_lines = defaultdict(set)
    for _, row in line.iterrows():
        line_id = row["ID"]
        bus_to_lines[row["Bus1"]].add(line_id)
        bus_to_lines[row["Bus2"]].add(line_id)
    
    parent = {}
    
    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    for bus, lines in bus_to_lines.items():
        lines_list = list(lines)
        for i in range(1, len(lines_list)):
            union(lines_list[0], lines_list[i])
    
    group_map = {}
    for line_id in parent.keys():
        group_map[line_id] = find(line_id)
    
    return pd.DataFrame(list(group_map.items()), columns=["LineID", "GroupID"])


def find_connected_line_groups_with_switches_and_transformers(
    line: pd.DataFrame,
    switch: pd.DataFrame,
    transformer: pd.DataFrame
) -> pd.DataFrame:
    """
    Find groups of electrically connected lines using Union-Find.
    Lines are connected if they:
    1. Share a common bus, OR
    2. Are connected through closed switches (State1=1 AND State2=1), OR
    3. Are connected through transformers
    
    Returns: DataFrame with columns [LineID, GroupID]
    """
    from collections import defaultdict
    
    if line.empty:
        return pd.DataFrame(columns=["LineID", "GroupID"])
    
    parent = {}
    
    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    for _, row in line.iterrows():
        line_id = row["ID"]
        find(line_id)
        union(line_id, row["Bus1"])
        union(line_id, row["Bus2"])
    
    if switch is not None and not switch.empty:
        closed_switches = switch[(switch["State1"] == 1) & (switch["State2"] == 1)]
        for _, row in closed_switches.iterrows():
            find(row["Bus1"])
            find(row["Bus2"])
            union(row["Bus1"], row["Bus2"])
    
    if transformer is not None and not transformer.empty:
        for _, row in transformer.iterrows():
            if pd.notna(row.get("Bus1")) and pd.notna(row.get("Bus2")):
                find(row["Bus1"])
                find(row["Bus2"])
                union(row["Bus1"], row["Bus2"])
    
    group_map = {}
    for line_id in line["ID"]:
        group_map[line_id] = find(line_id)
    
    return pd.DataFrame(list(group_map.items()), columns=["LineID", "GroupID"])


def calculate_group_hc(line_group_df: pd.DataFrame, line: pd.DataFrame, hc_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average HC for each line group based on ALL circles connected to that group.
    
    Args:
        line_group_df: DataFrame with [LineID, GroupID]
        line: DataFrame with [ID, Bus1, Bus2]
        hc_data: DataFrame with [BusId, HC]
    
    Returns: DataFrame with [GroupID, AvgHC]
    """
    if line_group_df.empty or line.empty or hc_data.empty:
        return pd.DataFrame(columns=["GroupID", "AvgHC"])
    
    hc_lookup = hc_data.set_index("BusId")["HC"].to_dict()
    line_to_group = line_group_df.set_index("LineID")["GroupID"].to_dict()
    
    line = line.rename(columns={"ID": "LineID"})
    line_with_group = line.merge(line_group_df, on="LineID", how="left")
    
    results = []
    for group_id in line_group_df["GroupID"].unique():
        group_lines = line_with_group[line_with_group["GroupID"] == group_id]
        
        all_buses = set()
        for _, lrow in group_lines.iterrows():
            all_buses.add(lrow["Bus1"])
            all_buses.add(lrow["Bus2"])
        
        hc_values = []
        for bus_id in all_buses:
            if bus_id in hc_lookup and pd.notna(hc_lookup[bus_id]):
                hc_values.append(hc_lookup[bus_id])
        
        if hc_values:
            avg_hc = sum(hc_values) / len(hc_values)
        else:
            avg_hc = None
            # print(f"[WARN] Group {group_id}: No HC data for any of {len(all_buses)} buses")
        
        results.append({"GroupID": group_id, "AvgHC": avg_hc})
    
    return pd.DataFrame(results)


def _find_window_pids_with_title(substring: str):
    """Return list of PIDs for visible top-level windows whose title contains substring (case-insensitive)."""
    user32 = ctypes.windll.user32
    substring_l = substring.lower()
    pids = set()

    CALLBACK = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    @CALLBACK
    def _enum_proc(hwnd, lparam):
        try:
            if not user32.IsWindowVisible(hwnd):
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            if length == 0:
                return True
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value
            if substring_l in title.lower():
                pid = wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                pids.add(pid.value)
        except Exception:
            pass
        return True

    try:
        user32.EnumWindows(_enum_proc, 0)
    except Exception:
        pass

    return list(pids)


def close_file_viewers_windows(file_path: Path):
    """
    Detect windows showing the file (by window title), attempt to terminate those processes,
    and report detected/closed/remaining counts.
    """
    try:
        target_name = file_path.name
        print("\n[STEP 1] Detecting open instances by window title...")
        detected_pids = _find_window_pids_with_title(target_name)
        detected = []
        for pid in detected_pids:
            try:
                name = psutil.Process(pid).name()
            except Exception:
                name = "<unknown>"
            detected.append((pid, name))

        print(f"[DETECTED] {len(detected)} open instance(s) found")
        for pid, name in detected:
            print(f"  - PID {pid}: {name}")

        # Step 2: close processes
        print(f"\n[STEP 2] Closing {len(detected)} instance(s)...")
        closed = 0
        for pid, name in detected:
            try:
                proc = psutil.Process(pid)
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                    closed += 1
                    print(f"  ✓ Terminated PID {pid} ({name})")
                except psutil.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=3)
                    closed += 1
                    print(f"  ✓ Force killed PID {pid} ({name})")
            except Exception as e:
                print(f"  ✗ Failed to close PID {pid}: {e}")

        print(f"[CLOSED] {closed}/{len(detected)} instance(s) closed")

        # Step 3: verify remaining
        print(f"\n[STEP 3] Verifying remaining open instances...")
        time.sleep(1)
        remaining_pids = _find_window_pids_with_title(target_name)
        remaining = []
        for pid in remaining_pids:
            try:
                name = psutil.Process(pid).name()
            except Exception:
                name = "<unknown>"
            remaining.append((pid, name))

        print(f"[REMAINING] {len(remaining)} open instance(s) still detected")
        for pid, name in remaining:
            print(f"  - PID {pid}: {name}")
        if not remaining:
            print("  ✓ File successfully released (no windows with filename in title detected)")
        else:
            print("  ! Some windows still display the file; close them manually if needed")

    except Exception as e:
        print(f"[WARN] Error during window-title based closing: {e}")
        # fallback: best-effort psutil open_files scan (may miss UWP viewers)
        try:
            print("[INFO] Falling back to psutil.open_files scan...")
            file_path_str = str(file_path).lower()
            open_processes = []
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    for of in proc.open_files():
                        if file_path_str == of.path.lower():
                            open_processes.append((proc.pid, proc.name(), proc))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            print(f"[DETECTED-PSUTIL] {len(open_processes)} instance(s) found")
            closed2 = 0
            for pid, name, proc in open_processes:
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                    closed2 += 1
                except Exception:
                    try:
                        proc.kill()
                        proc.wait(timeout=2)
                        closed2 += 1
                    except Exception:
                        pass
            print(f"[CLOSED-PSUTIL] {closed2}/{len(open_processes)} instance(s) closed")
        except Exception as e2:
            print(f"[WARN] psutil fallback failed: {e2}")


def main():
    wd = Path(os.getcwd())
    root = wd / "pola_results"
    outdir = wd / "ch4_figures_all"
    outdir.mkdir(parents=True, exist_ok=True)

    outpath_template = f"map_demand_global_hour{CRITICAL_HOUR_DEMAND:02d}_175x132mm.png"
    outpath_full = outdir / outpath_template

    # Close file viewers BEFORE creating new file (Windows-specific)
    if platform.system() == "Windows":
        close_file_viewers_windows(outpath_full)
    else:
        print("[INFO] File closing only implemented for Windows")

    pts_all = []
    seg_all = []
    proper_seg_all = []
    all_lines = []
    all_hc_data = []
    all_buses = []
    all_switches = []
    all_transformers = []

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

        bus, cgp, line, switch, transformer = read_db_tables(db_path)
        cgp_xy = cgp_coordinates(bus, cgp)

        dem = dem.merge(cgp_xy, on="BusId", how="left")
        pts_all.append(dem[["X", "Y", "HC"]])
        
        if PLOT_LINES:
            seg = line_segments(bus, line, dem[["BusId", "HC"]])
            if not seg.empty:
                seg_all.append(seg)
        
        if PLOT_LINES and LINE_GROUP_CONNECTIVITY:
            all_lines.append(line)
            all_switches.append(switch)
            all_transformers.append(transformer)
            all_hc_data.append(dem[["BusId", "HC"]])
            all_buses.append(bus)
        
        if PLOT_PROPER_LINES and PROPER_LINE_GROUP_CONNECTIVITY:
            all_lines.append(line)
            all_switches.append(switch)
            all_transformers.append(transformer)
            all_hc_data.append(dem[["BusId", "HC"]])
            all_buses.append(bus)
        
        if PLOT_PROPER_LINES and not PROPER_LINE_GROUP_CONNECTIVITY:
            proper_seg = proper_line_segments(bus, line, db_path, dem[["BusId", "HC"]])
            if not proper_seg.empty:
                proper_seg_all.append(proper_seg)

    pts = pd.concat(pts_all, ignore_index=True).dropna(subset=["X", "Y", "HC"])
    # Don't concatenate here - wait until after group connectivity processing
    
    # Calculate line groups and group HC once (can be used for both normal lines and proper lines)
    lines_for_grouping = []
    use_switches_and_transformers = False
    
    if PLOT_LINES and LINE_GROUP_CONNECTIVITY:
        lines_for_grouping = all_lines
        use_switches_and_transformers = LINE_GROUP_CONNECTIVITY_WITH_SWITCHES_AND_TRANSFORMERS
    elif PLOT_PROPER_LINES and PROPER_LINE_GROUP_CONNECTIVITY:
        lines_for_grouping = all_lines
        use_switches_and_transformers = PROPER_LINE_GROUP_CONNECTIVITY_WITH_SWITCHES_AND_TRANSFORMERS
    
    if lines_for_grouping:
        all_lines_df = pd.concat(lines_for_grouping, ignore_index=True)
        all_hc_df = pd.concat(all_hc_data, ignore_index=True).dropna(subset=["BusId", "HC"])
        all_buses_df = pd.concat(all_buses, ignore_index=True)
        
        if use_switches_and_transformers:
            all_switches_df = pd.concat(all_switches, ignore_index=True) if all_switches else pd.DataFrame()
            all_transformers_df = pd.concat(all_transformers, ignore_index=True) if all_transformers else pd.DataFrame()
            line_group_df = find_connected_line_groups_with_switches_and_transformers(
                all_lines_df, all_switches_df, all_transformers_df
            )
        else:
            line_group_df = find_connected_line_groups(all_lines_df)
        group_hc_df = calculate_group_hc(line_group_df, all_lines_df, all_hc_df)
        
        # Process normal lines with group connectivity if enabled
        if PLOT_LINES and LINE_GROUP_CONNECTIVITY:
            seg_all.clear()  # Clear first pass results
            for st_dir in station_dirs:
                db_path = st_dir / "net.db"
                if not db_path.exists():
                    continue
                
                cap_path = st_dir / "out_6_3" / "CapMap_chk.txt"
                if not cap_path.exists():
                    continue
                
                cap = read_capmap(cap_path)
                dem = cap[(cap["F"] >= 0) & (cap["F"] <= 23)].copy()
                dem["hour"] = dem["F"].astype(int)
                dem = dem[dem["hour"] == CRITICAL_HOUR_DEMAND].copy()
                
                for c in ["PA", "PB", "PC"]:
                    dem[c] = pd.to_numeric(dem[c], errors="coerce")
                
                dem["HC"] = dem[["PA", "PB", "PC"]].min(axis=1)
                
                bus, cgp, line, switch, transformer = read_db_tables(db_path)
                seg = line_segments(bus, line, dem[["BusId", "HC"]], group_hc_df, line_group_df)
                if not seg.empty:
                    seg_all.append(seg)
        
        # Process proper lines with group connectivity if enabled
        if PLOT_PROPER_LINES and PROPER_LINE_GROUP_CONNECTIVITY:
            proper_seg_all.clear()  # Clear first pass results
            for st_dir in station_dirs:
                db_path = st_dir / "net.db"
                if not db_path.exists():
                    continue
                
                cap_path = st_dir / "out_6_3" / "CapMap_chk.txt"
                if not cap_path.exists():
                    continue
                
                cap = read_capmap(cap_path)
                dem = cap[(cap["F"] >= 0) & (cap["F"] <= 23)].copy()
                dem["hour"] = dem["F"].astype(int)
                dem = dem[dem["hour"] == CRITICAL_HOUR_DEMAND].copy()
                
                for c in ["PA", "PB", "PC"]:
                    dem[c] = pd.to_numeric(dem[c], errors="coerce")
                
                dem["HC"] = dem[["PA", "PB", "PC"]].min(axis=1)
                
                bus, cgp, line, switch, transformer = read_db_tables(db_path)
                proper_seg = proper_line_segments(bus, line, db_path, dem[["BusId", "HC"]], group_hc_df, line_group_df)
                if not proper_seg.empty:
                    proper_seg_all.append(proper_seg)
        
        # Concatenate results after second pass
        seg = pd.concat(seg_all, ignore_index=True) if seg_all else pd.DataFrame()
        proper_seg = pd.concat(proper_seg_all, ignore_index=True) if proper_seg_all else pd.DataFrame()
    else:
        line_group_df = pd.DataFrame()
        group_hc_df = pd.DataFrame()
        # Regular processing (no group connectivity)
        seg = pd.concat(seg_all, ignore_index=True) if seg_all else pd.DataFrame()
        proper_seg = pd.concat(proper_seg_all, ignore_index=True) if proper_seg_all else pd.DataFrame()

    # --------- Plot ---------
    fig_w_in = OUT_W_MM * INCH_PER_MM
    fig_h_in = OUT_H_MM * INCH_PER_MM

    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))
    ax.set_facecolor(BACKGROUND_COLOR)

    hc_min = pts["HC"].min()
    hc_max = pts["HC"].max()

    if not seg.empty:
        for _, r in seg.iterrows():
            if LINE_DYNAMIC_COLOR and pd.notna(r.get("HC")):
                if hc_max > hc_min:
                    normalized = (r["HC"] - hc_min) / (hc_max - hc_min)
                else:
                    normalized = 0.5
                color = plt.cm.get_cmap(CIRCLE_CMAP)(normalized)
            else:
                color = LINE_COLOR
            ax.plot(
                [r["X1"], r["X2"]],
                [r["Y1"], r["Y2"]],
                linewidth=LINE_WIDTH,
                color=color,
                alpha=LINE_ALPHA
            )

    if PLOT_PROPER_LINES and not proper_seg.empty:
        for _, r in proper_seg.iterrows():
            if PROPER_LINE_DYNAMIC_COLOR and pd.notna(r.get("HC")):
                if hc_max > hc_min:
                    normalized = (r["HC"] - hc_min) / (hc_max - hc_min)
                else:
                    normalized = 0.5
                color = plt.cm.get_cmap(CIRCLE_CMAP)(normalized)
            else:
                color = PROPER_LINE_COLOR
            ax.plot(
                r["X"],
                r["Y"],
                linewidth=PROPER_LINE_WIDTH,
                color=color,
                alpha=PROPER_LINE_ALPHA
            )

    if PLOT_CIRCLES:
        # Determine edge properties based on toggle
        edge_color = CIRCLE_EDGE_COLOR if PLOT_CIRCLE_EDGE else "none"
        edge_width = CIRCLE_EDGE_WIDTH if PLOT_CIRCLE_EDGE else 0

        sc = ax.scatter(
            pts["X"],
            pts["Y"],
            c=pts["HC"] if PLOT_CIRCLE_FILL else "lightgray",
            s=CIRCLE_SIZE,
            cmap=CIRCLE_CMAP if PLOT_CIRCLE_FILL else None,
            alpha=CIRCLE_ALPHA if PLOT_CIRCLE_FILL else 0.3,
            edgecolors=edge_color,
            linewidth=edge_width,
            vmin=hc_min if PLOT_CIRCLE_FILL else None,
            vmax=hc_max if PLOT_CIRCLE_FILL else None
        )

        if PLOT_CIRCLE_FILL:
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
    print("[OK] Saved:", outpath.resolve())

    if SAVE_HIGH_RES:
        outpath_hires = outdir / f"map_demand_global_hour{CRITICAL_HOUR_DEMAND:02d}_175x132mm_high_resolution.png"
        fig.savefig(outpath_hires, dpi=HIGH_RES_DPI)
        print("[OK] Saved high-res:", outpath_hires.resolve())

    if SAVE_SVG:
        outpath_svg = outdir / f"map_demand_global_hour{CRITICAL_HOUR_DEMAND:02d}_175x132mm_vector.svg"
        fig.savefig(outpath_svg, format="svg")
        print("[OK] Saved SVG:", outpath_svg.resolve())

    plt.close(fig)

    # Open the output file with a delay to ensure file is fully written
    time.sleep(0.5)
    try:
        if platform.system() == "Windows":
            os.startfile(str(outpath))
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", str(outpath)])
        else:  # Linux
            subprocess.run(["xdg-open", str(outpath)])
    except Exception as e:
        print(f"[WARN] Could not open file: {e}")

    total_open_switches = sum(len(s[(s["State1"] != 1) | (s["State2"] != 1)]) for s in all_switches if not s.empty)
    total_closed_switches = sum(len(s[(s["State1"] == 1) & (s["State2"] == 1)]) for s in all_switches if not s.empty)
    total_transformers = sum(len(t) for t in all_transformers if not t.empty)

    print(f"[INFO] Total open switches: {total_open_switches}")
    print(f"[INFO] Total closed switches: {total_closed_switches}")
    print(f"[INFO] Total transformers: {total_transformers}")
    print("[OK] Code finished without problems.")


if __name__ == "__main__":
    main()
