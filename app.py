# Nutcracker Sales Trajectory Tracker (Streamlit)
# -------------------------------------------------
# Purpose
# - Compare this year's Nutcracker sales trajectory to prior years
# - Normalize for different numbers of performances (and capacity if available)
# - Project final sales using historical cumulative share curves
# - Handle Calgary vs Edmonton (or combined) views
# - **Reads CSVs from a local `data/` folder in the repo** (no file uploads)
# - **Timeline anchored to closing day (Dec 24) each year**
#
# Data expectations (CSV)
# -----------------------
# Place CSVs in `data/` at the repo root. The app will auto-discover them.
# You can either use fixed names:
#   - `data/historical.csv`      (multi-year history)
#   - `data/this_year.csv`       (this-season to date; exactly one season)
# or use patterns (the app picks the most recent by modified time):
#   - `data/historical_*.csv`
#   - `data/this_year_*.csv`
#
# Required columns (case-insensitive, flexible names allowed):
#   - season / year             : e.g., 2019, 2022-23, etc. (string or int)
#   - order_date / sale_date    : date of ticket sale (YYYY-MM-DD)
#   - performance_date          : date of first performance (YYYY-MM-DD) for the Nutcracker run in that city
#   - qty / tickets_sold        : integer quantity sold for that day
# Optional columns:
#   - city                      : 'Calgary' / 'Edmonton' (or any label)
#   - performance_id            : identifier for a performance/run; if missing, unique performance_date is used
#   - capacity / seats          : capacity (per perf or total run). If absent, per-show normalization is used.
#   - revenue                   : enables revenue curves
#   - channel                   : optional segmentation
#
# -------------------------------------------------

from __future__ import annotations
import io
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# ----------------------------
# Streamlit page configuration
# ----------------------------
st.set_page_config(
    page_title="Nutcracker Trajectory Tracker",
    page_icon="🩰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🩰 Nutcracker Sales Trajectory Tracker")
st.caption(
    "Reads CSVs from the repo's `data/` folder. Compare trajectories, normalize for show count/capacity, and project final totals."
)

# ----------------------------
# Config: data folder & discovery
# ----------------------------
DATA_DIR = Path.cwd() / "data"
HIST_FIXED = DATA_DIR / "historical.csv"
THIS_FIXED = DATA_DIR / "this_year.csv"
HIST_PATTERN = "historical_*.csv"
THIS_PATTERN = "this_year_*.csv"

# ----------------------------
# Helpers (closing-date; no opening-date dependency)
# ----------------------------
DATE_COL_CANDIDATES = ["order_date", "sale_date", "sales_date", "transaction_date", "date"]
QTY_COL_CANDIDATES  = ["qty", "tickets_sold", "units", "tickets", "tickets sold"]
SEASON_COL_CANDIDATES = ["season", "year"]
CITY_COL_CANDIDATES   = ["city", "market"]
PERF_ID_CANDIDATES    = ["performance_id", "perf_id", "show_id"]  # optional
CAPACITY_COL_CANDIDATES = ["capacity", "seats", "house_capacity", "total_capacity"]  # optional
REVENUE_COL_CANDIDATES  = ["revenue", "gross", "amount"]  # optional
SOURCE_COL_CANDIDATES   = ["source", "platform", "channel", "system"]  # optional (Ticketmaster/Archtics)

def _find_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for c in candidates:
        if c in low:
            return low[c]
    return None


def _coerce_dates(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_datetime(df[col], errors="coerce").dt.tz_localize(None)

@st.cache_data(show_spinner=False)
def load_and_standardize_from_path(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}

    season_col = _find_col(list(lower.keys()), SEASON_COL_CANDIDATES)
    date_col   = _find_col(list(lower.keys()), DATE_COL_CANDIDATES)
    qty_col    = _find_col(list(lower.keys()), QTY_COL_CANDIDATES)
    city_col   = _find_col(list(lower.keys()), CITY_COL_CANDIDATES)
    perf_id_col= _find_col(list(lower.keys()), PERF_ID_CANDIDATES)
    cap_col    = _find_col(list(lower.keys()), CAPACITY_COL_CANDIDATES)
    rev_col    = _find_col(list(lower.keys()), REVENUE_COL_CANDIDATES)
    src_col    = _find_col(list(lower.keys()), SOURCE_COL_CANDIDATES)

    # Required: season, sale date, qty (NO performance_date required)
    missing = []
    if season_col is None: missing.append("season/year")
    if date_col   is None: missing.append("order_date/sale_date")
    if qty_col    is None: missing.append("qty/tickets_sold")
    if missing:
        raise ValueError(f"{path.name}: Missing required column(s): " + ", ".join(missing))

    # Rename core + optional
    ren = {season_col: "season", date_col: "sale_date", qty_col: "qty"}
    if city_col:     ren[city_col]     = "city"
    if rev_col:      ren[rev_col]      = "revenue"
    if cap_col:      ren[cap_col]      = "capacity"
    if src_col:      ren[src_col]      = "source"
    if perf_id_col:  ren[perf_id_col]  = "performance_id"
    df = df.rename(columns=ren)

    # Types
    df["sale_date"] = _coerce_dates(df, "sale_date")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0).astype(int)
    if "revenue" in df.columns:
        df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0.0)
    if "capacity" in df.columns:
        df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce").astype("Float64")

    # Normalizations
    df["season"] = df["season"].astype(str).str.strip()
    if "city" not in df.columns:
        df["city"] = "Combined"
    if "source" in df.columns:
        df["source"] = df["source"].fillna("Unknown").replace({"": "Unknown"})

    return df


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["season", "city", "sale_date"], dropna=False, as_index=False).agg(
        qty=("qty", "sum"),
        revenue=("revenue", "sum") if "revenue" in df.columns else ("qty", "sum"),
    )
    grp = grp.sort_values(["season", "city", "sale_date"]).reset_index(drop=True)
    return grp


def _season_to_closing_year(s: str) -> Optional[int]:
    """
    Robust-ish parser:
    - '2025' -> 2025
    - '2024-25' or '2024/25' -> 2025
    - '2022–2023' (en dash) -> 2023
    Returns None if we can't parse.
    """
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    import re
    # 4-digit only
    m = re.search(r"(20\d{2})$", s)
    if m:
        return int(m.group(1))
    # Patterns like 2024-25, 2024/25, 2024–25
    m = re.search(r"(20\d{2})\s*[-/–]\s*(\d{2,4})", s)
    if m:
        start = int(m.group(1))
        tail = m.group(2)
        if len(tail) == 2:
            # 2024-25 -> 2025
            return int(str(start)[:2] + tail)
        elif len(tail) == 4:
            return int(tail)
    # Fallback: first 4-digit year in string
    m = re.search(r"(20\d{2})", s)
    if m:
        return int(m.group(1))
    return None


def compute_calendar_refs(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Derive closing_year from season string
    meta = (
        df[["season", "city"]].drop_duplicates()
          .assign(closing_year=lambda x: x["season"].apply(_season_to_closing_year))
    )
    if meta["closing_year"].isna().any():
        bad = meta[meta["closing_year"].isna()]["season"].unique().tolist()
        raise ValueError(f"Could not parse closing year from season labels: {bad}")

    meta["closing_date"] = pd.to_datetime(
        dict(year=meta["closing_year"].astype(int), month=12, day=24)
    )

    # Bring optional capacity/num_shows if present
    out_meta = meta.copy()
    if "capacity" in df.columns:
        cap = (df.groupby(["season", "city"], dropna=False)["capacity"]
                 .max().rename("total_capacity").reset_index())
        out_meta = out_meta.merge(cap, on=["season", "city"], how="left")
    if "num_shows" in df.columns:
        ns = (df.groupby(["season", "city"], dropna=False)["num_shows"]
                .max().reset_index())
        out_meta = out_meta.merge(ns, on=["season", "city"], how="left")

    # Aggregate daily sales
    daily = aggregate_daily(df).merge(out_meta, on=["season", "city"], how="left")

    # Days to closing: negative before Dec 24, zero on Dec 24
    daily["days_to_close"] = (daily["sale_date"].dt.normalize() - daily["closing_date"]).dt.days

    # Cumulative
    daily = daily.sort_values(["season", "city", "sale_date"]).reset_index(drop=True)
    daily["cum_qty"] = daily.groupby(["season", "city"], as_index=False)["qty"].cumsum()
    if "revenue" in daily.columns:
        daily["cum_rev"] = daily.groupby(["season", "city"], as_index=False)["revenue"].cumsum()
    else:
        daily["cum_rev"] = np.nan

    finals = daily.groupby(["season", "city"], dropna=False).agg(
        final_qty=("qty", "sum"),
        final_rev=("revenue", "sum") if "revenue" in daily.columns else ("qty", "sum"),
    ).reset_index()

    out = daily.merge(finals, on=["season", "city"], how="left")

    # Per-show normalization only if num_shows is known
    if "num_shows" in out_meta.columns:
        out = out.merge(out_meta[["season", "city", "num_shows"]], on=["season", "city"], how="left")
        out["per_show_cum_qty"] = out["cum_qty"] / out["num_shows"].replace(0, np.nan)
    else:
        out["per_show_cum_qty"] = np.nan

    out["share_of_final_qty"] = out["cum_qty"] / out["final_qty"].replace(0, np.nan)

    if "total_capacity" in out_meta.columns:
        out = out.merge(out_meta[["season", "city", "total_capacity"]], on=["season", "city"], how="left")
        out["share_of_capacity"] = out["cum_qty"] / out["total_capacity"].replace(0, np.nan)
    else:
        out["total_capacity"] = np.nan
        out["share_of_capacity"] = np.nan

    return out, out_meta


def build_reference_curve(daily: pd.DataFrame, seasons_ref: List[str]) -> pd.DataFrame:
    ref = daily[daily["season"].isin(seasons_ref)].copy()
    needed = ["season", "city", "days_to_close", "share_of_final_qty", "per_show_cum_qty"]
    missing = [c for c in needed if c not in ref.columns]
    if missing:
        raise ValueError(f"Missing columns for reference curve: {missing}")

    agg = (
        ref.groupby(["city", "days_to_close"], dropna=False)
           .agg(mean_share=("share_of_final_qty", "mean"),
                min_share=("share_of_final_qty", "min"),
                max_share=("share_of_final_qty", "max"),
                mean_per_show=("per_show_cum_qty", "mean"))
           .reset_index()
    )
    return agg


def project_this_year(
    daily: pd.DataFrame,
    this_season: str,
    ref_curve: pd.DataFrame,
    open_meta: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cur = daily[daily["season"] == this_season].copy()
    cur = cur.merge(open_meta, on=["season", "city"], how="left")

    cur = cur.merge(ref_curve, on=["city", "days_to_close"], how="left")

    proj_rows = []
    for city, g in cur.groupby("city"):
        g = g.sort_values("sale_date")
        cum_qty_today = g["cum_qty"].iloc[-1]
        cap_total = g["total_capacity"].iloc[-1] if "total_capacity" in g.columns else np.nan
        ref_today_share = g["mean_share"].dropna().iloc[-1] if g["mean_share"].notna().any() else np.nan

        proj_final_qty_shape = np.nan
        if pd.notna(ref_today_share) and ref_today_share > 0:
            proj_final_qty_shape = cum_qty_today / ref_today_share

        proj_pct_capacity = np.nan
        if pd.notna(cap_total) and cap_total > 0 and pd.notna(proj_final_qty_shape):
            proj_pct_capacity = proj_final_qty_shape / cap_total

        g["proj_cum_qty"] = g["mean_share"] * proj_final_qty_shape if pd.notna(proj_final_qty_shape) else np.nan
        g["proj_min_cum_qty"] = g["min_share"] * proj_final_qty_shape if pd.notna(proj_final_qty_shape) else np.nan
        g["proj_max_cum_qty"] = g["max_share"] * proj_final_qty_shape if pd.notna(proj_final_qty_shape) else np.nan

        g["city"] = city
        proj_rows.append(g)

    proj = pd.concat(proj_rows, ignore_index=True) if proj_rows else cur

    summaries = []
    for city, g in proj.groupby("city"):
        row = dict(
            season=this_season,
            city=city,
            current_cum_qty=g["cum_qty"].iloc[-1] if len(g) else np.nan,
            projected_final_qty=(g["proj_cum_qty"].iloc[-1] if g["proj_cum_qty"].notna().any() else np.nan),
            projected_pct_capacity=(g["proj_cum_qty"].iloc[-1] / g["total_capacity"].iloc[-1]
                                    if ("total_capacity" in g.columns and g["total_capacity"].iloc[-1] > 0 and g["proj_cum_qty"].notna().any()) else np.nan),
            num_shows=g["num_shows"].iloc[-1] if "num_shows" in g.columns else np.nan,
            total_capacity=g["total_capacity"].iloc[-1] if "total_capacity" in g.columns else np.nan,
        )
        summaries.append(row)
    summary_df = pd.DataFrame(summaries)
    return proj, summary_df

# ----------------------------
# Data discovery (no uploads)
# ----------------------------
with st.sidebar:
    st.header("1) Data source: `data/` folder")
    st.caption(str(DATA_DIR))

    if not DATA_DIR.exists():
        st.error("`data/` folder not found at repo root. Create it and add CSVs.")
        st.stop()

    # Gather candidates
    hist_fixed = [HIST_FIXED] if HIST_FIXED.exists() else []
    hist_glob  = sorted(DATA_DIR.glob(HIST_PATTERN))
    this_fixed = [THIS_FIXED] if THIS_FIXED.exists() else []
    this_glob  = sorted(DATA_DIR.glob(THIS_PATTERN))

    if not (hist_fixed or hist_glob):
        st.error("No historical CSVs found. Add `historical.csv` or `historical_*.csv` in `data/`.")
        st.stop()
    if not (this_fixed or this_glob):
        st.error("No this-season CSVs found. Add `this_year.csv` or `this_year_*.csv` in `data/`.")
        st.stop()

    # Toggle: merge all historical_* or pick one
    merge_all_hist = st.checkbox("Merge all `historical_*.csv` files", value=True)

    if merge_all_hist and hist_glob:
        st.write(f"Found {len(hist_glob)} historical files to merge.")
        sel_hist_list = [p for p in hist_glob]  # use all matches
    else:
        # fall back to single-file selection (fixed file + any matches)
        hist_candidates = hist_fixed + hist_glob
        names = [p.name for p in hist_candidates]
        sel_name = st.selectbox("Historical CSV (single file)", options=names, index=len(names)-1)
        sel_hist_list = [hist_candidates[names.index(sel_name)]]

    # Current year: pick one file (usually just one)
    this_candidates = this_fixed + this_glob
    this_names = [p.name for p in this_candidates]
    sel_this_name = st.selectbox("This-year CSV", options=this_names, index=len(this_names)-1)
    sel_this_path = this_candidates[this_names.index(sel_this_name)]

    city_mode = st.selectbox("City view", ["Combined", "By City"], index=0)
    show_revenue = st.checkbox("Include revenue curves when available", value=True)

# Load CSVs from disk
try:
    # merge all selected historical files
    hist_parts = [load_and_standardize_from_path(p) for p in sel_hist_list]
    hist_df = pd.concat(hist_parts, ignore_index=True)
    this_df = load_and_standardize_from_path(sel_this_path)
except Exception as e:
    st.error(f"Failed to read data files: {e}")
    st.stop()

# Determine the current season label (this_year must contain exactly one season)
this_season_values = sorted(this_df["season"].unique().tolist())
if len(this_season_values) != 1:
    st.warning("Your this-season file should contain exactly one season value. Using the first found.")
this_season = this_season_values[0]

# Optional city collapsing
all_df = pd.concat([hist_df, this_df], ignore_index=True)
if city_mode == "Combined":
    all_df["city"] = "Combined"

# Core calendar + cumulative metrics
daily, open_meta = compute_calendar_refs(all_df)

# Sidebar: choose reference seasons (exclude current)
with st.sidebar:
    seasons_all = sorted(daily["season"].unique().tolist())
    ref_default = [s for s in seasons_all if s != this_season]
    seasons_ref = st.multiselect("Reference seasons (exclude this year)", options=[s for s in seasons_all if s != this_season], default=ref_default)
    if not seasons_ref:
        st.warning("Select at least one reference season for projections.")

# Separate frames for ref vs current
ref_daily = daily[daily["season"].isin(seasons_ref)].copy()
this_daily = daily[daily["season"] == this_season].copy()

# Build reference curve and project
try:
    ref_curve = build_reference_curve(daily, seasons_ref) if seasons_ref else pd.DataFrame()
except Exception as e:
    st.error(f"Could not build reference curve: {e}")
    st.stop()

proj_df, summary_df = project_this_year(daily, this_season, ref_curve, open_meta)

# ----------------------------
# Main layout
# ----------------------------
left, right = st.columns([2, 1])

with left:
    st.subheader("Reference share band (historical)")
    if not ref_curve.empty:
        band = alt.Chart(ref_curve).mark_area(opacity=0.2).encode(
            x=alt.X("days_to_close:Q", title="Days to closing (Dec 24) — negative before closing"),
            y="min_share:Q",
            y2="max_share:Q",
            tooltip=["min_share", "max_share"],
        )
        mean_line = alt.Chart(ref_curve).mark_line(strokeDash=[4,2]).encode(
            x="days_to_close:Q",
            y=alt.Y("mean_share:Q", title="Cumulative share of final"),
            tooltip=["mean_share"],
        )
        st.altair_chart((band + mean_line).properties(height=240), use_container_width=True)

    st.subheader("Actual vs projected cumulative tickets")
    abs_join = proj_df.dropna(subset=["days_to_close"]).copy()
    act = alt.Chart(abs_join).mark_line().encode(
        x=alt.X("days_to_close:Q", title="Days to closing (Dec 24)"),
        y=alt.Y("cum_qty:Q", title="Cumulative tickets"),
        color=alt.Color("city:N", title="City"),
        tooltip=["city", "sale_date", "cum_qty", "num_shows"],
    )
    proj = alt.Chart(abs_join).mark_line(strokeDash=[4,2]).encode(
        x="days_to_close:Q",
        y="proj_cum_qty:Q",
        color=alt.Color("city:N", title="City"),
        tooltip=["city", "sale_date", "proj_cum_qty"],
    )
    st.altair_chart((act + proj).properties(height=300), use_container_width=True)

    st.subheader("Normalized per-show cumulative (historical vs this year)")
    hist_norm = ref_daily.dropna(subset=["days_to_close"]).copy()
    lines_hist = alt.Chart(hist_norm).mark_line(opacity=0.35).encode(
        x=alt.X("days_to_close:Q", title="Days to closing (Dec 24)"),
        y=alt.Y("per_show_cum_qty:Q", title="Cumulative tickets per show"),
        color=alt.Color("season:N", title="Season"),
        tooltip=["season", "city", "per_show_cum_qty"],
    )

    this_norm = this_daily.dropna(subset=["days_to_close"]).copy()
    line_this = alt.Chart(this_norm).mark_line(size=3).encode(
        x="days_to_close:Q",
        y="per_show_cum_qty:Q",
        color=alt.Color("city:N", title="City"),
        tooltip=["city", "per_show_cum_qty"],
    )
    st.altair_chart((lines_hist + line_this).properties(height=260), use_container_width=True)

    if show_revenue and ("cum_rev" in daily.columns) and daily["cum_rev"].notna().any():
        st.subheader("Revenue trajectory (if provided)")
        rev = daily.dropna(subset=["days_to_close"]).copy()
        rev_chart = alt.Chart(rev).mark_line().encode(
            x=alt.X("days_to_close:Q", title="Days to closing (Dec 24)"),
            y=alt.Y("cum_rev:Q", title="Cumulative revenue"),
            color=alt.Color("season:N", title="Season"),
            tooltip=["season", "city", "cum_rev"],
        )
        st.altair_chart(rev_chart.properties(height=260), use_container_width=True)

with right:
    st.subheader("Summary & projection")
    st.dataframe(
        summary_df.assign(
            projected_final_qty=summary_df["projected_final_qty"].round(0),
            projected_pct_capacity=(summary_df["projected_pct_capacity"] * 100).round(1),
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        label="Download projection by day (CSV)",
        data=proj_df.to_csv(index=False).encode("utf-8"),
        file_name=f"nutcracker_projection_by_day_{this_season}.csv",
        mime="text/csv",
    )

    st.download_button(
        label="Download summary (CSV)",
        data=summary_df.to_csv(index=False).encode("utf-8"),
        file_name=f"nutcracker_projection_summary_{this_season}.csv",
        mime="text/csv",
    )

# ----------------------------
# Footer notes
# ----------------------------
st.markdown(
    """
**How projections work**  
We compute an average **cumulative share curve** across your selected reference seasons (by city). 
At today's *days to closing*, we read the mean share and divide current cumulative tickets by that share
(e.g., if reference says 62% sold at −14 days to closing and you’re at 15,500, the shape-projected final is ~25,000).  
Where available, we also show a band using the min/max shares from reference seasons.

**Normalizing for more shows**  
If capacity per performance is provided, we compute capacity-based metrics and show **% of capacity**. 
If not, we normalize **per show** so seasons with more shows are comparable to seasons with fewer shows.

**`data/` usage**  
- Put your files in `data/` at repo root.
- Use fixed names (`historical.csv`, `this_year.csv`) or date-stamped patterns (`historical_2022-2024.csv`, `this_year_2025_2025-11-02.csv`).
- The sidebar lets you pick which files to use when multiple matches exist.
"""
)
