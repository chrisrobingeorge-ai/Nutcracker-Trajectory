# Nutcracker Sales Trajectory Tracker (Streamlit)
# -------------------------------------------------
# Purpose
# - Compare this year's Nutcracker sales trajectory to prior years
# - Normalize for different numbers of performances (and capacity if available)
# - Project final sales using historical cumulative share curves
# - Handle Calgary vs Edmonton (or combined) views
# - Reads CSVs from a local `data/` folder in the repo (no file uploads)
# - Timeline anchored to closing day (Dec 24) each year
# -------------------------------------------------

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple

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
DATE_COL_CANDIDATES = ["order_date","sale_date","sales_date","transaction_date","date","sales date"]
QTY_COL_CANDIDATES = ["qty","tickets_sold","tickets","units","tickets sold"]
SEASON_COL_CANDIDATES = ["season","year"]
CITY_COL_CANDIDATES = ["city","market"]
CAPACITY_COL_CANDIDATES = ["capacity","seats","house_capacity","total_capacity"]
REVENUE_COL_CANDIDATES = ["revenue","gross","amount"]
SOURCE_COL_CANDIDATES  = ["source","platform","channel","system"]

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
    cols = list(df.columns)

    season_col = _find_col(cols, SEASON_COL_CANDIDATES)
    date_col   = _find_col(cols, DATE_COL_CANDIDATES)
    qty_col    = _find_col(cols, QTY_COL_CANDIDATES)
    city_col   = _find_col(cols, CITY_COL_CANDIDATES)
    cap_col    = _find_col(cols, CAPACITY_COL_CANDIDATES)
    rev_col    = _find_col(cols, REVENUE_COL_CANDIDATES)
    src_col    = _find_col(cols, SOURCE_COL_CANDIDATES)

    missing = []
    if season_col is None: missing.append("season/year")
    if date_col   is None: missing.append("sale_date (Date / order_date / sales_date)")
    if qty_col    is None: missing.append("qty/tickets_sold")
    if missing:
        raise ValueError(f"{path.name}: Missing required column(s): {', '.join(missing)}. Found columns: {cols}")

    ren = {season_col: "season", date_col: "sale_date", qty_col: "qty"}
    if city_col: ren[city_col] = "city"
    if cap_col:  ren[cap_col]  = "capacity"
    if rev_col:  ren[rev_col]  = "revenue"
    if src_col:  ren[src_col]  = "source"
    df = df.rename(columns=ren)

    df["sale_date"] = _coerce_dates(df, "sale_date")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0).astype(int)
    if "revenue" in df.columns:
        df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0.0)
    if "capacity" in df.columns:
        df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce").astype("Float64")

    df["season"] = df["season"].astype(str).str.strip()
    if "city" not in df.columns:
        df["city"] = "Combined"
    if "source" in df.columns:
        df["source"] = df["source"].fillna("Unknown").replace({"": "Unknown"})

    if df["sale_date"].isna().any():
        bad = df.loc[df["sale_date"].isna()].head(5)
        raise ValueError(f"{path.name}: Some sale_date values could not be parsed.\n{bad.to_string(index=False)}")

    return df

def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["season", "city", "sale_date"], dropna=False, as_index=False).agg(
        qty=("qty", "sum"),
        revenue=("revenue", "sum") if "revenue" in df.columns else ("qty", "sum"),
    )
    grp = grp.sort_values(["season", "city", "sale_date"]).reset_index(drop=True)
    return grp

def _season_to_closing_year(s: str) -> Optional[int]:
    # '2025' -> 2025; '2024-25'/'2024/25'/'2024–25' -> 2025; '2022–2023' -> 2023
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    import re
    m = re.search(r"(20\d{2})$", s)
    if m:
        return int(m.group(1))
    m = re.search(r"(20\d{2})\s*[-/–]\s*(\d{2,4})", s)
    if m:
        start = int(m.group(1))
        tail = m.group(2)
        if len(tail) == 2:
            return int(str(start)[:2] + tail)
        elif len(tail) == 4:
            return int(tail)
    m = re.search(r"(20\d{2})", s)
    if m:
        return int(m.group(1))
    return None

def compute_calendar_refs(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Build meta per (season,city) with Dec 24 closing
    meta = (
        df[["season", "city"]].drop_duplicates()
          .assign(closing_year=lambda x: x["season"].apply(_season_to_closing_year))
    )
    if meta["closing_year"].isna().any():
        bad = meta[meta["closing_year"].isna()]["season"].unique().tolist()
        raise ValueError(f"Could not parse closing year from season labels: {bad}")

    meta["closing_date"] = pd.to_datetime(dict(year=meta["closing_year"].astype(int), month=12, day=24))

    out_meta = meta.copy()
    # total_capacity
    if "capacity" in df.columns:
        cap = (df.groupby(["season", "city"], dropna=False)["capacity"]
                 .max().rename("total_capacity").reset_index())
        out_meta = out_meta.merge(cap, on=["season", "city"], how="left")
    else:
        out_meta["total_capacity"] = np.nan

    # num_shows (optional if provided somewhere upstream)
    if "num_shows" in df.columns:
        ns = (df.groupby(["season", "city"], dropna=False)["num_shows"]
                .max().reset_index())
        out_meta = out_meta.merge(ns, on=["season", "city"], how="left")
    else:
        out_meta["num_shows"] = np.nan

    daily = aggregate_daily(df).merge(out_meta, on=["season", "city"], how="left")

    # days_to_close: negative before Dec 24, 0 on Dec 24
    daily["days_to_close"] = (daily["sale_date"].dt.normalize() - daily["closing_date"]).dt.days

    # cumulative
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

    out["per_show_cum_qty"] = out["cum_qty"] / out["num_shows"].replace(0, np.nan)
    out["share_of_final_qty"] = out["cum_qty"] / out["final_qty"].replace(0, np.nan)
    out["share_of_capacity"] = out["cum_qty"] / out["total_capacity"].replace(0, np.nan)

    return out, out_meta

def build_reference_curve(daily: pd.DataFrame, seasons_ref: List[str]) -> pd.DataFrame:
    """
    Build a reference 'cumulative share of final' curve by days_to_close,
    averaged across the selected reference seasons, per city.
    """
    if not seasons_ref:
        return pd.DataFrame(columns=["city","days_to_close","mean_share","min_share","max_share","mean_per_show"])

    ref = daily[daily["season"].isin(seasons_ref)].copy()
    needed = ["season", "city", "days_to_close", "share_of_final_qty", "per_show_cum_qty"]
    missing = [c for c in needed if c not in ref.columns]
    if missing:
        raise ValueError(f"Missing columns for reference curve: {missing}")

    if ref.empty:
        return pd.DataFrame(columns=["city","days_to_close","mean_share","min_share","max_share","mean_per_show"])

    agg = (
        ref.groupby(["city", "days_to_close"], dropna=False)
           .agg(
               mean_share=("share_of_final_qty", "mean"),
               min_share=("share_of_final_qty", "min"),
               max_share=("share_of_final_qty", "max"),
               mean_per_show=("per_show_cum_qty", "mean"),
           )
           .reset_index()
    )
    return agg

def project_this_year(
    daily: pd.DataFrame,
    this_season: str,
    ref_curve: pd.DataFrame,
    run_meta: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cur = daily[daily["season"] == this_season].copy()
    cur = cur.merge(run_meta[["season","city","num_shows","total_capacity"]], on=["season","city"], how="left")
    cur = cur.merge(ref_curve, on=["city", "days_to_close"], how="left")

    proj_rows = []
    for city, g in cur.groupby("city"):
        g = g.sort_values("sale_date")
        cum_qty_today = g["cum_qty"].iloc[-1] if len(g) else np.nan
        cap_total = g["total_capacity"].iloc[-1] if "total_capacity" in g.columns else np.nan
        ref_today_share = g["mean_share"].dropna().iloc[-1] if g["mean_share"].notna().any() else np.nan

        proj_final_qty_shape = np.nan
        if pd.notna(ref_today_share) and ref_today_share > 0:
            proj_final_qty_shape = cum_qty_today / ref_today_share

        g["proj_cum_qty"] = g["mean_share"] * proj_final_qty_shape if pd.notna(proj_final_qty_shape) else np.nan
        g["proj_min_cum_qty"] = g["min_share"] * proj_final_qty_shape if pd.notna(proj_final_qty_shape) else np.nan
        g["proj_max_cum_qty"] = g["max_share"] * proj_final_qty_shape if pd.notna(proj_final_qty_shape) else np.nan

        proj_rows.append(g)

    proj = pd.concat(proj_rows, ignore_index=True) if proj_rows else cur

    summaries = []
    for city, g in proj.groupby("city"):
        cap_total = g["total_capacity"].iloc[-1] if "total_capacity" in g.columns else np.nan
        proj_final = g["proj_cum_qty"].dropna().iloc[-1] if g["proj_cum_qty"].notna().any() else np.nan
        pct_cap = (proj_final / cap_total) if (pd.notna(proj_final) and pd.notna(cap_total) and cap_total > 0) else np.nan
        row = dict(
            season=this_season,
            city=city,
            current_cum_qty=g["cum_qty"].iloc[-1] if len(g) else np.nan,
            projected_final_qty=proj_final,
            projected_pct_capacity=pct_cap,
            num_shows=g["num_shows"].iloc[-1] if "num_shows" in g.columns else np.nan,
            total_capacity=cap_total,
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

    merge_all_hist = st.checkbox("Merge all `historical_*.csv` files", value=True)

    if merge_all_hist and hist_glob:
        st.write(f"Found {len(hist_glob)} historical files to merge.")
        sel_hist_list = [p for p in hist_glob]
    else:
        hist_candidates = hist_fixed + hist_glob
        names = [p.name for p in hist_candidates]
        sel_name = st.selectbox("Historical CSV (single file)", options=names, index=len(names)-1)
        sel_hist_list = [hist_candidates[names.index(sel_name)]]

    this_candidates = this_fixed + this_glob
    this_names = [p.name for p in this_candidates]
    sel_this_name = st.selectbox("This-year CSV", options=this_names, index=len(this_names)-1)
    sel_this_path = this_candidates[this_names.index(sel_this_name)]

    city_mode = st.selectbox("City view", ["Combined", "By City"], index=0)
    show_revenue = st.checkbox("Include revenue curves when available", value=True)

with st.sidebar:
    st.header("2) Display")
    window_days = st.slider(
        "Window: show last N days before closing",
        min_value=90, max_value=700, value=365, step=15
    )
    min_share_to_project = st.slider(
        "Project only after ref share ≥",
        min_value=0.05, max_value=0.90, value=0.20, step=0.05
    )

# Load CSVs from disk
try:
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
daily, run_meta = compute_calendar_refs(all_df)

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
    ref_curve = build_reference_curve(daily, seasons_ref)
except Exception as e:
    st.error(f"Could not build reference curve: {e}")
    st.stop()

proj_df, summary_df = project_this_year(daily, this_season, ref_curve, run_meta)

# Trim plotting window
plot_ref  = ref_curve[ref_curve["days_to_close"] >= -window_days].copy()
plot_proj = proj_df[proj_df["days_to_close"] >= -window_days].copy()
plot_hist = ref_daily[ref_daily["days_to_close"] >= -window_days].copy()
plot_this = this_daily[this_daily["days_to_close"] >= -window_days].copy()

# Gate projections until curve is meaningful
if "mean_share" in plot_proj.columns:
    too_early = plot_proj["mean_share"].notna() & (plot_proj["mean_share"] < min_share_to_project)
    plot_proj.loc[too_early, ["proj_cum_qty", "proj_min_cum_qty", "proj_max_cum_qty"]] = np.nan

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
            tooltip=["city","days_to_close","min_share","max_share"],
        )
        mean_line = alt.Chart(ref_curve).mark_line(strokeDash=[4,2]).encode(
            x="days_to_close:Q",
            y=alt.Y("mean_share:Q", title="Cumulative share of final"),
            color=alt.value("black"),
            tooltip=["city","days_to_close","mean_share"],
        )
        st.altair_chart((band + mean_line).properties(height=240), use_container_width=True)
    else:
        st.info("No reference curve available for the selected seasons.")

    st.subheader("Actual vs projected cumulative tickets")
    abs_join = proj_df.dropna(subset=["days_to_close", "cum_qty"]).copy()
    if not abs_join.empty:
        act = alt.Chart(abs_join).mark_line().encode(
            x=alt.X("days_to_close:Q", title="Days to closing (Dec 24)"),
            y=alt.Y("cum_qty:Q", title="Cumulative tickets"),
            color=alt.Color("city:N", title="City"),
            tooltip=["city","season","sale_date","cum_qty"],
        )
        proj = alt.Chart(abs_join.dropna(subset=["proj_cum_qty"])).mark_line(strokeDash=[4,2]).encode(
            x="days_to_close:Q",
            y="proj_cum_qty:Q",
            color=alt.Color("city:N", title="City"),
            tooltip=["city","season","sale_date","proj_cum_qty"],
        )
        st.altair_chart((act + proj).properties(height=300), use_container_width=True)
    else:
        st.info("No data available yet for actuals/projections.")

    st.subheader("Normalized per-show cumulative (historical vs this year)")
    hist_norm = ref_daily.dropna(subset=["days_to_close","per_show_cum_qty"])
    this_norm = this_daily.dropna(subset=["days_to_close","per_show_cum_qty"])
    if not hist_norm.empty or not this_norm.empty:
        lines_hist = alt.Chart(hist_norm).mark_line(opacity=0.35).encode(
            x=alt.X("days_to_close:Q", title="Days to closing (Dec 24)"),
            y=alt.Y("per_show_cum_qty:Q", title="Cumulative tickets per show"),
            color=alt.Color("season:N", title="Season"),
            tooltip=["season","city","days_to_close","per_show_cum_qty"],
        )
        line_this = alt.Chart(this_norm).mark_line(size=3).encode(
            x="days_to_close:Q",
            y="per_show_cum_qty:Q",
            color=alt.Color("city:N", title="City"),
            tooltip=["city","season","days_to_close","per_show_cum_qty"],
        )
        st.altair_chart((lines_hist + line_this).properties(height=260), use_container_width=True)
    else:
        st.info("Per-show normalization requires `num_shows` to be known. Otherwise this chart will be empty.")

    if show_revenue and ("cum_rev" in daily.columns) and daily["cum_rev"].notna().any():
        st.subheader("Revenue trajectory (if provided)")
        rev = daily.dropna(subset=["days_to_close","cum_rev"])
        if not rev.empty:
            rev_chart = alt.Chart(rev).mark_line().encode(
                x=alt.X("days_to_close:Q", title="Days to closing (Dec 24)"),
                y=alt.Y("cum_rev:Q", title="Cumulative revenue"),
                color=alt.Color("season:N", title="Season"),
                tooltip=["season","city","sale_date","cum_rev"],
            )
            st.altair_chart(rev_chart.properties(height=260), use_container_width=True)
        else:
            st.info("No revenue values to plot.")
with right:
    st.subheader("Summary & projection")
    if not summary_df.empty:
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
    else:
        st.info("No summary available yet.")

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
