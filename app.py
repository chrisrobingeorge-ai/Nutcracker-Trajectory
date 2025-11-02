# Nutcracker Sales Trajectory Tracker (Streamlit)
# -------------------------------------------------
# Purpose
# - Compare this year's Nutcracker sales trajectory to prior years
# - Normalize for different numbers of performances (and capacity if available)
# - Project final sales using historical cumulative share curves
# - Handle Calgary vs Edmonton (or combined) views
# - Ready for Streamlit Cloud (single-file app)
#
# Data expectations (CSV)
# -----------------------
# Historical (multi-year) CSV — required columns (case-insensitive, flexible names allowed):
#   - season / year             : e.g., 2019, 2022-23, etc. (string or int)
#   - order_date / sale_date    : date of ticket sale (YYYY-MM-DD)
#   - performance_date          : date of performance (YYYY-MM-DD)
#   - qty / tickets_sold        : integer quantity of tickets sold in that transaction/day
# Optional columns:
#   - city                      : 'Calgary' / 'Edmonton' (or any label)
#   - performance_id            : identifier for a performance; if missing, unique performance_date is used
#   - capacity / seats          : capacity of that performance (row-wise or merge-able); if missing, per-show normalization is used instead of per-seat
#   - price / revenue           : if present, app will also compute revenue curves
#   - channel                   : optional segmentation
#
# Current year CSV — same columns, but may be partial to date.
#
# NOTES
# - If your historical file already aggregates daily totals per season, repeat the date in 'order_date' and keep 'qty' as the daily total.
# - If you have per-transaction data, the app will aggregate by day.
# - Capacity normalization (preferred) requires total capacity per season (sum of performance capacities). If not available,
#   the app uses per-show normalization so seasons with more shows are comparable.
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
    "Upload historical multi-year data and this year's to-date. Compare trajectories, normalize for show count/capacity, and project final totals."
)

# ----------------------------
# Helpers
# ----------------------------

DATE_COL_CANDIDATES = ["order_date", "sale_date", "sales_date", "transaction_date"]
PERF_DATE_CANDIDATES = ["performance_date", "perf_date", "show_date"]
QTY_COL_CANDIDATES = ["qty", "tickets_sold", "units", "tickets"]
SEASON_COL_CANDIDATES = ["season", "year"]
CITY_COL_CANDIDATES = ["city", "market"]
PERF_ID_CANDIDATES = ["performance_id", "perf_id", "show_id"]
CAPACITY_COL_CANDIDATES = ["capacity", "seats", "house_capacity"]
REVENUE_COL_CANDIDATES = ["revenue", "gross", "amount"]


def _find_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for c in candidates:
        if c in low:
            return low[c]
    return None


def _coerce_dates(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_datetime(df[col], errors="coerce").dt.tz_localize(None)


@st.cache_data(show_spinner=False)
def load_and_standardize(csv_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(csv_bytes))
    # Standardize column names to lower for detection but keep originals
    colmap = {c: c for c in df.columns}
    lower = {c.lower(): c for c in df.columns}

    season_col = _find_col(list(lower.keys()), SEASON_COL_CANDIDATES)
    date_col = _find_col(list(lower.keys()), DATE_COL_CANDIDATES)
    perf_col = _find_col(list(lower.keys()), PERF_DATE_CANDIDATES)
    qty_col = _find_col(list(lower.keys()), QTY_COL_CANDIDATES)
    city_col = _find_col(list(lower.keys()), CITY_COL_CANDIDATES)
    perf_id_col = _find_col(list(lower.keys()), PERF_ID_CANDIDATES)
    cap_col = _find_col(list(lower.keys()), CAPACITY_COL_CANDIDATES)
    rev_col = _find_col(list(lower.keys()), REVENUE_COL_CANDIDATES)

    required = [season_col, date_col, perf_col, qty_col]
    if any(c is None for c in required):
        missing = [name for cands, name in [
            (SEASON_COL_CANDIDATES, "season/year"),
            (DATE_COL_CANDIDATES, "order_date/sale_date"),
            (PERF_DATE_CANDIDATES, "performance_date"),
            (QTY_COL_CANDIDATES, "qty/tickets_sold"),
        ] if _find_col(list(lower.keys()), cands) is None]
        raise ValueError(
            "Missing required column(s): " + ", ".join(missing)
        )

    # Normalize core columns
    df = df.rename(columns={
        season_col: "season",
        date_col: "sale_date",
        perf_col: "performance_date",
        qty_col: "qty",
    })
    if city_col: df = df.rename(columns={city_col: "city"})
    if perf_id_col: df = df.rename(columns={perf_id_col: "performance_id"})
    if cap_col: df = df.rename(columns={cap_col: "capacity"})
    if rev_col: df = df.rename(columns={rev_col: "revenue"})

    # Types
    df["sale_date"] = _coerce_dates(df, "sale_date")
    df["performance_date"] = _coerce_dates(df, "performance_date")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0).astype(int)
    if "revenue" in df.columns:
        df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0.0)
    if "capacity" in df.columns:
        df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce").fillna(np.nan)

    # Clean season labels as strings
    df["season"] = df["season"].astype(str).str.strip()

    # Fallback performance_id
    if "performance_id" not in df.columns:
        df["performance_id"] = df["season"].astype(str) + "_" + df["performance_date"].dt.strftime("%Y-%m-%d")

    # If no city, mark as Combined
    if "city" not in df.columns:
        df["city"] = "Combined"

    return df


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per season/city/day. Supports transaction-level input."""
    grp = df.groupby(["season", "city", "sale_date"], dropna=False, as_index=False).agg(
        qty=("qty", "sum"),
        revenue=("revenue", "sum") if "revenue" in df.columns else ("qty", "sum"),
    )
    grp = grp.sort_values(["season", "city", "sale_date"]).reset_index(drop=True)
    return grp


def compute_calendar_refs(df: pd.DataFrame) -> pd.DataFrame:
    """Attach opening_date, days_out (negative before opening) and show counts/capacity by season+city."""
    perf_info = (
        df.groupby(["season", "city", "performance_id"], dropna=False)
          .agg(opening_date=("performance_date", "min"),
               perf_capacity=("capacity", "max"))
          .reset_index()
    )
    open_by_sc = perf_info.groupby(["season", "city"], dropna=False).agg(
        opening_date=("opening_date", "min"),
        num_shows=("performance_id", "nunique"),
        total_capacity=("perf_capacity", "sum")
    ).reset_index()

    daily = aggregate_daily(df).merge(open_by_sc, on=["season", "city"], how="left")
    daily["days_out"] = (daily["sale_date"].dt.normalize() - daily["opening_date"]).dt.days

    # Cumulative qty and revenue per season+city
    daily = daily.sort_values(["season", "city", "sale_date"]).reset_index(drop=True)
    daily["cum_qty"] = daily.groupby(["season", "city"], as_index=False)["qty"].cumsum()
    if "revenue" in daily.columns:
        daily["cum_rev"] = daily.groupby(["season", "city"], as_index=False)["revenue"].cumsum()
    else:
        daily["cum_rev"] = np.nan

    # Totals per season+city (finals)
    finals = daily.groupby(["season", "city"], dropna=False).agg(
        final_qty=("qty", "sum"),
        final_rev=("revenue", "sum") if "revenue" in daily.columns else ("qty", "sum"),
    ).reset_index()

    out = daily.merge(finals, on=["season", "city"], how="left")

    # Normalizations
    out["per_show_cum_qty"] = out["cum_qty"] / out["num_shows"].replace(0, np.nan)
    out["share_of_final_qty"] = out["cum_qty"] / out["final_qty"].replace(0, np.nan)

    # Capacity normalization (if total_capacity known)
    if "total_capacity" in out.columns:
        out["share_of_capacity"] = out["cum_qty"] / out["total_capacity"].replace(0, np.nan)
    else:
        out["share_of_capacity"] = np.nan

    return out, open_by_sc


def build_reference_curve(daily: pd.DataFrame, seasons_ref: List[str]) -> pd.DataFrame:
    """Average cumulative shares by days_out over selected reference seasons (per season+city)."""
    ref = daily[daily["season"].isin(seasons_ref)].copy()
    needed = ["season", "city", "days_out", "share_of_final_qty", "per_show_cum_qty"]
    missing = [c for c in needed if c not in ref.columns]
    if missing:
        raise ValueError(f"Missing columns for reference curve: {missing}")

    # Compute mean and range across seasons for each city+days_out
    agg = (
        ref.groupby(["city", "days_out"], dropna=False)
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
    method: str = "capacity_or_shows",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Project final totals for this_season per city using a reference share curve.

    method:
      - "capacity_or_shows": prefer capacity normalization (share_of_capacity) if capacity exists; otherwise scale via per-show.
    Returns (proj_points_by_day, summary_by_city)
    """
    cur = daily[daily["season"] == this_season].copy()
    cur = cur.merge(open_meta, on=["season", "city"], how="left", suffixes=("", ""))

    # Today per city (latest sale_date row)
    latest = cur.sort_values(["city", "sale_date"]).groupby("city").tail(1)

    # Merge curve by days_out
    cur = cur.merge(ref_curve, on=["city", "days_out"], how="left")

    # Compute projections at each day (per city)
    proj_rows = []
    for city, g in cur.groupby("city"):
        g = g.sort_values("sale_date")
        # current values
        cum_qty_today = g["cum_qty"].iloc[-1]
        num_shows = g["num_shows"].iloc[-1]
        cap_total = g["total_capacity"].iloc[-1] if "total_capacity" in g.columns else np.nan

        # reference share at today (use last available mean_share)
        ref_today_share = g["mean_share"].dropna().iloc[-1] if g["mean_share"].notna().any() else np.nan

        # Projection approaches
        proj_final_qty_shape = np.nan
        if pd.notna(ref_today_share) and ref_today_share > 0:
            proj_final_qty_shape = cum_qty_today / ref_today_share

        # Capacity or per-show scaling is already baked into the share curve (share_of_final).
        # If capacity exists, we can also compute projected % of capacity as a check.
        proj_pct_capacity = np.nan
        if pd.notna(cap_total) and cap_total > 0 and pd.notna(proj_final_qty_shape):
            proj_pct_capacity = proj_final_qty_shape / cap_total

        # Build series of *expected* cumulative by day using the ref mean_share
        g["proj_cum_qty"] = g["mean_share"] * proj_final_qty_shape if pd.notna(proj_final_qty_shape) else np.nan
        g["proj_min_cum_qty"] = g["min_share"] * proj_final_qty_shape if pd.notna(proj_final_qty_shape) else np.nan
        g["proj_max_cum_qty"] = g["max_share"] * proj_final_qty_shape if pd.notna(proj_final_qty_shape) else np.nan

        g["city"] = city
        proj_rows.append(g)

    proj = pd.concat(proj_rows, ignore_index=True) if proj_rows else cur

    # Summaries per city
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
# Sidebar — File inputs & Controls
# ----------------------------
with st.sidebar:
    st.header("1) Upload data")
    hist_file = st.file_uploader("Historical (multi-year) CSV", type=["csv"], key="hist")
    this_file = st.file_uploader("This season (to-date) CSV", type=["csv"], key="this")

    st.markdown("---")
    st.header("2) Options")
    default_city_mode = st.selectbox("City view", ["Combined", "By City"], index=0)
    show_revenue = st.checkbox("Include revenue curves when available", value=True)

    st.markdown("---")
    st.header("3) Reference seasons")
    st.caption("Choose which historical seasons define the reference curve (average cumulative shares by day).")


if not hist_file or not this_file:
    st.info("⬅️ Upload both historical and this-season CSVs to begin.")
    st.stop()

# Load
try:
    hist_df = load_and_standardize(hist_file.read())
    this_df = load_and_standardize(this_file.read())
except Exception as e:
    st.error(f"Failed to read files: {e}")
    st.stop()

# Mark the current season label from uploaded 'this' file
this_season_values = sorted(this_df["season"].unique().tolist())
if len(this_season_values) != 1:
    st.warning("Your 'this season' file should contain exactly one season value. Using the first found.")
this_season = this_season_values[0]

# Combine data and compute references
all_df = pd.concat([hist_df, this_df], ignore_index=True)

# Optional city collapsing
if default_city_mode == "Combined":
    all_df["city"] = "Combined"

# Core calendar + cumulative metrics
daily, open_meta = compute_calendar_refs(all_df)

# Pick reference seasons in sidebar once we know season options
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
    st.subheader("Trajectory vs. reference (tickets)")
    # Trajectory chart (cumulative tickets)
    base = alt.Chart(this_daily.dropna(subset=["days_out"]))

    # Historical mean band
    if not ref_curve.empty:
        band = alt.Chart(ref_curve).mark_area(opacity=0.2).encode(
            x=alt.X("days_out:Q", title="Days from opening (negative = before opening)"),
            y="min_share:Q",
            y2="max_share:Q",
            color=alt.value("#888"),
            tooltip=["min_share", "max_share"],
        )
        mean_line = alt.Chart(ref_curve).mark_line(strokeDash=[4,2]).encode(
            x="days_out:Q",
            y=alt.Y("mean_share:Q", title="Cumulative share of final"),
            tooltip=["mean_share"],
        )
        st.altair_chart((band + mean_line).properties(height=260), use_container_width=True)

    # Actual vs projected cumulative tickets (absolute scale)
    abs_join = proj_df.dropna(subset=["days_out"]).copy()

    act = alt.Chart(abs_join).mark_line().encode(
        x=alt.X("days_out:Q", title="Days from opening"),
        y=alt.Y("cum_qty:Q", title="Cumulative tickets"),
        color=alt.Color("city:N", title="City"),
        tooltip=["city", "sale_date", "cum_qty", "num_shows"],
    )
    proj = alt.Chart(abs_join).mark_line(strokeDash=[4,2]).encode(
        x="days_out:Q",
        y="proj_cum_qty:Q",
        color=alt.Color("city:N", title="City"),
        tooltip=["city", "sale_date", "proj_cum_qty"],
    )
    st.altair_chart((act + proj).properties(height=320), use_container_width=True)

    # Normalized per-show cumulative comparison (historical vs this year)
    st.subheader("Normalized per-show cumulative (historical seasons)")
    hist_norm = ref_daily.dropna(subset=["days_out"]).copy()
    hist_norm["label"] = hist_norm["season"]
    lines_hist = alt.Chart(hist_norm).mark_line(opacity=0.35).encode(
        x=alt.X("days_out:Q", title="Days from opening"),
        y=alt.Y("per_show_cum_qty:Q", title="Cumulative tickets per show"),
        color=alt.Color("season:N", title="Season"),
        tooltip=["season", "city", "per_show_cum_qty"],
    )

    this_norm = this_daily.dropna(subset=["days_out"]).copy()
    line_this = alt.Chart(this_norm).mark_line(size=3).encode(
        x="days_out:Q",
        y="per_show_cum_qty:Q",
        color=alt.Color("city:N", title="City"),
        tooltip=["city", "per_show_cum_qty"],
    )
    st.altair_chart((lines_hist + line_this).properties(height=280), use_container_width=True)

    if show_revenue and ("cum_rev" in daily.columns) and daily["cum_rev"].notna().any():
        st.subheader("Revenue trajectory (if provided)")
        rev = daily.dropna(subset=["days_out"]).copy()
        rev_chart = alt.Chart(rev).mark_line().encode(
            x=alt.X("days_out:Q", title="Days from opening"),
            y=alt.Y("cum_rev:Q", title="Cumulative revenue"),
            color=alt.Color("season:N", title="Season"),
            tooltip=["season", "city", "cum_rev"],
        )
        st.altair_chart(rev_chart.properties(height=280), use_container_width=True)

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
At today's *days from opening*, we read that mean share and divide your current cumulative tickets by that share 
(e.g., if reference says 62% sold by this point and you've sold 15,500, the shape-projected final is ~25,000).  
Where available, we also show a band using the min/max shares from reference seasons.

**Normalizing for more shows**  
If capacity per performance is provided, we compute capacity-based metrics and show **% of capacity**. 
If not, we normalize **per show** so seasons with 5 extra shows are comparable to seasons with fewer shows.

**Tips**  
- Ensure this-year CSV contains only one season label.  
- Include `performance_id` and `capacity` to enable the most accurate capacity normalization.  
- Use the city switch in the sidebar to compare Calgary vs Edmonton or a Combined view.
"""
)
