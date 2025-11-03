# Nutcracker Sales Trajectory Tracker (Streamlit)
# -------------------------------------------------
# Purpose
# - Compare this year's Nutcracker sales trajectory to prior years
# - Normalize for different numbers of performances (and capacity if available)
# - Project final sales using historical cumulative share curves
# - Handle Calgary vs Edmonton (or combined) views
# - **Reads CSVs from a local `data/` folder in the repo** (no file uploads)
# - **Supports your 5-column schema**: Date, City, Source, Tickets Sold, Revenue
# - **Optional `data/runs.csv`** to assign season/opening/num_shows when missing
#
# Data expectations (CSV)
# -----------------------
# Place CSVs in `data/` at the repo root. The app auto-discovers them.
# You can use fixed names:
#   - `data/historical.csv`      (multi-year history)
#   - `data/this_year.csv`       (this-season to date; exactly one season)
# or patterns (select in the sidebar):
#   - `data/historical_*.csv`
#   - `data/this_year_*.csv`
#
# Supported schemas (case-insensitive headers):
# A) Rich schema (no runs.csv required):
#   - season / year
#   - order_date / sale_date / date
#   - performance_date
#   - qty / tickets_sold / tickets
#   - city
# Optional: performance_id, capacity / seats / total_capacity, revenue / gross, source / platform / channel
#
# B) Lean 5-column schema (requires `data/runs.csv`):
#   - Date, City, Source, Tickets Sold, Revenue
#   -> Add `data/runs.csv` with columns: season, city, opening_date, num_shows, total_capacity (optional)
#   The app will assign season + opening_date to each sale by matching City and the window
#   [opening_date-365d, Dec 24 (opening year)].
#
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
RUNS_FILE = DATA_DIR / "runs.csv"

# ----------------------------
# Header candidates / helpers
# ----------------------------
DATE_COL_CANDIDATES = ["order_date", "sale_date", "sales_date", "transaction_date", "date"]
PERF_DATE_CANDIDATES = ["performance_date", "perf_date", "show_date"]
QTY_COL_CANDIDATES = ["qty", "tickets_sold", "units", "tickets", "tickets sold"]
SEASON_COL_CANDIDATES = ["season", "year"]
CITY_COL_CANDIDATES = ["city", "market"]
PERF_ID_CANDIDATES = ["performance_id", "perf_id", "show_id"]
CAPACITY_COL_CANDIDATES = ["capacity", "seats", "house_capacity", "total_capacity"]
REVENUE_COL_CANDIDATES = ["revenue", "gross", "amount"]
SOURCE_COL_CANDIDATES  = ["source", "platform", "channel", "system"]


def _find_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for c in candidates:
        if c in low:
            return low[c]
    return None


def _coerce_dates(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_datetime(df[col], errors="coerce").dt.tz_localize(None)


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def _standardize_sales(df: pd.DataFrame, src_name: str) -> pd.DataFrame:
    """Map flexible headers onto canonical names. Does not inject season/perf date if missing."""
    lower = {c.lower(): c for c in df.columns}
    season_col = _find_col(list(lower.keys()), SEASON_COL_CANDIDATES)
    date_col   = _find_col(list(lower.keys()), DATE_COL_CANDIDATES)
    perf_col   = _find_col(list(lower.keys()), PERF_DATE_CANDIDATES)
    qty_col    = _find_col(list(lower.keys()), QTY_COL_CANDIDATES)
    city_col   = _find_col(list(lower.keys()), CITY_COL_CANDIDATES)
    perf_id    = _find_col(list(lower.keys()), PERF_ID_CANDIDATES)
    cap_col    = _find_col(list(lower.keys()), CAPACITY_COL_CANDIDATES)
    rev_col    = _find_col(list(lower.keys()), REVENUE_COL_CANDIDATES)
    src_col    = _find_col(list(lower.keys()), SOURCE_COL_CANDIDATES)

    # require at least date + qty
    missing_base = []
    if date_col is None: missing_base.append("Date")
    if qty_col  is None: missing_base.append("Tickets Sold")
    if missing_base:
        raise ValueError(f"{src_name}: Missing required column(s): {', '.join(missing_base)}")

    # rename what we have
    ren = {date_col: "sale_date", qty_col: "qty"}
    if city_col: ren[city_col] = "city"
    if rev_col:  ren[rev_col]  = "revenue"
    if cap_col:  ren[cap_col]  = "capacity"
    if src_col:  ren[src_col]  = "source"
    if perf_col: ren[perf_col] = "performance_date"
    if season_col: ren[season_col] = "season"
    if perf_id: ren[perf_id] = "performance_id"
    df = df.rename(columns=ren)

    # types
    df["sale_date"] = _coerce_dates(df, "sale_date")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0).astype(int)
    if "revenue" in df.columns:
        df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0.0)
    if "capacity" in df.columns:
        df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce").fillna(np.nan)

    # defaults
    if "city" not in df.columns:
        df["city"] = "Combined"
    if "source" in df.columns:
        df["source"] = df["source"].fillna("Unknown").replace({"": "Unknown"})

    return df


def _load_runs_meta(path: Path) -> pd.DataFrame:
    """runs.csv columns: season, city, opening_date, num_shows, total_capacity (optional)"""
    runs = load_csv(path)
    lc = {c.lower(): c for c in runs.columns}
    req = {
        "season": lc.get("season"),
        "city": lc.get("city"),
        "opening_date": lc.get("opening_date"),
    }
    if any(v is None for v in req.values()):
        raise ValueError("runs.csv must include: season, city, opening_date")
    runs = runs.rename(columns={req["season"]: "season", req["city"]: "city", req["opening_date"]: "opening_date"})
    # optional
    if "num_shows" in lc: runs = runs.rename(columns={lc["num_shows"]: "num_shows"})
    if "total_capacity" in lc: runs = runs.rename(columns={lc["total_capacity"]: "total_capacity"})

    runs["opening_date"] = pd.to_datetime(runs["opening_date"], errors="coerce").dt.tz_localize(None)
    # closing date = Dec 24 of opening year
    runs["closing_date"] = pd.to_datetime(dict(year=runs["opening_date"].dt.year, month=12, day=24))

    # ensure city/type
    runs["city"] = runs["city"].astype(str)
    runs["season"] = runs["season"].astype(str).str.strip()
    return runs


def _assign_season_and_perf_from_runs(sales: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """For lean schema without season/performance_date, assign them using runs.csv by (city, window)."""
    # Cartesian merge by city, then filter by window: sale_date in [opening_date-365, closing_date]
    x = sales.merge(runs, on="city", how="left", suffixes=("", "_run"))
    x["window_start"] = x["opening_date"] - pd.Timedelta(days=365)
    mask = (x["sale_date"] >= x["window_start"]) & (x["sale_date"] <= x["closing_date"])
    x = x[mask].copy()

    # If multiple runs match (rare), pick the nearest opening_date in the future relative to sale_date, else the latest past
    x["delta_open"] = (x["opening_date"] - x["sale_date"]).abs()
    x = x.sort_values(["season", "city", "sale_date", "delta_open"]).drop_duplicates(subset=["city", "sale_date", "qty"], keep="first")

    # Write back season & performance_date
    sales2 = sales.copy()
    sales2 = sales2.merge(x[["city", "sale_date", "season", "opening_date", "num_shows", "total_capacity", "closing_date"]],
                          on=["city", "sale_date"], how="left")
    # If we still lack matches, raise a friendly error
    if sales2["season"].isna().any() or sales2["opening_date"].isna().any():
        missing = sales2[sales2["season"].isna() | sales2["opening_date"].isna()][["city", "sale_date"]].head(10)
        raise ValueError("Some sales rows could not be matched to a run in runs.csv. Example rows:\n" + missing.to_string(index=False))

    # Set canonical names used elsewhere
    sales2 = sales2.rename(columns={"opening_date": "performance_date"})
    return sales2


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["season", "city", "sale_date"], dropna=False, as_index=False).agg(
        qty=("qty", "sum"),
        revenue=("revenue", "sum") if "revenue" in df.columns else ("qty", "sum"),
    )
    grp = grp.sort_values(["season", "city", "sale_date"]).reset_index(drop=True)
    return grp


def compute_calendar_refs(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    perf_info = (
        df.groupby(["season", "city"], dropna=False)
          .agg(opening_date=("performance_date", "min"))
          .reset_index()
    )
    # bring capacities if present per-row (sum per run) and number of shows if provided
    cap_info = (df.groupby(["season", "city"], dropna=False)["capacity"].max().rename("perf_capacity").reset_index() if "capacity" in df.columns else None)

    open_by_sc = perf_info
    if cap_info is not None:
        open_by_sc = open_by_sc.merge(cap_info, on=["season", "city"], how="left")
        open_by_sc = open_by_sc.rename(columns={"perf_capacity": "total_capacity"})

    # If num_shows is present (from runs.csv assignment), keep it
    if "num_shows" in df.columns:
        ns = df.groupby(["season", "city"], dropna=False)["num_shows"].max().reset_index()
        open_by_sc = open_by_sc.merge(ns, on=["season", "city"], how="left")

    # If not, approximate num_shows as unique performance_date values (if the file encodes multiple run dates)
    if "num_shows" not in open_by_sc.columns:
        approx = df.groupby(["season", "city"], dropna=False)["performance_date"].nunique().rename("num_shows").reset_index()
        open_by_sc = open_by_sc.merge(approx, on=["season", "city"], how="left")

    # Closing date = Dec 24 of opening year
    open_by_sc["closing_date"] = pd.to_datetime(dict(year=open_by_sc["opening_date"].dt.year, month=12, day=24))

    daily = aggregate_daily(df).merge(open_by_sc, on=["season", "city"], how="left")
    daily["days_out"] = (daily["sale_date"].dt.normalize() - daily["opening_date"]).dt.days
    daily["days_to_close"] = (daily["closing_date"] - daily["sale_date"].dt.normalize()).dt.days * -1  # negative before Dec 24, 0 on Dec 24

    daily = daily.sort_values(["season", "city", "sale_date"]).reset_index(drop=True)
    daily["cum_qty"] = daily.groupby(["season", "city"], as_index=False)["qty"].cumsum()
    if "revenue" in daily.columns:
        daily["cum_rev"] = daily.groupby(["season", "city"], as_index=False)["revenue"].cumsum()
    else:
        daily["cum_rev"] = np.nan

    finals
