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

    # --- Normalize key columns early ---
    # City: trim, unify NaNs, and standardize common variants
    if "city" in df.columns:
        df["city"] = (
            df["city"]
            .astype(str)
            .str.strip()
            .replace({"nan": np.nan})
            .fillna("Combined")
        )
        CITY_MAP = {
            "calgary": "Calgary",
            "edmonton": "Edmonton",
            "yyc": "Calgary",
            "yeg": "Edmonton",
            "combined": "Combined",
        }
        df["city"] = df["city"].str.lower().map(CITY_MAP).fillna(df["city"])
    else:
        df["city"] = "Combined"

    # Season: trim
    df["season"] = df["season"].astype(str).str.strip()

    df["sale_date"] = _coerce_dates(df, "sale_date")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0).astype(int)
    if "revenue" in df.columns:
        df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0.0)
    if "capacity" in df.columns:
        df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce").astype("Float64")

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
    """
    Extract the closing year from a season string.
    Accepts: "2025", "2024-25", "2024/25", "2024–25", "2024-2025", "Season 2024/25", etc.
    """
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    import re
    # explicit 4-digit at end
    m = re.search(r"(20\d{2})\s*$", s)
    if m:
        return int(m.group(1))
    # 4-digit sep 2/4-digit
    m = re.search(r"(20\d{2})\s*[-/–]\s*(\d{2,4})", s)
    if m:
        start = int(m.group(1))
        tail = m.group(2)
        if len(tail) == 2:
            return int(str(start)[:2] + tail)
        elif len(tail) == 4:
            return int(tail)
    # any 4-digit anywhere
    m = re.search(r"(20\d{2})", s)
    if m:
        return int(m.group(1))
    return None

def _extend_current_to_closing(daily: pd.DataFrame, this_season: str) -> pd.DataFrame:
    """
    Extend the current season to Dec 24 per city by adding daily rows with qty=0 (and revenue=0 if present).
    Critically: stamp `closing_date` onto these new rows to avoid NaNs later.
    """
    cur = daily[daily["season"] == this_season].copy()
    if cur.empty:
        return daily

    required = {"season", "city", "sale_date", "closing_date"}
    missing = required - set(cur.columns)
    if missing:
        raise ValueError(f"_extend_current_to_closing: missing columns in daily: {sorted(missing)}")

    new_rows = []
    for city, g in cur.groupby("city"):
        close = pd.to_datetime(g["closing_date"].iloc[0]).normalize()
        last_dt = pd.to_datetime(g["sale_date"].max()).normalize()
        if pd.isna(close) or pd.isna(last_dt) or last_dt >= close:
            continue

        future_days = pd.date_range(start=last_dt + pd.Timedelta(days=1), end=close, freq="D")
        n = len(future_days)
        if n == 0:
            continue

        block = {
            "season": [this_season] * n,
            "city":   [city] * n,
            "sale_date": future_days,
            "qty": np.zeros(n, dtype=int),
            "closing_date": [close] * n,           # <<<<<<<<<<<<<<  IMPORTANT
        }
        if "revenue" in daily.columns:
            block["revenue"] = np.zeros(n, dtype=float)

        new_rows.append(pd.DataFrame(block))

    if not new_rows:
        return daily

    fut = pd.concat(new_rows, ignore_index=True)
    out = pd.concat([daily, fut], ignore_index=True).sort_values(["season", "city", "sale_date"])
    return out.reset_index(drop=True)

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
    if "capacity" in df.columns:
        cap = (df.groupby(["season", "city"], dropna=False)["capacity"]
                 .max().rename("total_capacity").reset_index())
        out_meta = out_meta.merge(cap, on=["season", "city"], how="left")
    else:
        out_meta["total_capacity"] = np.nan

    if "num_shows" in df.columns:
        ns = (df.groupby(["season", "city"], dropna=False)["num_shows"]
                .max().reset_index())
        out_meta = out_meta.merge(ns, on=["season", "city"], how="left")
    else:
        out_meta["num_shows"] = np.nan

    daily = aggregate_daily(df).merge(out_meta, on=["season", "city"], how="left")

    # --- Repair any (season, city) that somehow missed a closing_date ---
    if "closing_date" not in daily.columns:
        daily["closing_date"] = pd.NaT
    
    mask_na_close = daily["closing_date"].isna()
    if mask_na_close.any():
        # Try recomputing from season string directly
        fix = daily.loc[mask_na_close, ["season"]].copy()
        fix["closing_year"] = fix["season"].apply(_season_to_closing_year)
        with np.errstate(all='ignore'):
            daily.loc[mask_na_close, "closing_date"] = pd.to_datetime(
                dict(year=fix["closing_year"].astype("Int64"), month=12, day=24),
                errors="coerce"
            )
    
        # If still missing, surface a clear error with the offending keys
        still = daily.loc[daily["closing_date"].isna(), ["season", "city"]].drop_duplicates()
        if not still.empty:
            raise ValueError(
                "Could not determine closing_date for these (season, city) pairs. "
                "Check season strings and city spellings:\n"
                + still.to_string(index=False)
            )

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
    if not seasons_ref:
        return pd.DataFrame(columns=[
            "city","days_to_close","mean_share","min_share","max_share",
            "mean_per_show","mean_rev_share","min_rev_share","max_rev_share"
        ])

    ref = daily[daily["season"].isin(seasons_ref)].copy()

    # Compute revenue share if we have rev
    if "cum_rev" in ref.columns and "final_rev" in ref.columns and ref["final_rev"].notna().any():
        ref["share_of_final_rev"] = ref["cum_rev"] / ref["final_rev"].replace(0, np.nan)
    else:
        ref["share_of_final_rev"] = np.nan

    agg = (
        ref.groupby(["city", "days_to_close"], dropna=False)
           .agg(
               mean_share=("share_of_final_qty", "mean"),
               min_share=("share_of_final_qty", "min"),
               max_share=("share_of_final_qty", "max"),
               mean_per_show=("per_show_cum_qty", "mean"),
               mean_rev_share=("share_of_final_rev", "mean"),
               min_rev_share=("share_of_final_rev", "min"),
               max_rev_share=("share_of_final_rev", "max"),
           )
           .reset_index()
    )
    return agg

def _densify_ref_curve(ref_curve: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    if ref_curve.empty:
        return ref_curve

    hist = daily[daily["share_of_final_qty"].notna()]
    limits = (hist.groupby("city")["days_to_close"].agg(["min", "max"]).reset_index()
              .rename(columns={"min":"min_dtc","max":"max_dtc"}))
    limits["max_dtc"] = 0

    out = []
    for _, row in limits.iterrows():
        city = row["city"]
        g = ref_curve[ref_curve["city"] == city].copy()
        if g.empty:
            continue

        idx = pd.Index(range(int(row["min_dtc"]), 1), name="days_to_close")
        g = (g.set_index("days_to_close").reindex(idx).sort_index())

        # interpolate shares/rev shares + per-show
        for col in ["mean_share","min_share","max_share","mean_per_show",
                    "mean_rev_share","min_rev_share","max_rev_share"]:
            if col in g.columns:
                g[col] = g[col].interpolate(method="linear", limit_direction="both")

        # **HARD ANCHOR** at closing day
        if 0 in g.index:
            for col in ["mean_share","min_share","max_share"]:
                if col in g.columns:
                    g.loc[0, col] = 1.0

        # clip + monotone for ticket shares
        for col in ["mean_share","min_share","max_share"]:
            if col in g.columns:
                g[col] = g[col].clip(0.0, 1.0).cummax()

        g = g.reset_index()
        g["city"] = city
        out.append(g)

    return pd.concat(out, ignore_index=True) if out else ref_curve

def project_this_year(daily: pd.DataFrame,
                      this_season: str,
                      ref_curve: pd.DataFrame,
                      run_meta: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # Ensure integer-ish DTC for joins
    daily["days_to_close"]     = pd.to_numeric(daily["days_to_close"], downcast="integer", errors="coerce")
    ref_curve["days_to_close"] = pd.to_numeric(ref_curve["days_to_close"], downcast="integer", errors="coerce")

    # Join this-year rows with meta + ref-curve
    cur = daily[daily["season"] == this_season].copy()
    cur = cur.merge(run_meta[["season","city","num_shows","total_capacity"]], on=["season","city"], how="left")
    cur = cur.merge(ref_curve, on=["city","days_to_close"], how="left")

    # ---- NEW: compute the last actual-sale date per city (pre-extension) ----
    this_raw = daily[(daily["season"] == this_season)].copy()
    # “Actual” = rows that existed before we padded zeros; treat any qty>0 OR the max date before the flat pad as actual
    last_actual_by_city = (
        this_raw[this_raw["qty"] > 0]
        .groupby("city")["sale_date"].max()
        .rename("last_actual_date")
    )

    proj_rows: List[pd.DataFrame] = []

    for city, g in cur.groupby("city"):
        g = g.sort_values("sale_date").copy()

        # Anchor window = rows up to last actual sale for this city
        last_actual = last_actual_by_city.get(city, pd.NaT)
        if pd.isna(last_actual):
            # fall back to max date in g that has any cum_qty
            mask_anchor = g["cum_qty"].notna()
        else:
            mask_anchor = g["sale_date"] <= last_actual

        g["mean_share_ffill"] = g["mean_share"].ffill()
        g["min_share_ffill"]  = g["min_share"].ffill() if "min_share" in g.columns else np.nan
        g["max_share_ffill"]  = g["max_share"].ffill() if "max_share" in g.columns else np.nan

        # ---- scale from "today" (last actual) ----
        if mask_anchor.any():
            cum_qty_today = g.loc[mask_anchor, "cum_qty"].dropna().iloc[-1] if g.loc[mask_anchor, "cum_qty"].notna().any() else np.nan
            ref_today_share = g.loc[mask_anchor, "mean_share_ffill"].dropna().iloc[-1] if g.loc[mask_anchor, "mean_share_ffill"].notna().any() else np.nan
        else:
            cum_qty_today, ref_today_share = np.nan, np.nan

        scale_qty = np.nan
        if pd.notna(ref_today_share) and ref_today_share > 0 and pd.notna(cum_qty_today):
            scale_qty = cum_qty_today / ref_today_share

        # Ticket projections along full horizon (including future dates)
        g["proj_cum_qty"]     = g["mean_share_ffill"] * scale_qty if pd.notna(scale_qty) else np.nan
        g["proj_min_cum_qty"] = g["min_share_ffill"]  * scale_qty if (pd.notna(scale_qty) and "min_share" in g.columns) else np.nan
        g["proj_max_cum_qty"] = g["max_share_ffill"]  * scale_qty if (pd.notna(scale_qty) and "max_share" in g.columns) else np.nan

        # ---------- Revenue projection ----------
        if "cum_rev" in g.columns and g["cum_rev"].notna().any():
            if mask_anchor.any():
                cum_rev_today = g.loc[mask_anchor, "cum_rev"].dropna().iloc[-1] if g.loc[mask_anchor, "cum_rev"].notna().any() else np.nan
            else:
                cum_rev_today = np.nan

            if "mean_rev_share" in g.columns and g["mean_rev_share"].notna().any():
                g["mean_rev_share_ffill"] = g["mean_rev_share"].ffill()
                if mask_anchor.any():
                    ref_rev_share_today = (
                        g.loc[mask_anchor, "mean_rev_share_ffill"].dropna().iloc[-1]
                        if g.loc[mask_anchor, "mean_rev_share_ffill"].notna().any()
                        else np.nan
                    )
                else:
                    ref_rev_share_today = np.nan

                proj_final_rev_shape = np.nan
                if pd.notna(ref_rev_share_today) and ref_rev_share_today > 0 and pd.notna(cum_rev_today):
                    proj_final_rev_shape = cum_rev_today / ref_rev_share_today

                g["proj_cum_rev"] = g["mean_rev_share_ffill"] * proj_final_rev_shape if pd.notna(proj_final_rev_shape) else np.nan
            else:
                # Fallback: avg price-so-far × projected qty (still anchored at last actual)
                avg_price = (cum_rev_today / cum_qty_today) if (pd.notna(cum_rev_today) and pd.notna(cum_qty_today) and cum_qty_today > 0) else np.nan
                g["proj_cum_rev"] = g["proj_cum_qty"] * avg_price if pd.notna(avg_price) else np.nan
        else:
            g["proj_cum_rev"] = np.nan

        proj_rows.append(g)

    proj = pd.concat(proj_rows, ignore_index=True) if proj_rows else cur

    # ---------- Summary (prefer exact Dec 24 row; else max DTC) ----------
    summaries = []
    for city, g in proj.groupby("city"):
        g = g.sort_values("days_to_close")
        final_rows = g[g["days_to_close"] == 0]
        final_row = final_rows.iloc[0] if not final_rows.empty else (g.loc[g["days_to_close"].idxmax()] if g["days_to_close"].notna().any() else None)

        cap_total = final_row["total_capacity"] if (final_row is not None and "total_capacity" in final_row) else np.nan
        proj_final_qty = final_row["proj_cum_qty"] if final_row is not None else np.nan
        proj_final_rev = final_row["proj_cum_rev"] if final_row is not None else np.nan
        pct_cap = (proj_final_qty / cap_total) if (pd.notna(proj_final_qty) and pd.notna(cap_total) and cap_total > 0) else np.nan

        current_cum_qty = g["cum_qty"].dropna().iloc[-1] if g["cum_qty"].notna().any() else np.nan

        summaries.append(dict(
            season=this_season, city=city,
            current_cum_qty=current_cum_qty,
            projected_final_qty=proj_final_qty,
            projected_pct_capacity=pct_cap,
            projected_final_revenue=proj_final_rev,
            num_shows=(final_row["num_shows"] if (final_row is not None and "num_shows" in final_row) else np.nan),
            total_capacity=cap_total,
        ))

    summary_df = pd.DataFrame(summaries)
    return proj, summary_df


# ----------------------------
# Sidebar: data discovery (no uploads)
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

# ----------------------------
# Load & prep data
# ----------------------------
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

# Build reference curve and densify
try:
    ref_curve = build_reference_curve(daily, seasons_ref)
except Exception as e:
    st.error(f"Could not build reference curve: {e}")
    st.stop()

ref_curve = _densify_ref_curve(ref_curve, daily)

# Extend the current season to closing day so we can draw projections out to Dec 24
daily_extended = _extend_current_to_closing(daily, this_season)

# ---- Backfill closing_date on any new rows that might still be NaN (belt & suspenders) ----
if daily_extended["closing_date"].isna().any():
    daily_extended = (
        daily_extended.drop(columns=["closing_date"])
        .merge(run_meta[["season","city","closing_date"]], on=["season","city"], how="left")
    )

# Recompute cum fields & days_to_close on the extended frame
daily_extended = daily_extended.sort_values(["season","city","sale_date"]).reset_index(drop=True)
daily_extended["cum_qty"] = daily_extended.groupby(["season","city"])["qty"].cumsum()
if "revenue" in daily_extended.columns and daily_extended["revenue"].notna().any():
    daily_extended["cum_rev"] = daily_extended.groupby(["season","city"])["revenue"].cumsum()
else:
    daily_extended["cum_rev"] = np.nan

# Compute days_to_close exactly once here for the extended frame
daily_extended["days_to_close"] = (
    daily_extended["sale_date"].dt.normalize() - daily_extended["closing_date"]
).dt.days

# 🔎 DEBUG — sanity checks
dbg = daily_extended[daily_extended["season"] == this_season].copy()

# Show last 5 rows per city (after extension)
dbg_tail = (
    dbg.sort_values(["city", "sale_date"])
       .groupby("city", as_index=False)
       .tail(5)
)

# Max DTC should be 0 for every city after extension
last_dtc = dbg.groupby("city")["days_to_close"].max().sort_index().to_dict()
st.caption("🔎 After extension, max days_to_close per city (should be 0)")
st.json(last_dtc)

# Any missing closing_date? (would block extension)
missing_close = (
    dbg[dbg["closing_date"].isna()][["season","city"]]
    .drop_duplicates()
    .sort_values(["season","city"])
)
if not missing_close.empty:
    st.error("⚠️ Missing closing_date for these (season, city) pairs — extension cannot run:")
    st.dataframe(missing_close, use_container_width=True)

# Peek at the tail rows to confirm we actually reached Dec 24 (days_to_close == 0)
st.caption("🔎 Tail rows per city after extension")
st.dataframe(dbg_tail[["season","city","sale_date","days_to_close","cum_qty"]], use_container_width=True)

# Reference curve coverage: do we have shares all the way to 0?
st.caption("🔎 Ref curve coverage (min→max days_to_close) per city")
ref_span = (
    ref_curve.groupby("city")["days_to_close"].agg(["min","max"]).reset_index()
             .sort_values("city")
)
st.dataframe(ref_span, use_container_width=True)

# Also helpful to ensure city labels align across frames
st.caption("🔎 City labels present")
st.write({
    "daily_extended": sorted(daily_extended["city"].dropna().unique().tolist()),
    "ref_curve":      sorted(ref_curve["city"].dropna().unique().tolist()),
})

# Project on the extended data
proj_df, summary_df = project_this_year(daily_extended, this_season, ref_curve, run_meta)

# Trim plotting window
plot_ref  = ref_curve[ref_curve["days_to_close"] >= -window_days].copy()
plot_proj = proj_df[proj_df["days_to_close"] >= -window_days].copy()
plot_hist = ref_daily[ref_daily["days_to_close"] >= -window_days].copy()
plot_this = this_daily[this_daily["days_to_close"] >= -window_days].copy()

# Gate projections only for visualization (not for summary computation)
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

    st.subheader("Actual vs projected cumulative tickets (historical vs this year)")

    # 1) Historical cumulative tickets (absolute), reference seasons only
    hist_abs = daily[
        daily["season"].isin(seasons_ref)
        & daily["cum_qty"].notna()
        & (daily["days_to_close"] >= -window_days)
    ].copy()
    
    hist_chart = alt.Chart(hist_abs).mark_line(opacity=0.35).encode(
        x=alt.X("days_to_close:Q", title="Days to closing (Dec 24)"),
        y=alt.Y("cum_qty:Q", title="Cumulative tickets"),
        color=alt.Color("season:N", title="Historical season"),
        tooltip=["season", "city", "sale_date", "cum_qty"],
    )
    
    # 2) This year ACTUALS — from non-extended frame (stops at last real sale)
    this_abs_actual = this_daily[
        this_daily["cum_qty"].notna()
        & this_daily["days_to_close"].notna()
        & (this_daily["days_to_close"] >= -window_days)
    ].copy()
    
    cur_actual_line = alt.Chart(this_abs_actual).mark_line(size=3).encode(
        x="days_to_close:Q",
        y="cum_qty:Q",
        color=alt.Color(
            "season:N",
            scale=alt.Scale(domain=[this_season], range=["navy"]),
            legend=alt.Legend(title="This season"),
        ),
        tooltip=["city", "season", "sale_date", "cum_qty"],
    )
    
    # 3) This year PROJECTION — from proj_df (extended with projected cum_qty)
    # Find last actual sales date for this season (from non-extended data)
    last_actual_date = this_daily["sale_date"].max()
    dtc_today = this_daily.loc[this_daily["sale_date"] == last_actual_date, "days_to_close"].iloc[0]
    this_abs_proj = proj_df[
        (proj_df["season"] == this_season)
        & proj_df["proj_cum_qty"].notna()
        & proj_df["days_to_close"].notna()
        & (proj_df["days_to_close"] >= dtc_today)          # << only from last actual onward
        & (proj_df["days_to_close"] >= -window_days)
    ].copy()
    
    cur_proj_line = alt.Chart(this_abs_proj).mark_line(strokeDash=[4, 2], size=2).encode(
        x="days_to_close:Q",
        y="proj_cum_qty:Q",
        color=alt.Color(
            "season:N",
            scale=alt.Scale(domain=[this_season], range=["navy"]),
            legend=alt.Legend(title="This season"),
        ),
        tooltip=["city", "season", "sale_date", "proj_cum_qty"],
    )
    
    layers = []
    if not hist_abs.empty:
        layers.append(hist_chart)
    if not this_abs_actual.empty:
        layers.append(cur_actual_line)
    if not this_abs_proj.empty:
        layers.append(cur_proj_line)
    
    if layers:
        chart = alt.layer(*layers).resolve_scale(color="independent").properties(height=300)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No data available yet for historicals and projections.")

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
        st.subheader("Revenue trajectory (with projection when possible)")
    
        # 1) Historical seasons (not this year) — from the original (non-extended) daily
        rev_hist = daily[(daily["season"] != this_season) & daily["cum_rev"].notna()]
        hist_chart = alt.Chart(rev_hist).mark_line(opacity=0.4).encode(
            x=alt.X("days_to_close:Q", title="Days to closing (Dec 24)"),
            y=alt.Y("cum_rev:Q", title="Cumulative revenue"),
            color=alt.Color("season:N", title="Season"),
            tooltip=["season","city","sale_date","cum_rev"],
            detail="city:N",
        )
    
        # 2) Current season actuals — from the non-extended frame (stops at last real sale)
        cur_actual = this_daily.dropna(subset=["days_to_close","cum_rev"])
        cur_line = alt.Chart(cur_actual).mark_line(size=3).encode(
            x="days_to_close:Q",
            y="cum_rev:Q",
            color=alt.Color("city:N", title="City (current)"),
            tooltip=["city","sale_date","cum_rev"],
        )
    
        # 3) Current season projection — from the extended/projection frame
        cur_proj = proj_df[(proj_df["season"] == this_season) & proj_df["proj_cum_rev"].notna()]
        cur_proj_line = alt.Chart(cur_proj).mark_line(strokeDash=[4,2]).encode(
            x="days_to_close:Q",
            y="proj_cum_rev:Q",
            color=alt.Color("city:N", title="City (current)"),
            tooltip=["city","sale_date","proj_cum_rev"],
        )
    
        st.altair_chart((hist_chart + cur_line + cur_proj_line).properties(height=260),
                        use_container_width=True)


with right:
    st.subheader("Summary & projection")
    if not summary_df.empty:
        st.dataframe(
            summary_df.assign(
                projected_final_qty=summary_df["projected_final_qty"].round(0),
                projected_pct_capacity=(summary_df["projected_pct_capacity"] * 100).round(1),
                projected_final_revenue=summary_df["projected_final_revenue"].round(0)
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
