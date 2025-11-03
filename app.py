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
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Nutcracker Sales Trajectory Tracker")
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

def _is_weird_season_label(s: str) -> bool:
    """
    Tag 'non-normal' seasons. Right now we only treat any label containing
    '2021' as weird (e.g., '2021', '2021/22', 'Season 2021-22').
    """
    return "2021" in str(s)


def _city_closing_day(city: str) -> int:
    """
    City-specific closing day within December.

    - Edmonton: runs to Dec 7
    - Calgary & Combined (and any others): Dec 24
    """
    if isinstance(city, str) and city.strip().lower() == "edmonton":
        return 7
    return 24

def compute_calendar_refs(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Build meta per (season,city) with Dec 24 closing
    meta = (
        df[["season", "city"]].drop_duplicates()
          .assign(closing_year=lambda x: x["season"].apply(_season_to_closing_year))
    )
    if meta["closing_year"].isna().any():
        bad = meta[meta["closing_year"].isna()]["season"].unique().tolist()
        raise ValueError(f"Could not parse closing year from season labels: {bad}")

    # City-specific closing dates: Edmonton ends on Dec 7, others on Dec 24
    meta["closing_date"] = meta.apply(
        lambda r: pd.Timestamp(
            year=int(r["closing_year"]),
            month=12,
            day=_city_closing_day(r["city"]),
        ),
        axis=1,
    )

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
                      run_meta: pd.DataFrame,
                      seasons_ref: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # Ensure integer-ish DTC for joins
    daily["days_to_close"]     = pd.to_numeric(daily["days_to_close"], downcast="integer", errors="coerce")
    ref_curve["days_to_close"] = pd.to_numeric(ref_curve["days_to_close"], downcast="integer", errors="coerce")

    # This-season rows (extended) + meta + ref-curve
    cur = daily[daily["season"] == this_season].copy()
    cur = cur.merge(run_meta[["season", "city", "num_shows", "total_capacity"]], on=["season", "city"], how="left")
    cur = cur.merge(ref_curve, on=["city", "days_to_close"], how="left")

    # Raw this-season (for "actual" anchor) and reference history
    this_raw = daily[(daily["season"] == this_season)].copy()
    ref_hist_all = daily[daily["season"].isin(seasons_ref)].copy()

    # Last actual-sale date per city (not counting padded zeros)
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
            mask_anchor = g["cum_qty"].notna()
        else:
            mask_anchor = g["sale_date"] <= last_actual

        g["mean_share_ffill"] = g["mean_share"].ffill()

        # ---- Central scale from "today" (shape-based, as before) ----
        if mask_anchor.any():
            cum_qty_today = (
                g.loc[mask_anchor, "cum_qty"].dropna().iloc[-1]
                if g.loc[mask_anchor, "cum_qty"].notna().any()
                else np.nan
            )
            ref_today_share = (
                g.loc[mask_anchor, "mean_share_ffill"].dropna().iloc[-1]
                if g.loc[mask_anchor, "mean_share_ffill"].notna().any()
                else np.nan
            )
            dtc_anchor = (
                g.loc[mask_anchor, "days_to_close"].dropna().iloc[-1]
                if g.loc[mask_anchor, "days_to_close"].notna().any()
                else np.nan
            )
        else:
            cum_qty_today, ref_today_share, dtc_anchor = np.nan, np.nan, np.nan

        scale_qty = np.nan
        if (
            pd.notna(ref_today_share)
            and ref_today_share > 0
            and pd.notna(cum_qty_today)
        ):
            scale_qty = cum_qty_today / ref_today_share  # central final tickets

        # ---- Backtest-style multipliers from historical seasons ----
        low_factor, high_factor = 1.0, 1.0
        if pd.notna(dtc_anchor) and not ref_hist_all.empty:
            ref_city = ref_hist_all[ref_hist_all["city"] == city]
            R_list = []

            for season, h in ref_city.groupby("season"):
                h = h.sort_values("days_to_close")

                # Use the last available date that is <= today's DTC in that season
                h_cut = h[h["days_to_close"] <= dtc_anchor]
                if h_cut.empty:
                    continue

                row = h_cut.iloc[-1]
                cum = row["cum_qty"]
                final = row["final_qty"]

                if (
                    pd.notna(cum) and cum > 0
                    and pd.notna(final) and final > 0
                ):
                    R_list.append(final / cum)  # "how much higher do we end vs today?"

            if len(R_list) >= 2:
                R_arr = np.array(R_list)
                median_R = np.median(R_arr)
                low_R = np.min(R_arr)
                high_R = np.max(R_arr)

                if median_R > 0:
                    low_factor = low_R / median_R
                    high_factor = high_R / median_R

        # ---- Apply capacity ceiling on ALL finals ----
        cap_total = np.nan
        if "total_capacity" in g.columns and g["total_capacity"].notna().any():
            cap_total = g["total_capacity"].dropna().iloc[0]

        final_mean = scale_qty
        final_low = scale_qty * low_factor if pd.notna(scale_qty) else np.nan
        final_high = scale_qty * high_factor if pd.notna(scale_qty) else np.nan

        if (
            pd.notna(cap_total)
            and cap_total > 0
        ):
            if pd.notna(final_mean):
                final_mean = min(final_mean, cap_total)
            if pd.notna(final_low):
                final_low = min(final_low, cap_total)
            if pd.notna(final_high):
                final_high = min(final_high, cap_total)

        # Ticket projections along full horizon (including future dates)
        g["proj_cum_qty"] = (
            g["mean_share_ffill"] * final_mean if pd.notna(final_mean) else np.nan
        )
        g["proj_min_cum_qty"] = (
            g["mean_share_ffill"] * final_low if pd.notna(final_low) else np.nan
        )
        g["proj_max_cum_qty"] = (
            g["mean_share_ffill"] * final_high if pd.notna(final_high) else np.nan
        )

        # ---------- Revenue projection ----------
        if "cum_rev" in g.columns and g["cum_rev"].notna().any():
            # Simple approach for now: keep your existing logic (share-of-final or ASP)
            if mask_anchor.any():
                cum_rev_today = (
                    g.loc[mask_anchor, "cum_rev"].dropna().iloc[-1]
                    if g.loc[mask_anchor, "cum_rev"].notna().any()
                    else np.nan
                )
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
                if (
                    pd.notna(ref_rev_share_today)
                    and ref_rev_share_today > 0
                    and pd.notna(cum_rev_today)
                ):
                    proj_final_rev_shape = cum_rev_today / ref_rev_share_today

                g["proj_cum_rev"] = (
                    g["mean_rev_share_ffill"] * proj_final_rev_shape
                    if pd.notna(proj_final_rev_shape)
                    else np.nan
                )
            else:
                # Fallback: avg price so far × projected qty
                avg_price = (
                    cum_rev_today / cum_qty_today
                    if (
                        pd.notna(cum_rev_today)
                        and pd.notna(cum_qty_today)
                        and cum_qty_today > 0
                    )
                    else np.nan
                )
                g["proj_cum_rev"] = (
                    g["proj_cum_qty"] * avg_price if pd.notna(avg_price) else np.nan
                )
        else:
            g["proj_cum_rev"] = np.nan

        proj_rows.append(g)

    proj = pd.concat(proj_rows, ignore_index=True) if proj_rows else cur

    # ---------- Summary (prefer exact closing-day row) ----------
    summaries = []
    for city, g in proj.groupby("city"):
        g = g.sort_values("days_to_close")
        final_rows = g[g["days_to_close"] == 0]
        final_row = (
            final_rows.iloc[0]
            if not final_rows.empty
            else (g.loc[g["days_to_close"].idxmax()] if g["days_to_close"].notna().any() else None)
        )

        cap_total = final_row["total_capacity"] if (final_row is not None and "total_capacity" in final_row) else np.nan
        proj_final_qty = final_row["proj_cum_qty"] if final_row is not None else np.nan
        proj_final_rev = final_row["proj_cum_rev"] if final_row is not None else np.nan
        pct_cap = (
            proj_final_qty / cap_total
            if (pd.notna(proj_final_qty) and pd.notna(cap_total) and cap_total > 0)
            else np.nan
        )

        current_cum_qty = g["cum_qty"].dropna().iloc[-1] if g["cum_qty"].notna().any() else np.nan

        summaries.append(dict(
            season=this_season,
            city=city,
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

# ----------------------------
# Optional: latest sales since CSV export
# ----------------------------
last_file_date = this_df["sale_date"].max().date()

with st.sidebar:
    st.header("3) Sales Play (optional)")
    latest_sales_date = st.date_input(
        "Enter date of sales",
        value=last_file_date,
        help="If you've booked more tickets since the CSV export, use this to update projections.",
    )
    latest_tickets = st.number_input(
        "Tickets sold since CSV export",
        min_value=0,
        step=1,
    )
    latest_revenue = st.number_input(
        "Revenue since CSV export",
        min_value=0.0,
        step=100.0,
        format="%.2f",
    )

# If the team entered additional sales, treat them as a new "as of" row
if (latest_tickets > 0) or (latest_revenue > 0):
    extra_date_ts = pd.to_datetime(latest_sales_date)

    # Warn if the date isn't actually after the CSV
    if extra_date_ts <= this_df["sale_date"].max():
        st.warning(
            "Latest sales date is on or before the last date in the CSV; "
            "treating these as extra sales on that same day."
        )

    new_row = {
        "season": this_season,
        "sale_date": extra_date_ts,
        "qty": int(latest_tickets),
    }

    if "revenue" in this_df.columns:
        new_row["revenue"] = float(latest_revenue)

    if "city" in this_df.columns:
        new_row["city"] = this_df["city"].mode()[0]

    if "source" in this_df.columns:
        # assume extra sales follow the dominant source this year
        new_row["source"] = this_df["source"].mode()[0]

    this_df = pd.concat([this_df, pd.DataFrame([new_row])], ignore_index=True)

# ----------------------------
# Ticketing source filter (Ticketmaster vs Archtics etc.)
# ----------------------------
with st.sidebar:
    st.header("4) Ticketing source")
    all_sources = []

    if "source" in hist_df.columns:
        all_sources += hist_df["source"].dropna().astype(str).tolist()
    if "source" in this_df.columns:
        all_sources += this_df["source"].dropna().astype(str).tolist()

    all_sources = sorted(set(all_sources))

    if all_sources:
        selected_sources = st.multiselect(
            "Include sources",
            options=all_sources,
            default=all_sources,
            help="Filter to Ticketmaster, Archtics, or both.",
        )
    else:
        selected_sources = []

# Apply source filter to data frames before building all_df
if selected_sources:
    if "source" in hist_df.columns:
        hist_df = hist_df[hist_df["source"].isin(selected_sources)]
    if "source" in this_df.columns:
        this_df = this_df[this_df["source"].isin(selected_sources)]

# Optional city collapsing
all_df = pd.concat([hist_df, this_df], ignore_index=True)
if city_mode == "Combined":
    all_df["city"] = "Combined"

# Core calendar + cumulative metrics
daily, run_meta = compute_calendar_refs(all_df)

# Sidebar: choose reference seasons (exclude current)
with st.sidebar:
    seasons_all = sorted(daily["season"].unique().tolist())

    # Exclude this year and any 'weird' seasons (e.g. 2021/22) by default
    ref_default = [
        s for s in seasons_all
        if (s != this_season) and (not _is_weird_season_label(s))
    ]

    seasons_ref = st.multiselect(
        "Reference seasons (exclude this year)",
        options=[s for s in seasons_all if s != this_season],
        default=ref_default,
        help="2021-style seasons are excluded by default but can be added if you really want.",
    )

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

# ----------------------------
# Projection + plotting prep
# ----------------------------
# Project on the extended data
proj_df, summary_df = project_this_year(
    daily_extended, this_season, ref_curve, run_meta, seasons_ref
)

# Optionally gate projections for display only (don't touch summary_df)
plot_proj = proj_df.copy()
if "mean_share" in plot_proj.columns:
    too_early = plot_proj["mean_share"].notna() & (plot_proj["mean_share"] < min_share_to_project)
    plot_proj.loc[too_early, ["proj_cum_qty", "proj_min_cum_qty", "proj_max_cum_qty"]] = np.nan

# ----------------------------
# MAIN PAGE (chart + data table)
# ----------------------------
st.subheader("Actual vs projected cumulative tickets (historical vs this year)")

# 1) Historical cumulative tickets (absolute), reference seasons only
hist_abs = daily[
    daily["season"].isin(seasons_ref)
    & daily["cum_qty"].notna()
    & daily["days_to_close"].notna()
    & (daily["days_to_close"] >= -window_days)
].copy()

# 2) This year ACTUALS — from non-extended frame (stops at last real sale)
this_abs_actual = this_daily[
    this_daily["cum_qty"].notna()
    & this_daily["days_to_close"].notna()
    & (this_daily["days_to_close"] >= -window_days)
].copy()

# 3) This year PROJECTION — from proj_df (only from last actual onward)
last_actual_date = this_daily["sale_date"].max()
dtc_today = this_daily.loc[
    this_daily["sale_date"] == last_actual_date, "days_to_close"
].iloc[0]

this_abs_proj = proj_df[
    (proj_df["season"] == this_season)
    & proj_df["proj_cum_qty"].notna()
    & proj_df["days_to_close"].notna()
    & (proj_df["days_to_close"] >= dtc_today)
    & (proj_df["days_to_close"] >= -window_days)
].copy()

# ----------------------------
# Combined view
# ----------------------------
if city_mode == "Combined":
    hist_chart = alt.Chart(hist_abs).mark_line(opacity=0.35).encode(
        x=alt.X("days_to_close:Q", title="Days to closing (Dec 24)"),
        y=alt.Y("cum_qty:Q", title="Cumulative tickets"),
        color=alt.Color("season:N", title="Historical seasons"),
        tooltip=["season", "city", "sale_date", "cum_qty"],
    )

    cur_actual_line = alt.Chart(this_abs_actual).mark_line(size=3).encode(
        x="days_to_close:Q",
        y=alt.Y("cum_qty:Q", title="Cumulative tickets"),
        color=alt.Color(
            "season:N",
            scale=alt.Scale(domain=[this_season], range=["navy"]),
            legend=alt.Legend(title="This season"),
        ),
        tooltip=["city", "season", "sale_date", "cum_qty"],
    )

    cur_proj_line = alt.Chart(this_abs_proj).mark_line(
        size=2,
        strokeDash=[4, 2],  # dashed
    ).encode(
        x="days_to_close:Q",
        y="proj_cum_qty:Q",
        color=alt.value("navy"),  # same color, no extra legend
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
        chart = (
            alt.layer(*layers)
            .resolve_scale(color="independent")
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No data available yet for historicals and projections.")

# ----------------------------
# By-city view: one chart per city
# ----------------------------
else:
    cities = sorted(daily["city"].dropna().unique().tolist())

    for city in cities:
        st.markdown(f"### {city}")

        hist_c = hist_abs[hist_abs["city"] == city]
        this_actual_c = this_abs_actual[this_abs_actual["city"] == city]
        this_proj_c = this_abs_proj[this_abs_proj["city"] == city]

        hist_chart_c = alt.Chart(hist_c).mark_line(opacity=0.35).encode(
            x=alt.X("days_to_close:Q", title="Days to closing (Dec 24)"),
            y=alt.Y("cum_qty:Q", title="Cumulative tickets"),
            color=alt.Color("season:N", title="Historical seasons"),
            tooltip=["season", "city", "sale_date", "cum_qty"],
        )

        cur_actual_line_c = alt.Chart(this_actual_c).mark_line(size=3).encode(
            x="days_to_close:Q",
            y=alt.Y("cum_qty:Q", title="Cumulative tickets"),
            color=alt.Color(
                "season:N",
                scale=alt.Scale(domain=[this_season], range=["navy"]),
                legend=alt.Legend(title="This season"),
            ),
            tooltip=["city", "season", "sale_date", "cum_qty"],
        )

        cur_proj_line_c = alt.Chart(this_proj_c).mark_line(
            size=2,
            strokeDash=[4, 2],
        ).encode(
            x="days_to_close:Q",
            y="proj_cum_qty:Q",
            color=alt.value("navy"),
            tooltip=["city", "season", "sale_date", "proj_cum_qty"],
        )

        layers_c = []
        if not hist_c.empty:
            layers_c.append(hist_chart_c)
        if not this_actual_c.empty:
            layers_c.append(cur_actual_line_c)
        if not this_proj_c.empty:
            layers_c.append(cur_proj_line_c)

        if layers_c:
            chart_c = (
                alt.layer(*layers_c)
                .resolve_scale(color="independent")
                .properties(height=260)
            )
            st.altair_chart(chart_c, use_container_width=True)
        else:
            st.info(f"No data available yet for {city}.")


# ----------------------------
# Data table + download (this season by day)
# ----------------------------
st.subheader("Projection by day (this season)")

table_df = plot_proj[plot_proj["season"] == this_season].copy()

# Column order (internal names)
cols_order = [
    "season",
    "city",
    "sale_date",
    "days_to_close",
    "cum_qty",
    "proj_cum_qty",
    "proj_min_cum_qty",
    "proj_max_cum_qty",
    "cum_rev",
    "proj_cum_rev",
]
cols_order = [c for c in cols_order if c in table_df.columns]
table_df = table_df[cols_order].sort_values(
    ["city", "sale_date"],
    ascending=[True, False],  # city A–Z, newest dates first
)

# Lay-friendly headers (display + CSV only)
pretty_cols = {
    "season": "Season",
    "city": "City",
    "sale_date": "Date",
    "days_to_close": "Days to closing (Dec 24)",
    "cum_qty": "Tickets sold so far",
    "proj_cum_qty": "Projected tickets",
    "proj_min_cum_qty": "Projection – low",
    "proj_max_cum_qty": "Projection – high",
    "cum_rev": "Revenue so far",
    "proj_cum_rev": "Projected revenue",
}

display_df = table_df.rename(columns=pretty_cols)

st.dataframe(display_df, use_container_width=True, hide_index=True)

st.download_button(
    label="Download projection by day (CSV)",
    data=display_df.to_csv(index=False).encode("utf-8"),
    file_name=f"nutcracker_projection_by_day_{this_season}.csv",
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
