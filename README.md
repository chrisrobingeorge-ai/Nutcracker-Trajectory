# Nutcracker Sales Trajectory Tracker

A Streamlit app that compares this year's Nutcracker sales trajectory with prior seasons, normalizing for different numbers of performances (and capacity if available), and projecting a final outcome from historical cumulative share curves.

## Python Version Requirement

This app requires **Python 3.11.10** for deployment. The `runtime.txt` file specifies this exact version to ensure:
- Optimal wheel availability for all dependencies (pandas, numpy, streamlit)
- No compilation from source is needed during deployment
- Compatibility with Streamlit Cloud and all dependencies including Altair charts
- Support for optional PyCaret ML features

**Note:** Using a specific Python version (e.g., 3.11.10) instead of a generic version (e.g., 3.11) ensures consistent deployments and avoids build failures.

## Features
- Upload historical multi-year CSV + this-year-to-date CSV
- Auto-detects columns (season/year, order_date, performance_date, qty, city, capacity, performance_id)
- Normalizes per-show (if no capacity) or by capacity (if provided)
- Reference curve: mean + min/max cumulative share by day from opening
- **ML-based projections using PyCaret** (optional) - Train AutoML regression models on historical data for alternative projections
- Calgary / Edmonton split or Combined
- Exports projection-by-day and summary CSVs

## Data format
Two CSVs:
1. **Historical** (multiple past seasons)
2. **This season (to-date)** (exactly one season label)

**Required columns** (case-insensitive):
- `season` or `year` — e.g., `2023` or `2023-24`
- `order_date` (or `sale_date`) — date tickets were sold (YYYY-MM-DD)
- `performance_date` — opening date for the Nutcracker run (YYYY-MM-DD). If you have multiple opening weeks, use the first performance date.
- `qty` (or `tickets_sold`) — integer count per day (or summed from transactions)

**Optional, recommended:**
- `city` — e.g., `Calgary`, `Edmonton` (otherwise the app uses `Combined`)
- `performance_id` — unique ID per run (fallback is `season + performance_date`)
- `capacity` — per-performance or total capacity (the app aggregates, but total per season is fine)
- `revenue` — enables a revenue trajectory view

> If you have per-transaction data, the app will aggregate daily.

## ML-based Projections (PyCaret) - Optional

The app includes **optional** machine learning-based projections using **PyCaret**, an AutoML library that automatically trains and selects the best regression model.

> **Note**: PyCaret is completely optional. The app works perfectly without it, using traditional curve-based projections.

### How it works:
1. **Enable in UI**: Check "Enable ML-based projections (PyCaret)" in the sidebar (only appears if PyCaret is installed)
2. **Automatic training**: The app trains multiple regression models on historical data and selects the best performer
3. **Features used**: Days to close, cumulative tickets, per-show metrics, seasonality (day of week, month)
4. **Comparison view**: See both curve-based and ML-based projections side-by-side

### Benefits:
- **Complementary approach**: ML models can capture non-linear patterns that simple averaging might miss
- **Automatic feature engineering**: PyCaret handles normalization and preprocessing
- **Multiple models**: Compares various algorithms (Linear, Ridge, Random Forest, XGBoost, etc.) and picks the best
- **No configuration needed**: Works out-of-the-box with default settings

### Installation (Optional):
PyCaret is **not included** in the default `requirements.txt` to ensure broad compatibility.

**Requirements:**
- **Python version**: 3.9, 3.10, or 3.11 (This app uses Python 3.11 for optimal compatibility with Streamlit Cloud and all dependencies)

**Install PyCaret separately:**
```bash
pip install -r requirements-pycaret.txt
```

Or directly:
```bash
pip install pycaret>=3.3.2
```

The app gracefully handles missing PyCaret and will show a notification in the sidebar when it's not available.

## Dependencies

The `requirements.txt` uses carefully chosen version constraints to ensure:
- **streamlit>=1.33**: Latest features and security fixes
- **pandas>=2.0**: Modern pandas API, allows latest stable versions
- **numpy>=1.23,<2.0**: Compatible with streamlit and pandas, with pre-built wheels
- **altair>=5.0**: Modern declarative visualization

These constraints ensure that pip can resolve dependencies using pre-built wheels, avoiding compilation errors during deployment.

## Deployment Notes

If you encounter "ninja: build stopped" or "metadata-generation-failed" errors during deployment:
1. Ensure `runtime.txt` specifies an exact Python version (e.g., `python-3.11.10`)
2. Verify package constraints allow for pre-built wheels
3. Avoid overly restrictive upper bounds (e.g., use `pandas>=2.0` instead of `pandas>=2.0,<2.2`)

## Repo structure
