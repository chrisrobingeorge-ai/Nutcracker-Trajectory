# Nutcracker Sales Trajectory Tracker

A Streamlit app that compares this year's Nutcracker sales trajectory with prior seasons, normalizing for different numbers of performances (and capacity if available), and projecting a final outcome from historical cumulative share curves.

## Features
- Upload historical multi-year CSV + this-year-to-date CSV
- Auto-detects columns (season/year, order_date, performance_date, qty, city, capacity, performance_id)
- Normalizes per-show (if no capacity) or by capacity (if provided)
- Reference curve: mean + min/max cumulative share by day from opening
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

## Repo structure
