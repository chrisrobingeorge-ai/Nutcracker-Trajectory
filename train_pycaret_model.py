# train_pycaret_model.py
# One-time script to train a PyCaret regression model from your historical sales CSV
# and save it as a .pkl model file for the app to load.
#
# Adapt DATA_PATH and TARGET_COL to match Nutcracker-Trajectory's historical data.

import pandas as pd
from pycaret.regression import setup, compare_models, save_model

# ---------- CONFIG (edit these for your repo) ----------
DATA_PATH = "data/history_sales.csv"   # adapt to your repo
TARGET_COL = "Total_Sales"             # adapt to your repo
ID_COL = "Show_Title"                  # optional identifier column to ignore as feature
MODEL_NAME = "nutcracker_demand_model"
# ------------------------------------------------------

# 1. Load data
df = pd.read_csv(DATA_PATH, thousands=",")

# 2. Normalize column names (PyCaret/LightGBM prefer no spaces)
df.columns = [c.replace(" ", "_") for c in df.columns]

# 3. If your history has separate ticket columns, compute the target. Otherwise skip.
ticket_cols = [c for c in df.columns if "Ticket" in c or "Tickets" in c]
if ticket_cols and TARGET_COL not in df.columns:
    df[TARGET_COL] = df[ticket_cols].sum(axis=1)

# 4. Ensure numeric columns are numeric where appropriate
for c in df.select_dtypes(include=["object"]).columns:
    try:
        df[c] = pd.to_numeric(df[c].str.replace(",", ""), errors="ignore")
    except Exception:
        pass

# 5. Preview
print("Preview of training data:")
print(df.head())

# 6. Setup PyCaret
s = setup(
    data=df,
    target=TARGET_COL,
    session_id=42,
    normalize=True,
    feature_selection=True,
    remove_multicollinearity=True,
    ignore_features=[ID_COL] if ID_COL in df.columns else None,
    silent=True,
)

# 7. Compare models and save best (sorted by MAE)
best_model = compare_models(n_select=1, sort="MAE")
save_model(best_model, MODEL_NAME)
print(f"\n✓ Model saved as '{MODEL_NAME}.pkl' in this folder.")
