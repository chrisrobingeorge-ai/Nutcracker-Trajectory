# train_pycaret_model.py
# One-time script to train a PyCaret regression model from your historical sales CSV
# and save it as a .pkl model file for the app to load.
#
# Adapt column names below to match Nutcracker-Trajectory's historical data.

import sys
import pandas as pd

# Check Python version compatibility before importing PyCaret
PYCARET_MIN_PYTHON = (3, 9)
PYCARET_MAX_PYTHON = (3, 12)

current_version = sys.version_info[:2]
if not (PYCARET_MIN_PYTHON <= current_version <= PYCARET_MAX_PYTHON):
    print(f"Error: PyCaret requires Python {PYCARET_MIN_PYTHON[0]}.{PYCARET_MIN_PYTHON[1]} "
          f"through {PYCARET_MAX_PYTHON[0]}.{PYCARET_MAX_PYTHON[1]}.")
    print(f"Your Python version is {sys.version_info.major}.{sys.version_info.minor}.")
    sys.exit(1)

from pycaret.regression import setup, compare_models, save_model

# ---------- CONFIG ----------
DATA_PATH = "data/historical.csv"      # adapt to your repo
TARGET_COL = "Total_Sales"             # adapt to your repo
ID_COL = "City"                        # optional identifier column to ignore as feature
MODEL_NAME = "nutcracker_demand_model"
# ----------------------------

# 1. Load data
df = pd.read_csv(DATA_PATH, thousands=",")

# 2. Normalize column names (PyCaret/LightGBM prefer no spaces)
df.columns = [c.replace(" ", "_") for c in df.columns]

# 3. If your history has separate ticket columns, compute the target. Otherwise skip.
# Example: if you have 'Single_Tickets' + 'Subscription_Tickets'
ticket_cols = [c for c in df.columns if "Ticket" in c or "Tickets" in c]
if ticket_cols and TARGET_COL not in df.columns:
    df[TARGET_COL] = df[ticket_cols].sum(axis=1)

# 4. Ensure numeric columns are numeric
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
