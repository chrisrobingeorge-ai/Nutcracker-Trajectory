# validation_utils.py
# Lightweight helpers for checking PyCaret availability and loading a saved model.
# Reuse these functions in your Streamlit or CLI validation page.

import sys
from typing import Any, Optional, List
from pathlib import Path
import pandas as pd

PYCARET_MIN_PYTHON = (3, 9)
PYCARET_MAX_PYTHON = (3, 12)

def is_python_compatible_with_pycaret() -> bool:
    current = sys.version_info[:2]
    return PYCARET_MIN_PYTHON <= current <= PYCARET_MAX_PYTHON

def get_pycaret_compatibility_message() -> str:
    return (
        f"PyCaret supports Python {PYCARET_MIN_PYTHON[0]}.{PYCARET_MIN_PYTHON[1]} "
        f"through {PYCARET_MAX_PYTHON[0]}.{PYCARET_MAX_PYTHON[1]}. "
        f"Your Python version is {sys.version_info.major}.{sys.version_info.minor}."
    )

def _check_pycaret_available():
    if not is_python_compatible_with_pycaret():
        raise RuntimeError(get_pycaret_compatibility_message())
    try:
        import pycaret  # noqa: F401
    except ImportError:
        raise ImportError(
            "PyCaret is required for Model Validation. Install with:\n"
            "  pip install git+https://github.com/pycaret/pycaret.git@master"
        )

def load_pycaret_model(model_name: str) -> Any:
    """
    Load a saved PyCaret regression model (name without .pkl).
    Raises informative errors if missing or PyCaret absent.
    """
    _check_pycaret_available()
    from pycaret.regression import load_model

    p = Path(f"{model_name}.pkl")
    if not p.exists():
        raise FileNotFoundError(
            f"Could not find '{p}'. Train & save a model using the training script, "
            f"then place the .pkl in the project root or provide the path."
        )
    return load_model(model_name)

def get_pycaret_predictions(model: Any, features_df: pd.DataFrame, id_cols: Optional[List[str]] = None) -> pd.Series:
    """
    Given a loaded PyCaret model and features DataFrame, return a Series
    with a 'PyCaret_Prediction' name aligned to input rows.
    """
    try:
        from pycaret.regression import predict_model
    except Exception as e:
        raise ImportError("Could not import predict_model from pycaret.regression") from e

    data_for_pred = features_df.copy()
    if id_cols:
        data_for_pred = data_for_pred.drop(columns=[c for c in id_cols if c in data_for_pred.columns], errors="ignore")

    pred_results = predict_model(model, data=data_for_pred)
    if "Label" in pred_results.columns:
        pred_series = pred_results["Label"]
    else:
        pred_series = pred_results.iloc[:, -1]

    return pred_series.rename("PyCaret_Prediction")
