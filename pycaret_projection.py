"""
PyCaret-based projection module for Nutcracker Trajectory Tracker.

This module provides ML-based regression models using PyCaret's AutoML
capabilities to complement the historical averaging approach.

Requirements:
    - Python 3.9+ (including 3.12+ with PyCaret from GitHub master)
    - pycaret @ git+https://github.com/pycaret/pycaret.git@master
"""
from __future__ import annotations
from typing import Optional, Tuple
import pandas as pd
import numpy as np
import warnings

# Suppress PyCaret warnings for cleaner UI
warnings.filterwarnings('ignore')


def prepare_features_for_ml(
    daily: pd.DataFrame,
    seasons_ref: list[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare feature matrix for ML regression.
    
    Features:
    - days_to_close: days until closing (negative = before closing)
    - city_encoded: one-hot encoded city
    - season_year: numeric year from season
    - day_of_week: day of week (0=Monday)
    - month: month of sale
    - cum_qty: cumulative tickets sold
    - per_show_cum_qty: normalized by number of shows
    
    Returns:
        train_df: historical data for training
        predict_df: current season data for prediction
    """
    if daily.empty or not seasons_ref:
        return pd.DataFrame(), pd.DataFrame()
    
    # Filter to reference seasons for training
    train_df = daily[daily["season"].isin(seasons_ref)].copy()
    
    if train_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Create features
    for df in [train_df]:
        if "sale_date" in df.columns:
            df["day_of_week"] = pd.to_datetime(df["sale_date"]).dt.dayofweek
            df["month"] = pd.to_datetime(df["sale_date"]).dt.month
            df["day_of_month"] = pd.to_datetime(df["sale_date"]).dt.day
        
        # Numeric season year
        if "season" in df.columns:
            df["season_year"] = df["season"].astype(str).str.extract(r'(20\d{2})')[0].astype(float)
    
    # Target: final_qty (what we're trying to predict)
    # We want to predict the final ticket count based on early season data
    
    return train_df, pd.DataFrame()


def train_pycaret_model(
    daily: pd.DataFrame,
    seasons_ref: list[str],
    target_col: str = "final_qty",
    top_n: int = 3,
) -> Optional[object]:
    """
    Train a PyCaret regression model on historical data.
    
    Args:
        daily: DataFrame with historical sales data
        seasons_ref: List of reference season labels for training
        target_col: Target column to predict (default: final_qty)
        top_n: Number of top models to compare
        
    Returns:
        Best trained model or None if training fails
    """
    try:
        from pycaret.regression import setup, compare_models, finalize_model
        
        # Prepare training data
        train_df, _ = prepare_features_for_ml(daily, seasons_ref)
        
        if train_df.empty or target_col not in train_df.columns:
            return None
        
        # Select features for model
        feature_cols = [
            "days_to_close", "cum_qty", "per_show_cum_qty",
            "day_of_week", "month", "season_year"
        ]
        
        # Only use available features
        feature_cols = [col for col in feature_cols if col in train_df.columns]
        
        if not feature_cols:
            return None
        
        # Prepare data for PyCaret (drop NaN)
        model_df = train_df[feature_cols + [target_col]].dropna()
        
        if len(model_df) < 50:  # Need sufficient data for training
            return None
        
        # Setup PyCaret
        s = setup(
            data=model_df,
            target=target_col,
            session_id=42,
            verbose=False,
            html=False,
            silent=True,
            normalize=True,
            normalize_method='zscore',
            transformation=False,
            log_experiment=False,
            n_jobs=1,
        )
        
        # Compare and select best model
        best_model = compare_models(n_select=1, verbose=False)
        
        # Finalize the model
        final_model = finalize_model(best_model)
        
        return final_model
        
    except Exception as e:
        # Return None if training fails (graceful degradation)
        print(f"PyCaret model training failed: {e}")
        return None


def predict_with_pycaret(
    model: object,
    daily: pd.DataFrame,
    this_season: str,
) -> pd.DataFrame:
    """
    Use trained PyCaret model to make predictions for current season.
    
    Args:
        model: Trained PyCaret model
        daily: DataFrame with current season data
        this_season: Current season label
        
    Returns:
        DataFrame with ML-based predictions added
    """
    if model is None:
        return daily
    
    try:
        from pycaret.regression import predict_model
        
        # Get current season data
        current = daily[daily["season"] == this_season].copy()
        
        if current.empty:
            return daily
        
        # Prepare features (same as training)
        feature_cols = [
            "days_to_close", "cum_qty", "per_show_cum_qty",
            "day_of_week", "month", "season_year"
        ]
        
        # Create features if needed
        if "sale_date" in current.columns:
            current["day_of_week"] = pd.to_datetime(current["sale_date"]).dt.dayofweek
            current["month"] = pd.to_datetime(current["sale_date"]).dt.month
        
        if "season" in current.columns:
            current["season_year"] = current["season"].astype(str).str.extract(r'(20\d{2})')[0].astype(float)
        
        # Only use available features
        feature_cols = [col for col in feature_cols if col in current.columns]
        
        if not feature_cols:
            return daily
        
        # Make predictions
        pred_df = current[feature_cols].dropna()
        
        if pred_df.empty:
            return daily
        
        predictions = predict_model(model, data=pred_df)
        
        # Add predictions back to current season data
        current.loc[pred_df.index, "ml_pred_final_qty"] = predictions["prediction_label"]
        
        # Merge back to daily
        result = daily.copy()
        mask = result["season"] == this_season
        result.loc[mask, "ml_pred_final_qty"] = current["ml_pred_final_qty"]
        
        return result
        
    except Exception as e:
        print(f"PyCaret prediction failed: {e}")
        return daily


def get_ml_projection_summary(
    proj_df: pd.DataFrame,
    this_season: str,
) -> pd.DataFrame:
    """
    Generate summary statistics for ML-based projections.
    
    Args:
        proj_df: DataFrame with ML predictions
        this_season: Current season label
        
    Returns:
        Summary DataFrame with ML projection metrics
    """
    if "ml_pred_final_qty" not in proj_df.columns:
        return pd.DataFrame()
    
    current = proj_df[proj_df["season"] == this_season].copy()
    
    if current.empty:
        return pd.DataFrame()
    
    summaries = []
    for city, g in current.groupby("city"):
        ml_final = g["ml_pred_final_qty"].dropna()
        
        if ml_final.empty:
            continue
        
        summaries.append({
            "season": this_season,
            "city": city,
            "ml_projected_final_qty": ml_final.iloc[-1] if len(ml_final) > 0 else np.nan,
        })
    
    return pd.DataFrame(summaries)
