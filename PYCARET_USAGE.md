# PyCaret Integration Usage Guide

## Overview

The Nutcracker Trajectory Tracker now supports **two projection methods**:

1. **Historical Curve-Based** (default) - Uses mean cumulative share curves from reference seasons
2. **ML-Based with PyCaret** (optional) - Trains AutoML regression models on historical data

## Prerequisites

### Python Version
PyCaret requires **Python 3.9, 3.10, or 3.11**. It does not support Python 3.12 yet.

Check your Python version:
```bash
python --version
```

### Installation

Install PyCaret:
```bash
pip install pycaret>=3.3.2
```

Or install all requirements including PyCaret:
```bash
pip install -r requirements.txt
```

## How to Use

### 1. Enable ML Projections

In the Streamlit app sidebar:
1. Check the box: **"Enable ML-based projections (PyCaret)"**
2. The app will train a model when you load data (this takes 30-60 seconds)

### 2. What Happens During Training

The app will:
1. Extract features from historical seasons (days_to_close, cumulative tickets, seasonality, etc.)
2. Train multiple regression models (Linear, Ridge, Random Forest, XGBoost, etc.)
3. Automatically select the best-performing model
4. Use it to generate predictions for the current season

### 3. View Results

The app displays three projection views:

#### A. Charts
- Historical trajectory lines (reference seasons)
- Current season actual sales (solid navy line)
- Curve-based projection (dashed navy line)

#### B. Projection Summary
- **Historical Curve-Based Projections**: Shows projected final tickets using the traditional method
- **ML-Based Projections (PyCaret)**: Shows ML model predictions
- **Comparison**: Side-by-side comparison showing the difference between methods

#### C. Daily Projection Table
- Detailed day-by-day breakdown with both projection types

## Understanding the ML Features

The PyCaret model uses these features to make predictions:

| Feature | Description | Example |
|---------|-------------|---------|
| `days_to_close` | Days until season closing | -180, -90, -30, 0 |
| `cum_qty` | Cumulative tickets sold | 1000, 5000, 15000 |
| `per_show_cum_qty` | Tickets normalized by shows | 100, 500, 1500 |
| `day_of_week` | Day of week (0=Mon, 6=Sun) | 0, 1, 2, 3, 4, 5, 6 |
| `month` | Month of sale | 1-12 |
| `season_year` | Numeric year | 2021, 2022, 2023 |

## Advantages of ML-Based Projections

✅ **Captures non-linear patterns**: ML models can detect complex relationships in sales data

✅ **Automatic feature engineering**: PyCaret handles normalization and preprocessing

✅ **Multiple algorithms tested**: Compares various models and picks the best

✅ **Seasonality aware**: Incorporates day-of-week and monthly patterns

✅ **Handles multiple cities**: Can learn city-specific patterns

## Limitations

⚠️ **Training time**: Takes 30-60 seconds to train (vs instant for curve-based)

⚠️ **Python version**: Only works with Python 3.9-3.11

⚠️ **Data requirements**: Needs sufficient historical data (recommended: 2+ seasons)

⚠️ **Interpretability**: ML models are less transparent than simple averages

## Troubleshooting

### "PyCaret requires Python 3.9-3.11"

**Solution**: Use a compatible Python version. You can:
- Use `pyenv` to install Python 3.11: `pyenv install 3.11 && pyenv local 3.11`
- Create a virtual environment with Python 3.11
- Use Docker with Python 3.11 base image

### "Model training skipped (insufficient data)"

**Solution**: Ensure you have:
- At least 2 reference seasons selected
- Sufficient daily sales records (recommended: 50+ days per season)
- Required columns: `days_to_close`, `cum_qty`, `final_qty`

### "ML model training failed"

**Possible causes**:
- Missing data in key columns
- All features are NaN
- Data format issues

**Solution**: 
- Check that your CSV files have all required columns
- Verify dates are properly formatted (YYYY-MM-DD)
- Ensure numeric columns contain valid numbers

## Comparing Both Methods

### When to Trust Curve-Based Projections:
- ✅ Stable, predictable sales patterns
- ✅ Similar to historical averages
- ✅ Want simple, interpretable results

### When to Trust ML-Based Projections:
- ✅ Unusual sales patterns this year
- ✅ Strong seasonality or day-of-week effects
- ✅ Want to explore alternative scenarios

### Best Practice:
**Use both methods** and compare them. Large differences may indicate:
- Unusual market conditions
- Data quality issues
- Model overfitting/underfitting

## Technical Details

### Model Selection Process

PyCaret automatically:
1. Splits data into train/test sets
2. Trains 10+ regression models
3. Evaluates using cross-validation
4. Ranks by R², RMSE, MAE
5. Returns the best model

Common models PyCaret tests:
- Linear Regression
- Ridge/Lasso Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- Support Vector Machines

### Data Preprocessing

PyCaret handles:
- Z-score normalization
- Missing value imputation
- Feature scaling
- Train/test splitting

## Example Workflow

```python
# 1. Load historical and current season data
# 2. Select reference seasons (e.g., 2021, 2022, 2023)
# 3. Enable "ML-based projections (PyCaret)"
# 4. Wait for training to complete (~30-60 sec)
# 5. Review both projection methods
# 6. Compare curve-based vs ML-based
# 7. Use insights to make decisions
```

## Getting Help

If you encounter issues:
1. Check the Streamlit app logs for error messages
2. Verify Python version: `python --version`
3. Verify PyCaret installation: `pip show pycaret`
4. Check data format matches requirements in README.md

## Future Enhancements

Possible future improvements:
- Save/load trained models
- Custom model selection
- Feature importance visualization
- Confidence intervals
- Ensemble methods combining both approaches
