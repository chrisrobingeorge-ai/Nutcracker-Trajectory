# MODEL_VALIDATION_GUIDE.md

This document explains how to enable PyCaret-based model validation in Nutcracker-Trajectory.

## 1. Install dependencies (recommended in a fresh venv)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Requirements (important compatibility pins)

The following dependencies are required for PyCaret model validation:

- pandas>=2.0,<2.2
- numpy>=1.21,<1.27
- matplotlib>=3.0,<3.8
- scikit-learn>=1.4,<1.5
- xgboost>=2.0.0
- pycaret @ git+https://github.com/pycaret/pycaret.git@master

**Note**: PyCaret required installing from GitHub to support newer Python versions at the time of integration.

## 3. Train a model

- Update `train_pycaret_model.py` to point to your historical sales CSV and correct column names.
- Run: `python train_pycaret_model.py`
- This will create e.g. `nutcracker_demand_model.pkl`

### Configuration in train_pycaret_model.py

```python
DATA_PATH = "data/historical.csv"      # adapt to your repo
TARGET_COL = "Total_Sales"             # adapt to your repo
ID_COL = "City"                        # optional identifier column to ignore as feature
MODEL_NAME = "nutcracker_demand_model"
```

## 4. Use in app

Load the model and get predictions:

```python
from validation_utils import load_pycaret_model, get_pycaret_predictions

# Load the model
model = load_pycaret_model('nutcracker_demand_model')

# Get predictions (features_df should contain your feature columns)
predictions = get_pycaret_predictions(model, features_df, id_cols=['City', 'Season'])
```

Compare metrics (MAE, RMSE, R²) to your current model.

## 5. Common pitfalls & notes

- **PyCaret is heavy** (many dependencies) — use a dedicated venv.
- Ensure pandas/numpy/scikit-learn versions match the pins to avoid conflicts.
- Training with small datasets can lead PyCaret to prefer simple models; evaluate CV results.
- If you want to avoid adding PyCaret as a permanent dependency, keep the validation feature optional in UI and document how to enable it.

## 6. Troubleshooting

### Import errors
- **Solution**: Re-create venv and re-install with pinned versions.

### Model file not found
- **Solution**: Confirm you saved with `save_model(model, '<name>')` and placed the .pkl in the project root or provided full path.

### Python version incompatibility
- **Solution**: PyCaret supports Python 3.9 through 3.12. Check your version with `python --version`.

### Training failures
- Check that your CSV has the correct column names
- Ensure target column exists or can be computed
- Verify numeric columns are properly formatted
- Review the PyCaret setup output for warnings

## 7. Integration with existing workflow

The `validation_utils.py` module provides compatibility checks:

```python
from validation_utils import is_python_compatible_with_pycaret, get_pycaret_compatibility_message

if not is_python_compatible_with_pycaret():
    print(get_pycaret_compatibility_message())
    # Handle gracefully - don't enable validation feature
```

This allows your app to gracefully degrade when PyCaret is not available or compatible.
