```markdown
# MODEL_VALIDATION_GUIDE.md

This document explains how to enable PyCaret-based model validation in Nutcracker-Trajectory.

1) Install dependencies (recommended in a fresh venv):
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

2) Requirements (important compatibility pins)
   - pandas>=2.0,<2.2
   - numpy>=1.21,<1.27
   - matplotlib>=3.0,<3.8
   - scikit-learn>=1.4,<1.5
   - xgboost>=2.0.0
   - pycaret @ git+https://github.com/pycaret/pycaret.git@master

   Note: PyCaret required installing from GitHub to support newer Python versions at the time of integration.

3) Train a model:
   - Update train_pycaret_model.py to point to your historical sales CSV and correct column names.
   - Run: python train_pycaret_model.py
   - This will create e.g. `nutcracker_demand_model.pkl`

4) Use in app:
   - Load the model via load_pycaret_model('nutcracker_demand_model')
   - Use get_pycaret_predictions(model, features_df, id_cols=[...]) to obtain aligned predictions.
   - Compare metrics (MAE, RMSE, R²) to your current model.

5) Common pitfalls & notes:
   - PyCaret is heavy (many dependencies) — use a dedicated venv.
   - Ensure pandas/numpy/scikit-learn versions match the pins to avoid conflicts.
   - Training with small datasets can lead PyCaret to prefer simple models; evaluate CV results.
   - If you want to avoid adding PyCaret as a permanent dependency, keep the validation feature optional in UI and document how to enable it.

6) Troubleshooting:
   - If import errors occur: re-create venv and re-install with pinned versions.
   - If model file not found: confirm you saved with save_model(model, '<name>') and placed the .pkl in the project root or provided full path.
```
