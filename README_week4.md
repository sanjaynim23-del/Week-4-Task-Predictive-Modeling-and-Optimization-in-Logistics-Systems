# Week 4 — Predictive Modeling and Optimization in Logistics Systems

Data Analytics Internship | Sanjay | BMS College of Engineering

## Overview
Builds a predictive model to forecast `delivery_minutes` on the Week 3 simulated
logistics dataset, compares four regression approaches, tunes the best-performing
tree model, and uses the model's insights to optimize vehicle allocation across
delivery zones.

## Script (`predictive_modeling.py`)
- Trains and compares Linear Regression, Decision Tree, Random Forest, and
  Gradient Boosting on an 80/20 train/test split with one-hot encoded categoricals
- Evaluates with MAE, RMSE, R², and 5-fold cross-validation
- Runs a GridSearchCV hyperparameter sweep on Random Forest
- Extracts feature importances from the tuned model
- Produces 3 charts: model comparison, actual vs. predicted, feature importance
- Runs a workload-weighted vehicle allocation optimization across the 5 zones
  for a fixed 20-vehicle fleet, based on each zone's late-delivery risk

Run it:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
python predictive_modeling.py
```

Requires `simulated_logistics_dataset.csv` from the Week 3 folder in the same
directory. Outputs `model_results.csv`, `feature_importances.csv`,
`optimized_vehicle_allocation.csv`, and 3 chart PNGs.

**Result:** Gradient Boosting was the best model (MAE 7.00 min, R² = 0.853).
Distance was the dominant feature (~70% importance), followed by traffic level.

Full write-up, all visualizations, and optimization recommendations are in
`Week4_Predictive_Modeling_Optimization_Report.docx` (submitted separately on the
Yuva Intern portal).

## Status
Weeks 1–4: Complete. Week 5: In progress.
