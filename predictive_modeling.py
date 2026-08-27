"""
Week 4 - Predictive Modeling and Optimization in Logistics Systems
Forecasts delivery_minutes and proposes an optimization strategy
based on model insights.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style("whitegrid")
np.random.seed(42)

df = pd.read_csv("simulated_logistics_dataset.csv")

# ---------------- Problem Definition ----------------
# Target: delivery_minutes (continuous, regression problem)
# Features: distance_km, traffic_level, package_weight_kg, order_hour, delivery_zone
target = "delivery_minutes"
num_features = ["distance_km", "package_weight_kg", "order_hour"]
cat_features = ["traffic_level", "delivery_zone"]

X = df[num_features + cat_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
], remainder="passthrough")

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42),
}

results = []
predictions_by_model = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    predictions_by_model[name] = preds

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    cv_scores = cross_val_score(pipe, X, y, cv=kf, scoring="neg_mean_absolute_error")
    cv_mae = -cv_scores.mean()

    results.append({
        "Model": name, "MAE": round(mae, 2), "RMSE": round(rmse, 2),
        "R2": round(r2, 3), "CV_MAE": round(cv_mae, 2)
    })

results_df = pd.DataFrame(results).sort_values("MAE")
print(results_df)
results_df.to_csv("model_results.csv", index=False)

best_model_name = results_df.iloc[0]["Model"]
print(f"\nBest model: {best_model_name}")

# ---------------- Hyperparameter tuning (Random Forest) ----------------
from sklearn.model_selection import GridSearchCV

rf_pipe = Pipeline([("prep", preprocess), ("model", RandomForestRegressor(random_state=42))])
param_grid = {
    "model__n_estimators": [100, 300],
    "model__max_depth": [6, 10, None],
}
grid = GridSearchCV(rf_pipe, param_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
grid.fit(X_train, y_train)
print("Best RF params:", grid.best_params_)
best_rf_preds = grid.predict(X_test)
best_rf_mae = mean_absolute_error(y_test, best_rf_preds)
print(f"Tuned RF MAE: {best_rf_mae:.2f}")

# ---------------- Feature importance (best tree-based model) ----------------
rf_final = grid.best_estimator_.named_steps["model"]
feature_names = (
    list(grid.best_estimator_.named_steps["prep"]
         .named_transformers_["cat"].get_feature_names_out(cat_features))
    + num_features
)
importances = pd.Series(rf_final.feature_importances_, index=feature_names).sort_values(ascending=False)
print(importances)
importances.to_csv("feature_importances.csv")

# ---------------- Visualizations ----------------
plt.figure(figsize=(7, 4.5))
sns.barplot(x="MAE", y="Model", data=results_df, color="#D9662B")
plt.title("Model Comparison — Mean Absolute Error (lower is better)")
plt.xlabel("MAE (minutes)")
plt.tight_layout()
plt.savefig("chart_model_comparison.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 4.5))
plt.scatter(y_test, predictions_by_model[best_model_name], alpha=0.4, color="#D9662B", s=20)
lims = [min(y_test.min(), predictions_by_model[best_model_name].min()),
        max(y_test.max(), predictions_by_model[best_model_name].max())]
plt.plot(lims, lims, "--", color="gray")
plt.xlabel("Actual Delivery Time (min)")
plt.ylabel("Predicted Delivery Time (min)")
plt.title(f"Actual vs Predicted — {best_model_name}")
plt.tight_layout()
plt.savefig("chart_actual_vs_predicted.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 5))
importances.head(8).sort_values().plot(kind="barh", color="#D9662B")
plt.title("Top Feature Importances (Tuned Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("chart_feature_importance.png", dpi=150)
plt.close()

# ---------------- Optimization: resource allocation across zones ----------------
# Use model insight (distance & traffic are dominant) to allocate a fixed
# fleet of vehicles across zones proportional to predicted total workload,
# maximizing coverage of orders at risk of breaching the 60-min promise.
zone_stats = df.groupby("delivery_zone").agg(
    orders=("order_id", "count"),
    avg_predicted_time=("delivery_minutes", "mean"),
    late_risk_orders=("delivery_minutes", lambda x: (x > 60).sum()),
).reset_index()
zone_stats["late_risk_share"] = zone_stats["late_risk_orders"] / zone_stats["orders"]

TOTAL_VEHICLES = 20
zone_stats["workload_score"] = zone_stats["orders"] * zone_stats["late_risk_share"]
zone_stats["allocated_vehicles"] = np.floor(
    TOTAL_VEHICLES * zone_stats["workload_score"] / zone_stats["workload_score"].sum()
).astype(int)
# distribute any remainder to the highest workload zone
remainder = TOTAL_VEHICLES - zone_stats["allocated_vehicles"].sum()
zone_stats.loc[zone_stats["workload_score"].idxmax(), "allocated_vehicles"] += remainder

print(zone_stats)
zone_stats.to_csv("optimized_vehicle_allocation.csv", index=False)

plt.figure(figsize=(7, 4.5))
sns.barplot(x="delivery_zone", y="allocated_vehicles", data=zone_stats.sort_values("allocated_vehicles"), color="#4C9A6E")
plt.title("Optimized Vehicle Allocation by Zone (Fleet of 20)")
plt.xlabel("Delivery Zone")
plt.ylabel("Vehicles Allocated")
plt.tight_layout()
plt.savefig("chart_vehicle_allocation.png", dpi=150)
plt.close()

print("\nAll modeling and optimization artifacts saved.")
