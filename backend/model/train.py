import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv("student_performance.csv")

print("\n--- Data Overview ---")
print(df.describe())
print("\nMissing values per column:")
print(df.isnull().sum())

FEATURES = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
]
TARGET = "Exam_Score"

# ----------------------------
# Correlation check (tells us how much signal exists)
# ----------------------------
print("\n--- Correlation with Exam_Score ---")
print(df[FEATURES + [TARGET]].corr()[TARGET].sort_values(ascending=False))

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Baseline: Ridge Regression (sanity check for linearity)
# ----------------------------
ridge = Ridge()
ridge.fit(X_train, y_train)
ridge_r2 = r2_score(y_test, ridge.predict(X_test))
print(f"\n[Baseline] Ridge Regression R2: {ridge_r2:.3f}")

# ----------------------------
# Model 1: Tuned Random Forest
# ----------------------------
rf_param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [None, 8, 12],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2, 4],
}

rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    rf_param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
)
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_
rf_preds = rf_best.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_preds)
rf_r2 = r2_score(y_test, rf_preds)

print(f"\n[Random Forest] Best params: {rf_grid.best_params_}")
print(f"[Random Forest] MAE: {rf_mae:.2f} | R2: {rf_r2:.3f}")

# ----------------------------
# Model 2: Gradient Boosting
# ----------------------------
gb = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
)
gb.fit(X_train, y_train)
gb_preds = gb.predict(X_test)
gb_mae = mean_absolute_error(y_test, gb_preds)
gb_r2 = r2_score(y_test, gb_preds)

print(f"\n[Gradient Boosting] MAE: {gb_mae:.2f} | R2: {gb_r2:.3f}")

# ----------------------------
# Pick the best model automatically
# ----------------------------
candidates = {
    "random_forest": (rf_best, rf_r2, rf_mae),
    "gradient_boosting": (gb, gb_r2, gb_mae),
}
best_name = max(candidates, key=lambda k: candidates[k][1])
best_model, best_r2, best_mae = candidates[best_name]

print(f"\n=== Best model: {best_name} | R2: {best_r2:.3f} | MAE: {best_mae:.2f} ===")

# ----------------------------
# Cross-validation on the winning model (more reliable than single split)
# ----------------------------
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring="r2")
print(f"5-fold CV R2 (mean +/- std): {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

# ----------------------------
# Feature importance (only for tree models)
# ----------------------------
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=FEATURES)
    print("\n--- Feature Importance ---")
    print(importances.sort_values(ascending=False))

# ----------------------------
# Save the best model
# ----------------------------
joblib.dump(best_model, "student_model.pkl")
print(f"\nstudent_model.pkl saved successfully! (model type: {best_name})")

# venv\Scripts\activate
#uvicorn main:app --reload
