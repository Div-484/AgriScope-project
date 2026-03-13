"""
Train ML models and save to models/ directory.
Run from AgriScope/ root: python train_model.py
Season names: Monsoon (was Kharif), Winter (was Rabi), Summer
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor, AdaBoostRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    USE_XGB = True
except ImportError:
    USE_XGB = False
    print("[WARN] XGBoost not available; skipping.")

try:
    from lightgbm import LGBMRegressor
    USE_LGB = True
except ImportError:
    USE_LGB = False
    print("[WARN] LightGBM not available; skipping.")

from utils.data_cleaning import clean_data

# ── Paths ──────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
RAW_PATH  = os.path.join(BASE, "data", "final_data.csv")
CLEAN_PATH= os.path.join(BASE, "cleaned_data", "cleaned_data.csv")
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Clean data ─────────────────────────────────────────────────────
print("[1/7] Cleaning data …")
df = clean_data(RAW_PATH, CLEAN_PATH)

# ── Reload from CSV ────────────────────────────────────────────────
print("[2/7] Loading & preprocessing cleaned data …")
df = pd.read_csv(CLEAN_PATH)

# Encode crop_type if present (crucial feature!)
if "crop_type" in df.columns and "crop_type_encoded" not in df.columns:
    le_crop = LabelEncoder()
    df["crop_type_encoded"] = le_crop.fit_transform(df["crop_type"].astype(str))
    print(f"    Encoded {df['crop_type'].nunique()} crop types as feature")
else:
    le_crop = None

# Build feature list - include crop_type_encoded as it's the most important feature
FEATURES = [
    'district_encoded', 'season_encoded',
    'crop_type_encoded',                  # KEY: different crops have very different yields
    'total_rainfall', 'rainy_days',
    'average_tmax', 'average_tmin', 'average_humidity'
]
FEATURES = [f for f in FEATURES if f in df.columns]
TARGET   = 'yield'

print(f"    Features used ({len(FEATURES)}): {FEATURES}")
df_model = df[FEATURES + [TARGET]].dropna()
df_model = df_model[df_model[TARGET] > 0]

X = df_model[FEATURES]
y = df_model[TARGET]
print(f"    Dataset: {X.shape[0]} samples")

# Log-transform yield for better model performance (reduce skewness/outlier impact)
y_log = np.log1p(y)
print(f"    Yield: mean={y.mean():.1f}, std={y.std():.1f}, skew={y.skew():.3f}")
print(f"    Log-yield: mean={y_log.mean():.3f}, std={y_log.std():.3f}, skew={y_log.skew():.3f}")

# ── Split ──────────────────────────────────────────────────────────
print("[3/7] Splitting dataset 80/20 …")
X_train, X_test, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)
y_train_orig = np.expm1(y_train_log)
y_test_orig  = np.expm1(y_test_log)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Define all models ──────────────────────────────────────────────
print("[4/7] Training models …")
models = {
    "RandomForest":       RandomForestRegressor(n_estimators=300, max_depth=12,
                                                min_samples_split=4, random_state=42, n_jobs=-1),
    "GradientBoosting":   GradientBoostingRegressor(n_estimators=300, learning_rate=0.08,
                                                    max_depth=5, random_state=42),
    "ExtraTrees":         ExtraTreesRegressor(n_estimators=300, max_depth=12,
                                              random_state=42, n_jobs=-1),
    "DecisionTree":       DecisionTreeRegressor(max_depth=12, min_samples_split=4,
                                                random_state=42),
    "Ridge":              Ridge(alpha=1.0),
    "ElasticNet":         ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=5000),
    "KNeighbors":         KNeighborsRegressor(n_neighbors=7, weights="distance"),
}
if USE_XGB:
    models["XGBoost"] = XGBRegressor(
        n_estimators=300, learning_rate=0.08, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="rmse", verbosity=0
    )
if USE_LGB:
    models["LightGBM"] = LGBMRegressor(
        n_estimators=300, learning_rate=0.08, max_depth=6,
        random_state=42, verbose=-1
    )

# ── Train & evaluate (predict in original space after inverse log) ──
results  = {}   # r2 in original space
metrics  = {}
fitted   = {}
print(f"    {'Model':<22} {'R²':>8} {'Accuracy%':>11} {'MAE':>10} {'RMSE':>10}")
print("    " + "-"*65)

for name, mdl in models.items():
    mdl.fit(X_train_sc, y_train_log)
    preds_log   = mdl.predict(X_test_sc)
    preds_orig  = np.expm1(preds_log)          # inverse log transform

    mae   = mean_absolute_error(y_test_orig, preds_orig)
    rmse  = np.sqrt(mean_squared_error(y_test_orig, preds_orig))
    r2    = r2_score(y_test_orig, preds_orig)

    # R² can be negative; clamp accuracy to [0, 100]
    accuracy_pct = max(0.0, min(100.0, r2 * 100))

    results[name] = r2
    fitted[name]  = mdl
    metrics[name] = {
        "MAE":      round(float(mae), 2),
        "RMSE":     round(float(rmse), 2),
        "R2":       round(float(r2), 4),
        "Accuracy": round(accuracy_pct, 2),
    }
    print(f"    {name:<22} {r2:>8.4f} {accuracy_pct:>10.2f}% {mae:>10.1f} {rmse:>10.1f}")

# ── Rank by R² and pick best ───────────────────────────────────────
print("[5/7] Ranking models …")
ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
print("    Ranking:")
for i, (name, r2) in enumerate(ranked, 1):
    crown = " ← BEST" if i == 1 else ""
    print(f"    #{i} {name}: R²={r2:.4f}{crown}")

best_name  = ranked[0][0]
best_model = fitted[best_name]
top5 = [name for name, _ in ranked[:5]]

# ── Save ───────────────────────────────────────────────────────────
print("[6/7] Saving models & artefacts …")
joblib.dump(best_model, os.path.join(MODEL_DIR, "model.pkl"))
joblib.dump(scaler,     os.path.join(MODEL_DIR, "scaler.pkl"))

# Save every model individually
for name, mdl in fitted.items():
    safe_name = name.lower().replace(" ", "_")
    joblib.dump(mdl, os.path.join(MODEL_DIR, f"{safe_name}_model.pkl"))

# Encoders
encoders = {}
for col in ["district", "season", "crop_type"]:
    if col in df.columns:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[col] = le
joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))

# Save log-transform flag + crop-type encoder
joblib.dump({"log_transform": True, "le_crop": le_crop},
            os.path.join(MODEL_DIR, "transform_info.pkl"))

# Save metrics JSON
metrics_out = {
    "best_model":   best_name,
    "top5":         top5,
    "models":       metrics,
    "features":     FEATURES,
    "log_transform": True,
    "dataset_size": int(X.shape[0]),
    "test_size":    int(X_test.shape[0]),
}
with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(metrics_out, f, indent=2)

print(f"    Best model : {best_name}  (R²={results[best_name]:.4f}, Accuracy={metrics[best_name]['Accuracy']:.2f}%)")
print(f"    Metrics saved → models/metrics.json")

# ── Summary ────────────────────────────────────────────────────────
print("\n[7/7] Summary")
print(f"    Best Model   : {best_name}")
print(f"    Accuracy (R²×100): {metrics[best_name]['Accuracy']:.2f}%")
print(f"    R² Score     : {results[best_name]:.4f}")
print(f"    MAE          : {metrics[best_name]['MAE']:,.1f} kg/ha")
print(f"    RMSE         : {metrics[best_name]['RMSE']:,.1f} kg/ha")
print("\nDone! Models saved to models/")
