"""
Per-bridge cross-chain cost prediction model training pipeline.

Trains XGBoost and LightGBM on cleaned per-bridge data, picks the best,
and serializes models + metadata for the prediction API.

CCIP (11 samples): Median-based fallback only.
"""

import json
import logging
import warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import joblib

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "cleaned_split_data"
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

BRIDGES = {
    "across": "across_cleaned.csv",
    "cctp": "cctp_cleaned.csv",
    "stargate_oft": "stargate_oft_cleaned.csv",
    "stargate_bus": "stargate_bus_cleaned.csv",
}

TARGET = "user_cost"

NUMERIC_FEATURES = [
    "amount_usd",
    "dune_hourly_gas_gwei",
    "gas_1h_lag",
    "gas_6h_avg",
    "gas_24h_avg",
    "gas_volatility_24h",
    "eth_price_at_src",
    "eth_price_change_1h",
    "eth_price_24h_avg",
    "bridge_hourly_volume",
]

TEMPORAL_FEATURES = ["hour_of_day", "day_of_week", "is_weekend", "month"]
CATEGORICAL_FEATURES = ["route", "src_symbol"]


def load_and_prepare(bridge_name: str, filename: str):
    """Load cleaned data, engineer features, encode categoricals."""
    df = pd.read_csv(DATA_DIR / filename)

    time_col = next((c for c in ("datetime", "src_timestamp") if c in df.columns), None)
    if time_col:
        df = df.sort_values(time_col).reset_index(drop=True)

    df = df[df[TARGET].notna() & (df[TARGET] > 0)].copy()

    # Remove extreme outliers (beyond 99th percentile) — keeps model focused
    cap = df[TARGET].quantile(0.99)
    n_before = len(df)
    df = df[df[TARGET] <= cap].copy()
    if n_before - len(df) > 0:
        log.info(f"  Removed {n_before - len(df)} outliers (>{cap:.2f} USD)")

    features = [f for f in NUMERIC_FEATURES + TEMPORAL_FEATURES if f in df.columns]

    for f in features:
        if df[f].isnull().any():
            df[f] = df[f].fillna(df[f].median())

    # Engineered features that improve tree splits
    if "amount_usd" in df.columns:
        df["log_amount"] = np.log1p(df["amount_usd"])
        features.append("log_amount")

    if "amount_usd" in df.columns and "dune_hourly_gas_gwei" in df.columns:
        df["amount_x_gas"] = df["amount_usd"] * df["dune_hourly_gas_gwei"]
        features.append("amount_x_gas")

    if "hour_of_day" in df.columns:
        hour_rad = 2 * np.pi * df["hour_of_day"] / 24
        df["hour_sin"] = np.sin(hour_rad)
        df["hour_cos"] = np.cos(hour_rad)
        features.extend(["hour_sin", "hour_cos"])

    encoders = {}
    for cat in CATEGORICAL_FEATURES:
        if cat in df.columns:
            le = LabelEncoder()
            col_name = f"{cat}_enc"
            df[col_name] = le.fit_transform(df[cat].fillna("unknown").astype(str))
            encoders[cat] = le
            features.append(col_name)

    enc_cols = {f"{c}_enc" for c in CATEGORICAL_FEATURES}
    feature_medians = {f: float(df[f].median()) for f in features if f not in enc_cols}

    X = df[features].values
    y = np.log1p(df[TARGET].values)

    return X, y, features, encoders, feature_medians, df


def train_xgb(X_train, y_train, X_test, y_test):
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    return model


def train_lgb(X_train, y_train, X_test, y_test):
    if not HAS_LGB:
        return None
    model = lgb.LGBMRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
    )
    return model


def evaluate(model, X_test, y_test):
    """Evaluate in original USD scale."""
    y_pred_log = model.predict(X_test)
    y_pred = np.maximum(np.expm1(y_pred_log), 0)
    y_true = np.expm1(y_test)

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def confidence_level(r2: float, n_samples: int) -> str:
    if n_samples < 50:
        return "low"
    if r2 >= 0.7:
        return "high"
    if r2 >= 0.4:
        return "medium"
    return "low"


def train_bridge(bridge_name: str, filename: str) -> dict:
    log.info(f"Training {bridge_name}...")
    X, y, features, encoders, feature_medians, df = load_and_prepare(bridge_name, filename)
    n = len(X)

    split_idx = int(n * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    log.info(f"  {n} samples, {len(features)} features, train={len(X_train)}, test={len(X_test)}")

    xgb_model = train_xgb(X_train, y_train, X_test, y_test)
    xgb_metrics = evaluate(xgb_model, X_test, y_test)
    log.info(f"  XGBoost:  MAE=${xgb_metrics['mae']:.4f}  R²={xgb_metrics['r2']:.4f}")

    best_model, best_metrics, best_type = xgb_model, xgb_metrics, "xgboost"

    lgb_model = train_lgb(X_train, y_train, X_test, y_test)
    if lgb_model is not None:
        lgb_metrics = evaluate(lgb_model, X_test, y_test)
        log.info(f"  LightGBM: MAE=${lgb_metrics['mae']:.4f}  R²={lgb_metrics['r2']:.4f}")
        if lgb_metrics["r2"] > xgb_metrics["r2"]:
            best_model, best_metrics, best_type = lgb_model, lgb_metrics, "lightgbm"

    conf = confidence_level(best_metrics["r2"], n)
    log.info(f"  Winner: {best_type} (R²={best_metrics['r2']:.4f}, confidence={conf})")

    joblib.dump(best_model, MODEL_DIR / f"{bridge_name}_model.joblib")
    joblib.dump(encoders, MODEL_DIR / f"{bridge_name}_encoders.joblib")

    meta = {
        "bridge": bridge_name,
        "model_type": best_type,
        "features": features,
        "feature_medians": feature_medians,
        "metrics": best_metrics,
        "confidence": conf,
        "n_samples": n,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "last_trained": datetime.now(timezone.utc).isoformat(),
        "feature_importance": dict(
            zip(features, best_model.feature_importances_.tolist())
        ),
    }
    with open(MODEL_DIR / f"{bridge_name}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def train_ccip_fallback() -> dict:
    """CCIP has too few samples — use median cost bucketed by transfer size."""
    log.info("Training CCIP (median fallback, 11 samples)...")
    df = pd.read_csv(DATA_DIR / "ccip_cleaned.csv")
    df = df[df[TARGET].notna() & (df[TARGET] >= 0)]

    buckets = [
        (0, 100), (100, 1_000), (1_000, 10_000),
        (10_000, 100_000), (100_000, float("inf")),
    ]
    cost_by_bucket = {}
    for lo, hi in buckets:
        subset = df[(df["amount_usd"] >= lo) & (df["amount_usd"] < hi)]
        if len(subset) > 0:
            cost_by_bucket[f"{lo}-{hi}"] = {
                "median": float(subset[TARGET].median()),
                "count": int(len(subset)),
            }

    meta = {
        "bridge": "ccip",
        "model_type": "median_fallback",
        "n_samples": len(df),
        "metrics": {
            "median_cost": float(df[TARGET].median()),
            "mean_cost": float(df[TARGET].mean()),
            "std_cost": float(df[TARGET].std()),
        },
        "confidence": "low",
        "last_trained": datetime.now(timezone.utc).isoformat(),
        "cost_by_amount_bucket": cost_by_bucket,
    }
    with open(MODEL_DIR / "ccip_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"  CCIP median cost: ${meta['metrics']['median_cost']:.4f} ({len(df)} samples)")
    return meta


def main():
    log.info("=" * 60)
    log.info("Cross-Chain Bridge Cost Prediction — Model Training")
    log.info("=" * 60)

    results = {}
    for bridge_name, filename in BRIDGES.items():
        try:
            results[bridge_name] = train_bridge(bridge_name, filename)
        except Exception as e:
            log.error(f"Failed training {bridge_name}: {e}")
            import traceback
            traceback.print_exc()

    try:
        results["ccip"] = train_ccip_fallback()
    except Exception as e:
        log.error(f"CCIP fallback failed: {e}")

    log.info("")
    log.info("=" * 60)
    log.info("Training Summary")
    log.info("=" * 60)
    for name, meta in results.items():
        if meta.get("model_type") == "median_fallback":
            log.info(f"  {name:15s}  median=${meta['metrics']['median_cost']:.4f}  n={meta['n_samples']}")
        else:
            m = meta["metrics"]
            log.info(
                f"  {name:15s}  {meta['model_type']:10s}  "
                f"R²={m['r2']:.4f}  MAE=${m['mae']:.4f}  "
                f"conf={meta['confidence']:6s}  n={meta['n_samples']}"
            )


if __name__ == "__main__":
    main()
