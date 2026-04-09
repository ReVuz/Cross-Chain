"""
Train Across bridge cost model from across_final_data.csv (358K rows).

Computes user_cost from raw input/output amounts, engineers features,
trains XGBoost + LightGBM, picks the best, and saves artifacts compatible
with the existing predictor.py pipeline.
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
DATA_PATH = BASE_DIR / "across_final_data.csv"
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

CONTRACT_TO_SYMBOL = {
    "0x4200000000000000000000000000000000000006": "WETH",
    "0x82af49447d8a07e3bd95bd0d56f35241523fbab1": "WETH",
    "0x7ceb23fd6bc0add59e62ac25578270cff1b9f619": "WETH",
    "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913": "USDC",
    "0xaf88d065e77c8cc2239327c5edb3a432268e5831": "USDC",
    "0x0b2c639c533813f4aa9d7837caf62653d097ff85": "USDC",
    "0x3c499c542cef5e3811e1192ce70d8cc03d5c3359": "USDC",
    "0x94b008aa00579c1307b0ef2c499ad98a8ce58e58": "USDT",
    "0xc2132d05d31c914a87c6611c10748aeb04b58e8f": "USDT",
    "0x2f2a2543b76a4166549f7aab2e75bef0aefc5b0f": "WBTC",
    "0x68f180fcce6836688e9084f035309e29bf0a2095": "WBTC",
    "0x1bfd67037b42cf73acf2047067bd4f2c47d9bfd6": "WBTC",
    "0xda10009cbd5d07dd0cecc66161fc93d7c9000da1": "DAI",
    "0x50c5725949a6f0c72e6c4a641f24049a917db0cb": "DAI",
    "0x8f3cf7ad23cd3cadbd9735aff958023239c6a063": "DAI",
    "0xd652c5425aea2afd5fb142e120fecf79e18fafc3": "USDbC",
    "0x395ae52bb17aef68c2888d941736a71dc6d4e125": "POOL",
    "0xfe8b128ba8c78aabc59d4c64cee7ff28e9379921": "BALD",
    "0x4158734d47fc9692176b5085e0f52ee0da5d47f1": "OTHER",
    "0xcf934e2402a5e072928a39a956964eb8f2b5b79c": "OTHER",
    "0x040d1edc9569d4bab2d15287dc5a4f10f56a56b8": "OTHER",
    "0x25788a1a171ec66da6502f9975a15b609ff54cf6": "OTHER",
    "0x9a71012b13ca4d3d0cdc72a177df3ef03b0e76a3": "BAL",
    "0xff733b2a3557a7ed6697007ab5d11b79fdd1b76b": "OTHER",
}

TARGET = "user_cost"


def load_and_prepare():
    log.info(f"Loading {DATA_PATH.name} ...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    log.info(f"  Raw rows: {len(df):,}")

    for col in ("input_amount_usd", "output_amount_usd", "src_fee_usd"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["input_amount_usd", "output_amount_usd"]).copy()

    df[TARGET] = (df["input_amount_usd"] - df["output_amount_usd"]) + df["src_fee_usd"].fillna(0)

    df = df[(df[TARGET] > 0) & (df["input_amount_usd"] > 0)].copy()
    log.info(f"  After user_cost > 0 filter: {len(df):,}")

    df.rename(columns={"input_amount_usd": "amount_usd"}, inplace=True)

    df["src_symbol"] = (
        df["src_contract_address"]
        .str.lower()
        .map(CONTRACT_TO_SYMBOL)
        .fillna("OTHER")
    )

    df["route"] = df["src_blockchain"].str.lower() + "\u2192" + df["dst_blockchain"].str.lower()

    df["src_timestamp"] = pd.to_numeric(df["src_timestamp"], errors="coerce")
    df = df.dropna(subset=["src_timestamp"]).copy()
    df["datetime"] = pd.to_datetime(df["src_timestamp"], unit="s", utc=True)
    df = df.sort_values("src_timestamp").reset_index(drop=True)

    df["hour_of_day"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["datetime"].dt.month

    df["hour_bucket"] = df["datetime"].dt.floor("h")
    vol = df.groupby("hour_bucket").size().rename("bridge_hourly_volume")
    df = df.join(vol, on="hour_bucket")

    cap = df[TARGET].quantile(0.99)
    n_before = len(df)
    df = df[df[TARGET] <= cap].copy()
    log.info(f"  Removed {n_before - len(df):,} outliers (>{cap:.2f} USD)")
    log.info(f"  Final rows: {len(df):,}")

    # --- Feature engineering ---
    features = [
        "amount_usd", "hour_of_day", "day_of_week", "is_weekend", "month",
        "bridge_hourly_volume",
    ]

    df["log_amount"] = np.log1p(df["amount_usd"])
    features.append("log_amount")

    hour_rad = 2 * np.pi * df["hour_of_day"] / 24
    df["hour_sin"] = np.sin(hour_rad)
    df["hour_cos"] = np.cos(hour_rad)
    features.extend(["hour_sin", "hour_cos"])

    for f in features:
        if df[f].isnull().any():
            df[f] = df[f].fillna(df[f].median())

    # --- Categorical encoding ---
    encoders = {}
    for cat in ("route", "src_symbol"):
        le = LabelEncoder()
        col_name = f"{cat}_enc"
        df[col_name] = le.fit_transform(df[cat].astype(str))
        encoders[cat] = le
        features.append(col_name)

    enc_cols = {"route_enc", "src_symbol_enc"}
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
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    return model


def evaluate(model, X_test, y_test):
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


def main():
    log.info("=" * 60)
    log.info("Across Model Training (from across_final_data.csv)")
    log.info("=" * 60)

    X, y, features, encoders, feature_medians, df = load_and_prepare()
    n = len(X)

    split_idx = int(n * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    log.info(f"  {n:,} samples, {len(features)} features, train={len(X_train):,}, test={len(X_test):,}")

    xgb_model = train_xgb(X_train, y_train, X_test, y_test)
    xgb_metrics = evaluate(xgb_model, X_test, y_test)
    log.info(f"  XGBoost:  MAE=${xgb_metrics['mae']:.4f}  RMSE=${xgb_metrics['rmse']:.4f}  R²={xgb_metrics['r2']:.4f}")

    best_model, best_metrics, best_type = xgb_model, xgb_metrics, "xgboost"

    lgb_model = train_lgb(X_train, y_train, X_test, y_test)
    if lgb_model is not None:
        lgb_metrics = evaluate(lgb_model, X_test, y_test)
        log.info(f"  LightGBM: MAE=${lgb_metrics['mae']:.4f}  RMSE=${lgb_metrics['rmse']:.4f}  R²={lgb_metrics['r2']:.4f}")
        if lgb_metrics["r2"] > xgb_metrics["r2"]:
            best_model, best_metrics, best_type = lgb_model, lgb_metrics, "lightgbm"

    conf = confidence_level(best_metrics["r2"], n)
    log.info(f"  Winner: {best_type} (R²={best_metrics['r2']:.4f}, confidence={conf})")

    # --- Save artifacts ---
    joblib.dump(best_model, MODEL_DIR / "across_model.joblib")
    joblib.dump(encoders, MODEL_DIR / "across_encoders.joblib")

    meta = {
        "bridge": "across",
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
    with open(MODEL_DIR / "across_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"\n  Saved to {MODEL_DIR}/")
    log.info(f"    across_model.joblib")
    log.info(f"    across_encoders.joblib")
    log.info(f"    across_meta.json")

    # --- Sample predictions on test set ---
    log.info("\n  Sample predictions (last 10 test rows):")
    y_pred_log = best_model.predict(X_test[-10:])
    y_pred = np.maximum(np.expm1(y_pred_log), 0)
    y_actual = np.expm1(y_test[-10:])
    for i, (actual, pred) in enumerate(zip(y_actual, y_pred)):
        log.info(f"    [{i}] actual=${actual:.4f}  predicted=${pred:.4f}  diff=${abs(actual-pred):.4f}")

    log.info("\n" + "=" * 60)
    log.info("Training complete!")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
