"""
BridgeCompare Backend — FastAPI server powering the frontend.

Endpoints:
  GET  /quotes        — Live bridge fee quotes from protocol APIs
  GET  /predict       — ML model predictions for transaction cost
  GET  /data/stats    — Training data statistics for dashboard
  GET  /data/recent   — Recent transaction rows
  GET  /model/status  — Model metrics and confidence
  POST /model/retrain — Retrain all models from latest CSV data

Run:  uvicorn backend.main:app --reload --port 8000
"""

import os
import time
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import httpx
from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

# ─── Paths ───────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "cleaned_split_data"
MODEL_DIR = ROOT / "trained_models"

# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(title="BridgeCompare API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

log = logging.getLogger("uvicorn.error")

# ─── Constants ───────────────────────────────────────────────────────────────

CHAIN_IDS = {
    "Ethereum": 1, "Arbitrum": 42161, "Optimism": 10,
    "Base": 8453, "Polygon": 137,
}

USDC_ADDRESSES = {
    1:     "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    42161: "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
    10:    "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85",
    8453:  "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
    137:   "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359",
}

USDC_DECIMALS = 6

BRIDGE_FILES = {
    "across": "across_cleaned.csv",
    "cctp": "cctp_cleaned.csv",
    "ccip": "ccip_cleaned.csv",
    "stargate_bus": "stargate_bus_cleaned.csv",
    "stargate_oft": "stargate_oft_cleaned.csv",
}

PROTOCOLS = ["across", "cctp", "stargate_bus", "stargate_oft"]

# ─── Model Cache ─────────────────────────────────────────────────────────────

_model_cache: dict = {}
_model_load_time: float = 0


def _load_models():
    global _model_cache, _model_load_time
    _model_cache = {}
    for proto in PROTOCOLS:
        path = MODEL_DIR / f"{proto}_xgboost.joblib"
        if path.exists():
            _model_cache[proto] = joblib.load(path)
    _model_load_time = time.time()
    log.info(f"Loaded {len(_model_cache)} models: {list(_model_cache.keys())}")


@app.on_event("startup")
async def startup():
    _load_models()


# ─── Quote Cache ─────────────────────────────────────────────────────────────

_quote_cache: dict = {}
CACHE_TTL = 30  # seconds


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _chain_id(name: str) -> int:
    cid = CHAIN_IDS.get(name)
    if cid is None:
        raise HTTPException(400, f"Unknown chain: {name}")
    return cid


def _human_amount(raw: str) -> float:
    return int(raw) / (10 ** USDC_DECIMALS)


# ═════════════════════════════════════════════════════════════════════════════
# 1. GET /quotes — Live bridge quotes
# ═════════════════════════════════════════════════════════════════════════════

async def _fetch_across_quote(
    client: httpx.AsyncClient, src_id: int, dst_id: int, amount_raw: str
) -> dict | None:
    try:
        token = USDC_ADDRESSES.get(src_id)
        if not token:
            return None
        url = "https://app.across.to/api/suggested-fees"
        params = {
            "inputToken": token,
            "outputToken": USDC_ADDRESSES.get(dst_id, token),
            "originChainId": src_id,
            "destinationChainId": dst_id,
            "amount": amount_raw,
        }
        r = await client.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return None
        d = r.json()
        total_fee_pct = float(d.get("totalRelayFee", {}).get("pct", "0")) / 1e18
        amount_human = _human_amount(amount_raw)
        fee_usd = amount_human * total_fee_pct

        relay_fee_pct = float(d.get("relayerCapitalFee", {}).get("pct", "0")) / 1e18
        lp_fee_pct = float(d.get("lpFee", {}).get("pct", "0")) / 1e18
        relay_gas_pct = float(d.get("relayerGasFee", {}).get("pct", "0")) / 1e18

        return {
            "protocol": "Across",
            "normalized_usd_fee": round(fee_usd, 6),
            "estimated_time_seconds": 15,
            "fee_breakdown": [
                {"name": "Relayer Capital Fee", "usd": round(amount_human * relay_fee_pct, 6), "description": "Fee for relayer capital risk"},
                {"name": "LP Fee", "usd": round(amount_human * lp_fee_pct, 6), "description": "Liquidity provider fee"},
                {"name": "Relayer Gas Fee", "usd": round(amount_human * relay_gas_pct, 6), "description": "Destination gas paid by relayer"},
            ],
        }
    except Exception as e:
        log.warning(f"Across quote failed: {e}")
        return None


async def _fetch_debridge_quote(
    client: httpx.AsyncClient, src_id: int, dst_id: int, amount_raw: str
) -> dict | None:
    try:
        token_src = USDC_ADDRESSES.get(src_id)
        token_dst = USDC_ADDRESSES.get(dst_id)
        if not token_src or not token_dst:
            return None
        url = "https://deswap.debridge.finance/v1.0/dln/order/quote"
        params = {
            "srcChainId": src_id,
            "srcChainTokenIn": token_src,
            "srcChainTokenInAmount": amount_raw,
            "dstChainId": dst_id,
            "dstChainTokenOut": token_dst,
            "prependOperatingExpenses": "true",
        }
        r = await client.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return None
        d = r.json()
        est = d.get("estimation", {})
        dst_amount = int(est.get("dstChainTokenOut", {}).get("amount", "0"))
        src_amount = int(amount_raw)
        fee_raw = src_amount - dst_amount
        fee_usd = fee_raw / (10 ** USDC_DECIMALS)

        costs = est.get("costsDetails", [])
        breakdown = []
        for c in costs:
            breakdown.append({
                "name": c.get("title", "Fee"),
                "usd": round(float(c.get("payload", {}).get("feeAmount", "0")) / (10 ** USDC_DECIMALS), 6),
                "description": c.get("type", ""),
            })

        return {
            "protocol": "deBridge",
            "normalized_usd_fee": round(max(fee_usd, 0), 6),
            "estimated_time_seconds": 30,
            "fee_breakdown": breakdown if breakdown else [
                {"name": "Total Protocol Fee", "usd": round(max(fee_usd, 0), 6), "description": "DLN solver fee"}
            ],
        }
    except Exception as e:
        log.warning(f"deBridge quote failed: {e}")
        return None


def _estimate_from_history(bridge: str, src_chain: str, dst_chain: str, amount_human: float) -> dict | None:
    """Estimate fee from historical CSV data when no live API is available."""
    csv_path = DATA_DIR / BRIDGE_FILES.get(bridge, "")
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        route_mask = (
            (df["src_blockchain"].str.lower() == src_chain.lower()) &
            (df["dst_blockchain"].str.lower() == dst_chain.lower())
        )
        route_df = df[route_mask] if route_mask.any() else df

        if route_df.empty or "user_cost" not in route_df.columns:
            return None

        recent = route_df.tail(200)
        valid = recent[recent["user_cost"] > 0]["user_cost"]
        if valid.empty:
            return None

        median_cost = float(valid.median())
        if "amount_usd" in recent.columns:
            valid_amounts = recent[recent["amount_usd"] > 0]["amount_usd"]
            if not valid_amounts.empty:
                median_amount = float(valid_amounts.median())
                if median_amount > 0:
                    rate = median_cost / median_amount
                    median_cost = rate * amount_human

        display_names = {
            "cctp": "CCTP (Standard)",
            "ccip": "CCIP",
            "stargate_bus": "Stargate V2 (Bus)",
            "stargate_oft": "Stargate V2",
        }

        estimated_times = {
            "cctp": 780, "ccip": 900,
            "stargate_bus": 240, "stargate_oft": 180,
        }

        return {
            "protocol": display_names.get(bridge, bridge),
            "normalized_usd_fee": round(median_cost, 6),
            "estimated_time_seconds": estimated_times.get(bridge, 300),
            "fee_breakdown": [
                {"name": "Estimated Fee", "usd": round(median_cost, 6),
                 "description": f"Median from {len(valid)} recent transactions"},
            ],
        }
    except Exception as e:
        log.warning(f"History estimate for {bridge} failed: {e}")
        return None


@app.get("/quotes")
async def get_quotes(
    source_chain: str = Query(...),
    dest_chain: str = Query(...),
    token: str = Query("USDC"),
    amount: str = Query(...),
):
    src_id = _chain_id(source_chain)
    dst_id = _chain_id(dest_chain)
    if src_id == dst_id:
        raise HTTPException(400, "Source and destination must differ")

    cache_key = f"{src_id}:{dst_id}:{amount}"
    now = time.time()
    if cache_key in _quote_cache and (now - _quote_cache[cache_key]["ts"]) < CACHE_TTL:
        return {"source": "cache", "quotes": _quote_cache[cache_key]["quotes"]}

    amount_human = _human_amount(amount)

    async with httpx.AsyncClient() as client:
        tasks = [
            _fetch_across_quote(client, src_id, dst_id, amount),
            _fetch_debridge_quote(client, src_id, dst_id, amount),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    quotes = [r for r in results if isinstance(r, dict)]

    for bridge in ["cctp", "ccip", "stargate_bus", "stargate_oft"]:
        est = _estimate_from_history(bridge, source_chain, dest_chain, amount_human)
        if est:
            quotes.append(est)

    if not quotes:
        raise HTTPException(502, "No bridge quotes available")

    _quote_cache[cache_key] = {"ts": now, "quotes": quotes}
    return {"source": "live", "quotes": quotes}


# ═════════════════════════════════════════════════════════════════════════════
# 2. GET /predict — ML model predictions
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/predict")
async def predict_fees(
    source_chain: str = Query(...),
    dest_chain: str = Query(...),
    token: str = Query("USDC"),
    amount: str = Query(...),
):
    amount_human = _human_amount(amount)
    route = f"{source_chain}→{dest_chain}"
    now = datetime.now(timezone.utc)

    predictions = []

    for bridge, bundle in _model_cache.items():
        try:
            model = bundle["model"]
            feature_cols = bundle["feature_cols"]
            label_encoders = bundle["label_encoders"]
            metrics = bundle.get("metrics", {})

            features = {}
            for col in feature_cols:
                if col == "amount_usd":
                    features[col] = amount_human
                elif col == "hour_of_day":
                    features[col] = now.hour
                elif col == "day_of_week":
                    features[col] = now.weekday()
                elif col == "is_weekend":
                    features[col] = 1 if now.weekday() >= 5 else 0
                elif col == "month":
                    features[col] = now.month
                elif col == "route":
                    le = label_encoders.get("route")
                    if le and route in le.classes_:
                        features[col] = int(le.transform([route])[0])
                    else:
                        features[col] = 0
                elif col == "src_symbol":
                    le = label_encoders.get("src_symbol")
                    if le and token in le.classes_:
                        features[col] = int(le.transform([token])[0])
                    else:
                        features[col] = 0
                else:
                    features[col] = _get_recent_feature_value(bridge, col)

            X = pd.DataFrame([features])[feature_cols]
            pred_log = model.predict(X)[0]
            pred_usd = max(float(np.expm1(pred_log)), 0)

            predictions.append({
                "bridge": bridge,
                "predicted_fee_usd": round(pred_usd, 6),
                "confidence": metrics.get("confidence", "low"),
                "model_r2": metrics.get("r2_log"),
                "mae": metrics.get("mae"),
                "n_samples": bundle.get("trained_on", 0),
                "prediction_source": "model",
                "last_trained": bundle.get("trained_at"),
            })

        except Exception as e:
            log.warning(f"Prediction failed for {bridge}: {e}")

    # Add recent_median fallback for bridges without models
    for bridge in PROTOCOLS:
        if bridge not in _model_cache:
            med = _get_recent_median(bridge, source_chain, dest_chain, amount_human)
            if med is not None:
                predictions.append(med)

    return {"predictions": predictions}


_feature_cache: dict = {}
_feature_cache_ts: float = 0
FEATURE_CACHE_TTL = 60


def _get_recent_feature_value(bridge: str, col: str) -> float:
    """Get the most recent value of a feature column from the CSV (cached)."""
    global _feature_cache, _feature_cache_ts
    now = time.time()

    cache_key = f"{bridge}:{col}"
    if cache_key in _feature_cache and (now - _feature_cache_ts) < FEATURE_CACHE_TTL:
        return _feature_cache[cache_key]

    csv_path = DATA_DIR / BRIDGE_FILES.get(bridge, "")
    if not csv_path.exists():
        return 0
    try:
        header = pd.read_csv(csv_path, nrows=0).columns.tolist()
        if col not in header:
            return 0
        df = pd.read_csv(csv_path, usecols=[col])
        valid = pd.to_numeric(df[col], errors="coerce").dropna()
        if not valid.empty:
            val = float(valid.iloc[-1])
            _feature_cache[cache_key] = val
            _feature_cache_ts = now
            return val
    except Exception:
        pass
    return 0


def _get_recent_median(
    bridge: str, src: str, dst: str, amount: float
) -> dict | None:
    csv_path = DATA_DIR / BRIDGE_FILES.get(bridge, "")
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        mask = (
            (df["src_blockchain"].str.lower() == src.lower()) &
            (df["dst_blockchain"].str.lower() == dst.lower())
        )
        subset = df[mask] if mask.any() else df
        recent = subset.tail(90)
        valid = recent[recent["user_cost"] > 0]
        if valid.empty:
            return None
        med = float(valid["user_cost"].median())
        return {
            "bridge": bridge,
            "predicted_fee_usd": round(med, 6),
            "confidence": "medium",
            "model_r2": None,
            "mae": None,
            "n_samples": len(valid),
            "prediction_source": "recent_median",
            "last_trained": None,
        }
    except Exception:
        return None


# ═════════════════════════════════════════════════════════════════════════════
# 3. GET /data/stats — Training data statistics
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/data/stats")
async def data_stats():
    bridges = {}
    total_rows = 0
    all_timestamps = []
    all_routes = []

    for bridge, fname in BRIDGE_FILES.items():
        path = DATA_DIR / fname
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            count = len(df)
            bridges[bridge] = count
            total_rows += count

            ts_col = pd.to_numeric(df.get("src_timestamp", pd.Series()), errors="coerce").dropna()
            valid_ts = ts_col[ts_col > 1600000000]
            if not valid_ts.empty:
                all_timestamps.extend([valid_ts.min(), valid_ts.max()])

            if "route" in df.columns:
                all_routes.extend(df["route"].dropna().tolist())
            elif "src_blockchain" in df.columns and "dst_blockchain" in df.columns:
                routes = df["src_blockchain"].astype(str) + "→" + df["dst_blockchain"].astype(str)
                all_routes.extend(routes.tolist())
        except Exception as e:
            log.warning(f"Failed to read {fname}: {e}")

    route_counts = pd.Series(all_routes).value_counts().head(10).to_dict()

    total_size = sum(
        (DATA_DIR / f).stat().st_size for f in BRIDGE_FILES.values() if (DATA_DIR / f).exists()
    )

    return {
        "total_rows": total_rows,
        "seed_rows": total_rows,
        "collected_rows": 0,
        "file_size_mb": round(total_size / (1024 * 1024), 2),
        "oldest_timestamp": int(min(all_timestamps)) if all_timestamps else None,
        "newest_timestamp": int(max(all_timestamps)) if all_timestamps else None,
        "bridges": bridges,
        "top_routes": route_counts,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 4. GET /data/recent — Recent data rows
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/data/recent")
async def data_recent(
    limit: int = Query(100, ge=1, le=1000),
    bridge: str = Query(None),
):
    frames = []
    targets = {bridge: BRIDGE_FILES[bridge]} if bridge and bridge in BRIDGE_FILES else BRIDGE_FILES

    for bname, fname in targets.items():
        path = DATA_DIR / fname
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            df["bridge"] = bname
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return {"rows": []}

    combined = pd.concat(frames, ignore_index=True)
    combined["src_timestamp"] = pd.to_numeric(combined["src_timestamp"], errors="coerce")
    combined = combined.sort_values("src_timestamp", ascending=False).head(limit)

    cols = ["src_timestamp", "bridge", "src_blockchain", "dst_blockchain",
            "amount_usd", "src_fee_usd", "user_cost", "latency"]
    existing = [c for c in cols if c in combined.columns]
    result = combined[existing].replace({np.nan: None})

    return {"rows": result.to_dict(orient="records")}


# ═════════════════════════════════════════════════════════════════════════════
# 5. GET /model/status — Model metrics
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/model/status")
async def model_status():
    if not _model_cache:
        return {"metrics": {}}

    metrics = {}
    for bridge, bundle in _model_cache.items():
        m = bundle.get("metrics", {})
        metrics[bridge] = {
            "r2": m.get("r2_log"),
            "mae": m.get("mae"),
            "rmse": m.get("rmse"),
            "confidence": m.get("confidence"),
            "last_trained": bundle.get("trained_at"),
        }
    return {"metrics": metrics}


# ═════════════════════════════════════════════════════════════════════════════
# 6. POST /model/retrain — Retrain models
# ═════════════════════════════════════════════════════════════════════════════

_retraining = False


def _retrain_sync():
    global _retraining
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from xgboost import XGBRegressor

    numeric_features = [
        "amount_usd", "dune_hourly_gas_gwei", "gas_1h_lag", "gas_6h_avg",
        "gas_24h_avg", "gas_volatility_24h", "eth_price_at_src",
        "eth_price_change_1h", "eth_price_24h_avg", "bridge_hourly_volume",
        "hour_of_day", "day_of_week", "is_weekend", "month",
    ]
    categorical_features = ["route", "src_symbol"]
    TARGET = "user_cost"

    MODEL_DIR.mkdir(exist_ok=True)

    for proto in PROTOCOLS:
        path = DATA_DIR / BRIDGE_FILES.get(proto, "")
        if not path.exists():
            continue

        df = pd.read_csv(path)
        df["src_timestamp"] = pd.to_numeric(df["src_timestamp"], errors="coerce")
        df = df.sort_values("src_timestamp").reset_index(drop=True)

        an = [f for f in numeric_features if f in df.columns]
        ac = [f for f in categorical_features if f in df.columns]
        fc = an + ac

        m = df[fc + [TARGET]].copy()
        les = {}
        for c in ac:
            le = LabelEncoder()
            m[c] = le.fit_transform(m[c].astype(str))
            les[c] = le

        m = m.replace([np.inf, -np.inf], np.nan).dropna()
        m = m[m[TARGET] > 0]
        m["log_target"] = np.log1p(m[TARGET])

        if len(m) < 50:
            continue

        si = int(len(m) * 0.8)
        xgb = XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbosity=0,
        )
        xgb.fit(m.iloc[:si][fc], m.iloc[:si]["log_target"])

        yp = xgb.predict(m.iloc[si:][fc])
        yr = np.maximum(np.expm1(yp), 0)
        mae = mean_absolute_error(m.iloc[si:][TARGET], yr)
        rmse = float(np.sqrt(mean_squared_error(m.iloc[si:][TARGET], yr)))
        r2 = r2_score(m.iloc[si:]["log_target"], yp)
        conf = "high" if r2 > 0.7 else "medium" if r2 > 0.4 else "low"

        bundle = {
            "model": xgb, "feature_cols": fc, "label_encoders": les,
            "trained_on": len(m),
            "last_timestamp": int(df["src_timestamp"].max()),
            "metrics": {"mae": mae, "rmse": rmse, "r2_log": r2, "confidence": conf},
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        joblib.dump(bundle, MODEL_DIR / f"{proto}_xgboost.joblib")
        log.info(f"Retrained {proto}: R²={r2:.4f}, MAE=${mae:.4f}")

    _load_models()
    _retraining = False


@app.post("/model/retrain")
async def retrain_models(background_tasks: BackgroundTasks):
    global _retraining
    if _retraining:
        raise HTTPException(409, "Retraining already in progress")
    _retraining = True
    background_tasks.add_task(_retrain_sync)
    return {"status": "retraining_started"}


# ─── Health ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": len(_model_cache),
        "data_files": sum(1 for f in BRIDGE_FILES.values() if (DATA_DIR / f).exists()),
    }
