"""
FastAPI backend — cross-chain bridge cost comparison + ML prediction.

Endpoints:
    GET  /quotes       Live bridge quotes (fee + time + breakdown)
    GET  /predict      ML predictions per bridge
    GET  /eda          Pre-computed EDA stats
    GET  /data/stats   Dataset statistics
    GET  /data/recent  Recent data rows
    GET  /model/status Model performance metrics
    POST /model/retrain  Trigger retraining
"""

import os
import logging
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import json as _json

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


class _NumpyEncoder(_json.JSONEncoder):
    """Handle numpy types that stdlib json chokes on."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            if np.isnan(v) or np.isinf(v):
                return None
            return v
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class SafeJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return _json.dumps(content, cls=_NumpyEncoder).encode("utf-8")

try:
    from .bridge_apis import get_all_quotes, CHAIN_IDS
    from .predictor import BridgePredictor
    from .data_pipeline import append_quote_row
except ImportError:
    from bridge_apis import get_all_quotes, CHAIN_IDS
    from predictor import BridgePredictor
    from data_pipeline import append_quote_row

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "cleaned_split_data"
COLLECTED_DATA = BASE_DIR / "collected_live_data.csv"

CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "30"))

predictor: BridgePredictor | None = None
_quote_cache: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = BridgePredictor()
    log.info("BridgeCompare API ready")
    yield


app = FastAPI(
    title="BridgeCompare API",
    description="Cross-chain bridge cost comparison and ML prediction",
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=SafeJSONResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ──────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "service": "bridgecompare-api"}


# ── Live Quotes ─────────────────────────────────────────────────────────────

@app.get("/quotes")
async def quotes(
    source_chain: str = Query(...),
    dest_chain: str = Query(...),
    token: str = Query(default="USDC"),
    amount: str = Query(..., description="Raw token amount (with decimals, e.g. 1000000000 for 1000 USDC)"),
):
    if source_chain.lower() not in CHAIN_IDS:
        raise HTTPException(400, f"Unknown source chain: {source_chain}")
    if dest_chain.lower() not in CHAIN_IDS:
        raise HTTPException(400, f"Unknown dest chain: {dest_chain}")
    try:
        amount_raw = int(amount)
    except ValueError:
        raise HTTPException(400, "amount must be an integer (raw token units)")

    cache_key = f"{source_chain}:{dest_chain}:{token}:{amount}"
    cached = _quote_cache.get(cache_key)
    now = datetime.now(timezone.utc)
    if cached and (now - cached["ts"]).total_seconds() < CACHE_TTL:
        return {"quotes": cached["quotes"], "source": "cache"}

    bridge_quotes = await get_all_quotes(source_chain, dest_chain, token, amount_raw)
    if not bridge_quotes:
        raise HTTPException(502, "No quotes returned — bridge APIs may be temporarily unavailable")

    _quote_cache[cache_key] = {"quotes": bridge_quotes, "ts": now}

    # Persist every live quote for retraining
    gas_gwei, eth_price = await _fetch_market_data()
    amount_usd = amount_raw / 1e6
    for q in bridge_quotes:
        try:
            append_quote_row(
                bridge=q["protocol"],
                source_chain=source_chain,
                dest_chain=dest_chain,
                token=token,
                amount_usd=amount_usd,
                fee_usd=q["normalized_usd_fee"],
                estimated_time=q.get("estimated_time_seconds", 0),
                gas_gwei=gas_gwei,
                eth_price=eth_price,
            )
        except Exception as e:
            log.warning(f"Failed to save quote row: {e}")

    return {"quotes": bridge_quotes, "source": "live"}


# ── ML Predictions ──────────────────────────────────────────────────────────

@app.get("/predict")
async def predict(
    source_chain: str = Query(...),
    dest_chain: str = Query(...),
    token: str = Query(default="USDC"),
    amount: str = Query(...),
):
    if not predictor:
        raise HTTPException(503, "Models not loaded yet")

    try:
        amount_raw = int(amount)
        amount_usd = amount_raw / 1e6
    except ValueError:
        raise HTTPException(400, "Invalid amount")

    gas_gwei, eth_price = await _fetch_market_data()

    predictions = predictor.predict(
        source_chain=source_chain,
        dest_chain=dest_chain,
        token=token,
        amount_usd=amount_usd,
        gas_gwei=gas_gwei,
        eth_price=eth_price,
    )

    return {"predictions": predictions}


# ── EDA Stats ───────────────────────────────────────────────────────────────

@app.get("/eda")
def eda_stats():
    """Pre-computed EDA statistics for the frontend visualisations."""
    result = {"bridges": {}}
    try:
        for f in sorted(DATA_DIR.glob("*_cleaned.csv")):
            bridge = f.stem.replace("_cleaned", "")
            df = pd.read_csv(f)

            cost_desc = df["user_cost"].describe().to_dict() if "user_cost" in df.columns else {}

            fee_decomp = {}
            for col in ("adjusted_src_fee_usd", "adjusted_dst_fee_usd", "operator_cost"):
                if col in df.columns:
                    fee_decomp[col] = round(float(df[col].median()), 6)

            hourly = {}
            if "hour_of_day" in df.columns and "user_cost" in df.columns:
                hourly = df.groupby("hour_of_day")["user_cost"].median().round(6).to_dict()

            top_routes = {}
            if "route" in df.columns:
                top_routes = df["route"].value_counts().head(5).to_dict()

            amount_vs_cost = {}
            if "amount_usd" in df.columns and "user_cost" in df.columns:
                corr = df[["amount_usd", "user_cost"]].corr().iloc[0, 1]
                if pd.notna(corr):
                    amount_vs_cost = {"correlation": round(float(corr), 4)}

            def _safe_float(v):
                f = float(v)
                if pd.isna(f) or np.isinf(f):
                    return 0.0
                return round(f, 6)

            result["bridges"][bridge] = {
                "n_rows": len(df),
                "cost_stats": {k: _safe_float(v) for k, v in cost_desc.items()},
                "fee_decomposition": fee_decomp,
                "hourly_cost": {str(k): _safe_float(v) for k, v in hourly.items()},
                "top_routes": top_routes,
                "amount_cost_corr": amount_vs_cost,
            }
    except Exception as e:
        log.error(f"EDA error: {e}")
    return result


# ── Dataset Stats ───────────────────────────────────────────────────────────

@app.get("/data/stats")
def data_stats():
    stats = {
        "total_rows": 0,
        "seed_rows": 0,
        "collected_rows": 0,
        "file_size_mb": "0",
        "bridges": {},
        "top_routes": {},
        "oldest_timestamp": None,
        "newest_timestamp": None,
    }

    try:
        all_dfs = []
        for f in sorted(DATA_DIR.glob("*_cleaned.csv")):
            df = pd.read_csv(f)
            bridge = f.stem.replace("_cleaned", "")
            stats["bridges"][bridge] = len(df)
            all_dfs.append(df)

        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            stats["total_rows"] = len(combined)
            stats["seed_rows"] = len(combined)

            total_bytes = sum(f.stat().st_size for f in DATA_DIR.glob("*_cleaned.csv"))
            stats["file_size_mb"] = f"{total_bytes / (1024 * 1024):.1f}"

            ts_col = "src_timestamp" if "src_timestamp" in combined.columns else None
            if ts_col:
                valid = combined[ts_col].dropna()
                if len(valid):
                    stats["oldest_timestamp"] = float(valid.min())
                    stats["newest_timestamp"] = float(valid.max())

            if "route" in combined.columns:
                stats["top_routes"] = combined["route"].value_counts().head(10).to_dict()

        if COLLECTED_DATA.exists():
            td = pd.read_csv(COLLECTED_DATA)
            stats["collected_rows"] = len(td)
            stats["total_rows"] = stats["seed_rows"] + len(td)

            if len(td) and "src_timestamp" in td.columns:
                live_ts = td["src_timestamp"].dropna()
                if len(live_ts):
                    cur_newest = stats["newest_timestamp"] or 0
                    stats["newest_timestamp"] = max(cur_newest, float(live_ts.max()))

            if "bridge" in td.columns:
                for br, cnt in td["bridge"].value_counts().items():
                    stats["bridges"][br] = stats["bridges"].get(br, 0) + int(cnt)
    except Exception as e:
        log.error(f"Stats error: {e}")

    return stats


@app.get("/data/recent")
def data_recent(
    limit: int = Query(default=100, le=500),
    bridge: str = Query(default=None),
):
    rows = []
    cols = [
        "src_timestamp", "bridge", "src_blockchain", "dst_blockchain",
        "amount_usd", "src_fee_usd", "user_cost", "latency", "source",
    ]
    try:
        frames: list[pd.DataFrame] = []

        # 1) Collected live data — shown first
        if COLLECTED_DATA.exists():
            td = pd.read_csv(COLLECTED_DATA)
            if len(td):
                td["source"] = "live"
                if bridge:
                    td = td[td["bridge"] == bridge]
                frames.append(td)

        # 2) Historical seed data
        pattern = f"{bridge}_cleaned.csv" if bridge else "*_cleaned.csv"
        for f in sorted(DATA_DIR.glob(pattern)):
            df = pd.read_csv(f)
            df["source"] = "seed"
            frames.append(df)

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            if "src_timestamp" in combined.columns:
                combined = combined.sort_values("src_timestamp", ascending=False)
            combined = combined.head(limit)
            available = [c for c in cols if c in combined.columns]
            rows = combined[available].fillna("").to_dict("records")
    except Exception as e:
        log.error(f"Recent data error: {e}")

    return {"rows": rows}


# ── Model Management ────────────────────────────────────────────────────────

@app.get("/model/status")
def model_status():
    if not predictor:
        return {"metrics": {}}

    metrics = {}
    for bridge, meta in predictor.metadata.items():
        m = meta.get("metrics", {})
        metrics[bridge] = {
            "r2": m.get("r2"),
            "mae": m.get("mae"),
            "rmse": m.get("rmse"),
            "confidence": meta.get("confidence"),
            "last_trained": meta.get("last_trained"),
            "model_type": meta.get("model_type"),
            "n_samples": meta.get("n_samples"),
        }

    return {"metrics": metrics}


@app.post("/model/retrain")
async def retrain_model():
    import subprocess

    try:
        result = subprocess.run(
            ["python", "-m", "backend.train_models"],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise HTTPException(500, f"Training failed: {result.stderr[-500:]}")

        if predictor:
            predictor.reload_models()

        return {"status": "ok", "message": "Models retrained successfully"}
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "Training timed out (5 min limit)")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Helpers ─────────────────────────────────────────────────────────────────

async def _fetch_market_data() -> tuple[float | None, float | None]:
    """Fetch real-time gas price (gwei) and ETH price (USD)."""
    gas_gwei = None
    eth_price = None

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            key = os.getenv("ETHERSCAN_API_KEY", "")
            url = "https://api.etherscan.io/api?module=gastracker&action=gasoracle"
            if key:
                url += f"&apikey={key}"
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "1":
                    gas_gwei = float(data["result"].get("ProposeGasPrice", 15))
        except Exception as e:
            log.warning(f"Gas fetch failed: {e}")

        try:
            resp = await client.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": "ethereum", "vs_currencies": "usd"},
            )
            if resp.status_code == 200:
                eth_price = resp.json().get("ethereum", {}).get("usd")
        except Exception as e:
            log.warning(f"ETH price fetch failed: {e}")

    return gas_gwei, eth_price
