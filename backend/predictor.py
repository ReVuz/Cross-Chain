"""
Model loading, prediction, and confidence scoring for cross-chain bridge costs.

Loads trained per-bridge models (XGBoost/LightGBM) and returns predictions
with fee decomposition and confidence levels.
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import joblib

log = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent / "models"

BRIDGES_ML = ["across", "cctp", "stargate_oft", "stargate_bus"]
BRIDGES_ALL = BRIDGES_ML + ["ccip"]

CHAIN_ALIASES = {
    "ethereum": "ethereum", "eth": "ethereum",
    "arbitrum": "arbitrum", "arb": "arbitrum",
    "optimism": "optimism", "op": "optimism",
    "base": "base",
    "polygon": "polygon", "matic": "polygon",
}


class BridgePredictor:
    """Loads per-bridge models and serves predictions."""

    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.metadata = {}
        self._load_models()

    def _load_models(self):
        loaded = 0
        for bridge in BRIDGES_ALL:
            meta_path = MODEL_DIR / f"{bridge}_meta.json"
            if not meta_path.exists():
                continue

            with open(meta_path) as f:
                self.metadata[bridge] = json.load(f)

            if self.metadata[bridge].get("model_type") == "median_fallback":
                loaded += 1
                continue

            model_path = MODEL_DIR / f"{bridge}_model.joblib"
            encoder_path = MODEL_DIR / f"{bridge}_encoders.joblib"

            if model_path.exists():
                self.models[bridge] = joblib.load(model_path)
                loaded += 1
            if encoder_path.exists():
                self.encoders[bridge] = joblib.load(encoder_path)

        log.info(f"Predictor loaded {loaded} bridge models/fallbacks")

    def predict(
        self,
        source_chain: str,
        dest_chain: str,
        token: str,
        amount_usd: float,
        gas_gwei: float | None = None,
        eth_price: float | None = None,
    ) -> list[dict]:
        src = CHAIN_ALIASES.get(source_chain.lower(), source_chain.lower())
        dst = CHAIN_ALIASES.get(dest_chain.lower(), dest_chain.lower())
        route = f"{src}\u2192{dst}"

        now = datetime.now(timezone.utc)

        g = gas_gwei or 15.0
        e = eth_price or 3000.0
        h = now.hour
        hour_rad = 2 * np.pi * h / 24

        base = {
            "amount_usd": amount_usd,
            "hour_of_day": h,
            "day_of_week": now.weekday(),
            "is_weekend": 1 if now.weekday() >= 5 else 0,
            "month": now.month,
            "dune_hourly_gas_gwei": g,
            "gas_1h_lag": g,
            "gas_6h_avg": g,
            "gas_24h_avg": g,
            "gas_volatility_24h": 5.0,
            "eth_price_at_src": e,
            "eth_price_change_1h": 0.0,
            "eth_price_24h_avg": e,
            "bridge_hourly_volume": 2.0,
            "log_amount": float(np.log1p(amount_usd)),
            "amount_x_gas": amount_usd * g,
            "hour_sin": float(np.sin(hour_rad)),
            "hour_cos": float(np.cos(hour_rad)),
        }

        results = []
        for bridge in BRIDGES_ALL:
            meta = self.metadata.get(bridge)
            if not meta:
                continue
            pred = self._predict_single(bridge, meta, base, route, token)
            if pred:
                results.append(pred)

        return results

    def _predict_single(
        self, bridge: str, meta: dict, features: dict, route: str, token: str
    ) -> dict | None:
        if meta.get("model_type") == "median_fallback":
            return self._predict_median(meta, features["amount_usd"])

        model = self.models.get(bridge)
        if not model:
            return None

        encoders = self.encoders.get(bridge, {})
        feature_names = meta["features"]
        medians = meta.get("feature_medians", {})

        row = {}
        for fname in feature_names:
            if fname == "route_enc":
                enc = encoders.get("route")
                row[fname] = enc.transform([route])[0] if enc is not None and route in enc.classes_ else 0
            elif fname == "src_symbol_enc":
                enc = encoders.get("src_symbol")
                tok = token.upper()
                row[fname] = enc.transform([tok])[0] if enc is not None and tok in enc.classes_ else 0
            elif fname in features:
                row[fname] = features[fname]
            else:
                row[fname] = medians.get(fname, 0)

        import pandas as pd
        X_df = pd.DataFrame([row], columns=feature_names)
        y_log = model.predict(X_df)[0]
        y = max(float(np.expm1(y_log)), 0)

        metrics = meta.get("metrics", {})
        return {
            "bridge": bridge,
            "predicted_fee_usd": round(y, 6),
            "confidence": meta.get("confidence", "low"),
            "model_r2": metrics.get("r2"),
            "mae": metrics.get("mae"),
            "rmse": metrics.get("rmse"),
            "n_samples": meta.get("n_samples", 0),
            "last_trained": meta.get("last_trained"),
            "prediction_source": "model",
            "model_type": meta.get("model_type"),
        }

    @staticmethod
    def _predict_median(meta: dict, amount_usd: float) -> dict:
        cost = meta["metrics"].get("median_cost", 3.0)
        for bucket_range, data in meta.get("cost_by_amount_bucket", {}).items():
            lo, hi = bucket_range.split("-")
            if float(lo) <= amount_usd < float(hi):
                cost = data["median"]
                break

        return {
            "bridge": "ccip",
            "predicted_fee_usd": round(cost, 6),
            "confidence": "low",
            "model_r2": None,
            "mae": None,
            "rmse": None,
            "n_samples": meta.get("n_samples", 0),
            "last_trained": meta.get("last_trained"),
            "prediction_source": "recent_median",
            "model_type": "median_fallback",
        }

    def reload_models(self):
        self.models.clear()
        self.encoders.clear()
        self.metadata.clear()
        self._load_models()
