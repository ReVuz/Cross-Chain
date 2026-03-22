"""
Fetch recent cross-chain bridge transaction data from live APIs.

Sources:
  - Across Protocol: Real filled deposits from app.across.to/api/deposits
  - CCTP / Stargate / CCIP: Systematic LiFi quotes across amounts & routes

Enriches every row with gas price + ETH price, then appends to
cleaned_split_data/ CSVs for retraining.
"""

import csv
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "cleaned_split_data"

CHAIN_IDS = {1: "ethereum", 42161: "arbitrum", 10: "optimism", 8453: "base", 137: "polygon"}
CHAIN_ID_REVERSE = {v: k for k, v in CHAIN_IDS.items()}

USDC_ADDRESSES = {
    1: "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    42161: "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
    10: "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85",
    8453: "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
    137: "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359",
}

USDC_ALL = {a.lower() for a in USDC_ADDRESSES.values()}
WETH_ALL = {
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
    "0x82af49447d8a07e3bd95bd0d56f35241523fbab1",
    "0x4200000000000000000000000000000000000006",
}

QUOTE_ADDRESS = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"

OUTPUT_COLS = [
    "src_timestamp", "bridge", "src_blockchain", "dst_blockchain",
    "amount_usd", "user_cost", "operator_cost", "latency", "src_symbol",
    "src_fee_usd", "dune_hourly_gas_gwei", "eth_price_at_src",
    "hour_of_day", "day_of_week", "is_weekend", "month", "route",
    "gas_1h_lag", "gas_6h_avg", "gas_24h_avg", "gas_volatility_24h",
    "eth_price_change_1h", "eth_price_24h_avg", "bridge_hourly_volume",
]


# ── Market data cache ───────────────────────────────────────────────────────

_gas_cache: float | None = None
_eth_cache: float | None = None


def _fetch_gas() -> float:
    global _gas_cache
    if _gas_cache is not None:
        return _gas_cache
    try:
        resp = httpx.get(
            "https://api.etherscan.io/api?module=gastracker&action=gasoracle",
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "1":
                _gas_cache = float(data["result"]["ProposeGasPrice"])
                return _gas_cache
    except Exception:
        pass
    _gas_cache = 10.0
    return _gas_cache


def _fetch_eth_price() -> float:
    global _eth_cache
    if _eth_cache is not None:
        return _eth_cache
    try:
        resp = httpx.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": "ethereum", "vs_currencies": "usd"},
            timeout=10,
        )
        if resp.status_code == 200:
            _eth_cache = resp.json()["ethereum"]["usd"]
            return _eth_cache
    except Exception:
        pass
    _eth_cache = 2000.0
    return _eth_cache


def _enrich_row(row: dict) -> dict:
    """Fill gas/price columns from cached market data."""
    gas = _fetch_gas()
    eth = _fetch_eth_price()
    row.setdefault("dune_hourly_gas_gwei", gas)
    row.setdefault("eth_price_at_src", eth)
    row.setdefault("gas_1h_lag", gas)
    row.setdefault("gas_6h_avg", gas)
    row.setdefault("gas_24h_avg", gas)
    row.setdefault("gas_volatility_24h", 3.0)
    row.setdefault("eth_price_change_1h", 0.0)
    row.setdefault("eth_price_24h_avg", eth)
    row.setdefault("bridge_hourly_volume", 2)
    return row


def _ts_features(ts: float) -> dict:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return {
        "hour_of_day": dt.hour,
        "day_of_week": dt.weekday(),
        "is_weekend": 1 if dt.weekday() >= 5 else 0,
        "month": dt.month,
    }


# ── Across: real filled deposits ────────────────────────────────────────────

def fetch_across_deposits(max_pages: int = 20, per_page: int = 100) -> list[dict]:
    """Fetch filled Across deposits from the last ~30 days."""
    log.info("Fetching Across filled deposits...")
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    rows = []
    offset = 0

    for page in range(max_pages):
        try:
            resp = httpx.get(
                "https://app.across.to/api/deposits",
                params={"limit": per_page, "offset": offset},
                timeout=30,
            )
            if resp.status_code != 200:
                log.warning(f"Across page {page}: HTTP {resp.status_code}")
                break

            deposits = resp.json()
            if not deposits:
                break

            for d in deposits:
                ts_str = d.get("depositBlockTimestamp")
                if not ts_str:
                    continue
                dep_ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if dep_ts < cutoff:
                    log.info(f"  Reached cutoff at page {page}, offset {offset}")
                    return rows

                if d.get("status") != "filled":
                    continue

                origin = d.get("originChainId")
                dest = d.get("destinationChainId")
                if origin not in CHAIN_IDS or dest not in CHAIN_IDS:
                    continue

                inp_raw = int(d.get("inputAmount", 0))
                out_raw = int(d.get("outputAmount", 0))
                token_addr = (d.get("inputToken") or "").lower()

                if token_addr in USDC_ALL:
                    decimals, symbol = 6, "USDC"
                elif token_addr in WETH_ALL:
                    decimals, symbol = 18, "WETH"
                else:
                    continue

                amount = inp_raw / 10**decimals
                fee = (inp_raw - out_raw) / 10**decimals
                if symbol == "WETH":
                    eth_p = _fetch_eth_price()
                    amount *= eth_p
                    fee *= eth_p

                if amount <= 0 or fee < 0:
                    continue

                fill_ts_str = d.get("fillBlockTimestamp")
                latency = 0
                if fill_ts_str:
                    fill_dt = datetime.fromisoformat(fill_ts_str.replace("Z", "+00:00"))
                    latency = max(0, (fill_dt - dep_ts).total_seconds())

                epoch = dep_ts.timestamp()
                row = {
                    "src_timestamp": epoch,
                    "bridge": "across",
                    "src_blockchain": CHAIN_IDS[origin],
                    "dst_blockchain": CHAIN_IDS[dest],
                    "amount_usd": round(amount, 6),
                    "user_cost": round(abs(fee), 6),
                    "operator_cost": 0,
                    "latency": latency,
                    "src_symbol": symbol,
                    "src_fee_usd": round(abs(fee), 6),
                    "route": f"{CHAIN_IDS[origin]}\u2192{CHAIN_IDS[dest]}",
                }
                row.update(_ts_features(epoch))
                _enrich_row(row)
                rows.append(row)

            offset += per_page
            log.info(f"  Page {page}: {len(deposits)} deposits, {len(rows)} usable rows so far")
            time.sleep(0.5)

        except Exception as e:
            log.error(f"Across page {page} error: {e}")
            break

    return rows


# ── LiFi: systematic quotes for CCTP, Stargate, CCIP ───────────────────────

LIFI_RELEVANT = {"cctp", "celercircle", "celercirclefast", "stargatev2", "stargatev2bus", "stargate", "ccip"}
AMOUNTS_USDC = [10, 50, 100, 500, 1_000, 5_000, 10_000, 50_000]
ROUTE_PAIRS = [
    ("ethereum", "arbitrum"), ("ethereum", "optimism"), ("ethereum", "base"),
    ("ethereum", "polygon"), ("arbitrum", "base"), ("arbitrum", "optimism"),
    ("arbitrum", "ethereum"), ("base", "arbitrum"), ("base", "optimism"),
    ("optimism", "arbitrum"), ("optimism", "base"), ("polygon", "ethereum"),
]


def _lifi_bridge_name(tool: str) -> str:
    t = tool.lower()
    if "celercirclefast" in t:
        return "cctp"
    if "celercircle" in t or "cctp" in t:
        return "cctp"
    if "stargatev2bus" in t or "bus" in t:
        return "stargate_bus"
    if "stargate" in t:
        return "stargate_oft"
    if "ccip" in t:
        return "ccip"
    return t


def fetch_lifi_quotes() -> list[dict]:
    """Generate systematic bridge quotes across amounts and routes."""
    log.info("Fetching LiFi quotes across routes and amounts...")
    rows = []
    total_combos = len(ROUTE_PAIRS) * len(AMOUNTS_USDC)
    done = 0

    for src_name, dst_name in ROUTE_PAIRS:
        src_id = CHAIN_ID_REVERSE[src_name]
        dst_id = CHAIN_ID_REVERSE[dst_name]
        src_token = USDC_ADDRESSES.get(src_id)
        dst_token = USDC_ADDRESSES.get(dst_id)
        if not src_token or not dst_token:
            continue

        for amount in AMOUNTS_USDC:
            done += 1
            amount_raw = amount * 10**6

            body = {
                "fromChainId": src_id,
                "toChainId": dst_id,
                "fromTokenAddress": src_token,
                "toTokenAddress": dst_token,
                "fromAmount": str(amount_raw),
                "fromAddress": QUOTE_ADDRESS,
                "options": {"order": "CHEAPEST"},
            }

            try:
                resp = None
                for attempt in range(3):
                    resp = httpx.post(
                        "https://li.quest/v1/advanced/routes",
                        json=body,
                        timeout=25,
                    )
                    if resp.status_code == 429:
                        wait = 5 * (attempt + 1)
                        log.info(f"  Rate limited, waiting {wait}s...")
                        time.sleep(wait)
                        continue
                    break
                if resp is None or resp.status_code != 200:
                    continue

                now = datetime.now(timezone.utc)
                epoch = now.timestamp()
                seen_bridges: set[str] = set()

                for route in resp.json().get("routes", []):
                    steps = route.get("steps", [])
                    if not steps:
                        continue
                    tool = steps[0].get("tool", "")
                    if tool.lower() not in LIFI_RELEVANT:
                        continue

                    bridge_key = _lifi_bridge_name(tool)
                    if bridge_key in seen_bridges:
                        continue
                    seen_bridges.add(bridge_key)

                    from_amt = int(route.get("fromAmount", amount_raw))
                    to_amt = int(route.get("toAmount", amount_raw))
                    fee_usd = max((from_amt - to_amt) / 1e6, 0)
                    gas_usd = float(route.get("gasCostUSD") or 0)

                    exec_time = sum(
                        s.get("estimate", {}).get("executionDuration", 60) for s in steps
                    )

                    row = {
                        "src_timestamp": epoch,
                        "bridge": bridge_key,
                        "src_blockchain": src_name,
                        "dst_blockchain": dst_name,
                        "amount_usd": amount,
                        "user_cost": round(fee_usd, 6),
                        "operator_cost": round(gas_usd, 6),
                        "latency": exec_time,
                        "src_symbol": "USDC",
                        "src_fee_usd": round(fee_usd, 6),
                        "route": f"{src_name}\u2192{dst_name}",
                    }
                    row.update(_ts_features(epoch))
                    _enrich_row(row)
                    rows.append(row)

            except Exception as e:
                log.warning(f"LiFi {src_name}→{dst_name} ${amount}: {e}")

            if done % 10 == 0:
                log.info(f"  {done}/{total_combos} route-amount combos, {len(rows)} quotes collected")
            time.sleep(1.5)

    return rows


# ── Merge into cleaned datasets ─────────────────────────────────────────────

def append_to_cleaned(rows: list[dict]):
    """Append new rows to per-bridge cleaned CSVs."""
    if not rows:
        return

    new_df = pd.DataFrame(rows)
    bridges = new_df["bridge"].unique()

    for bridge in bridges:
        bridge_rows = new_df[new_df["bridge"] == bridge].copy()
        csv_path = DATA_DIR / f"{bridge}_cleaned.csv"

        if csv_path.exists():
            existing = pd.read_csv(csv_path)
            available_cols = [c for c in existing.columns if c in bridge_rows.columns]
            missing_in_new = [c for c in existing.columns if c not in bridge_rows.columns]

            for col in missing_in_new:
                bridge_rows[col] = ""

            bridge_rows = bridge_rows[existing.columns]
            combined = pd.concat([existing, bridge_rows], ignore_index=True)

            if "src_timestamp" in combined.columns:
                combined = combined.sort_values("src_timestamp").reset_index(drop=True)

            combined.to_csv(csv_path, index=False)
            log.info(f"  {bridge}: {len(existing)} → {len(combined)} rows (+{len(bridge_rows)})")
        else:
            bridge_rows.to_csv(csv_path, index=False)
            log.info(f"  {bridge}: created with {len(bridge_rows)} rows")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Fetching Recent Bridge Transaction Data")
    log.info("=" * 60)

    _fetch_gas()
    _fetch_eth_price()
    log.info(f"Market: gas={_gas_cache} gwei, ETH=${_eth_cache}")

    all_rows: list[dict] = []

    across_rows = fetch_across_deposits()
    log.info(f"Across: {len(across_rows)} filled deposits")
    all_rows.extend(across_rows)

    lifi_rows = fetch_lifi_quotes()
    log.info(f"LiFi quotes: {len(lifi_rows)} data points")
    all_rows.extend(lifi_rows)

    log.info(f"\nTotal new rows: {len(all_rows)}")

    by_bridge = {}
    for r in all_rows:
        b = r["bridge"]
        by_bridge[b] = by_bridge.get(b, 0) + 1
    for b, c in sorted(by_bridge.items()):
        log.info(f"  {b}: {c}")

    append_to_cleaned(all_rows)

    log.info("\nDone. Run `python -m backend.train_models` to retrain.")


if __name__ == "__main__":
    main()
