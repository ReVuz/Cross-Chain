"""
Data collection and enrichment pipeline.

Appends live quote data to the training dataset and triggers retraining
when enough new rows have accumulated.
"""

import os
import csv
import logging
from pathlib import Path
from datetime import datetime, timezone

import httpx

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
COLLECTED_DATA = BASE_DIR / "collected_live_data.csv"

SEED_COLUMNS = [
    "src_timestamp", "bridge", "src_blockchain", "dst_blockchain",
    "amount_usd", "src_fee_usd", "user_cost", "operator_cost",
    "latency", "src_symbol", "dune_hourly_gas_gwei", "eth_price_at_src",
    "hour_of_day", "day_of_week", "is_weekend", "month", "route",
    "gas_1h_lag", "gas_6h_avg", "gas_24h_avg", "gas_volatility_24h",
    "eth_price_change_1h", "eth_price_24h_avg", "bridge_hourly_volume",
]


def _ensure_collected_file():
    """Create collected_live_data.csv with headers if it doesn't exist."""
    if not COLLECTED_DATA.exists():
        with open(COLLECTED_DATA, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(SEED_COLUMNS)


def append_quote_row(
    bridge: str,
    source_chain: str,
    dest_chain: str,
    token: str,
    amount_usd: float,
    fee_usd: float,
    estimated_time: int,
    gas_gwei: float | None = None,
    eth_price: float | None = None,
):
    """Append a live quote observation to the collected dataset.
    Timestamp = when the quote was fetched (current market snapshot).
    """
    _ensure_collected_file()
    now = datetime.now(timezone.utc)

    bridge_key = _normalise_bridge_key(bridge)
    route = f"{source_chain.lower()}\u2192{dest_chain.lower()}"

    row = {
        "src_timestamp": int(now.timestamp()),
        "bridge": bridge_key,
        "src_blockchain": source_chain.lower(),
        "dst_blockchain": dest_chain.lower(),
        "amount_usd": amount_usd,
        "src_fee_usd": fee_usd,
        "user_cost": fee_usd,
        "operator_cost": 0,
        "latency": estimated_time,
        "src_symbol": token.upper(),
        "dune_hourly_gas_gwei": gas_gwei or "",
        "eth_price_at_src": eth_price or "",
        "hour_of_day": now.hour,
        "day_of_week": now.weekday(),
        "is_weekend": 1 if now.weekday() >= 5 else 0,
        "month": now.month,
        "route": route,
        "gas_1h_lag": gas_gwei or "",
        "gas_6h_avg": gas_gwei or "",
        "gas_24h_avg": gas_gwei or "",
        "gas_volatility_24h": "",
        "eth_price_change_1h": "",
        "eth_price_24h_avg": eth_price or "",
        "bridge_hourly_volume": "",
    }

    try:
        with open(COLLECTED_DATA, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SEED_COLUMNS)
            writer.writerow(row)
    except Exception as e:
        log.error(f"Failed to append collected row: {e}")


def _normalise_bridge_key(name: str) -> str:
    n = name.lower()
    if "across" in n:
        return "across"
    if "cctp" in n:
        return "cctp"
    if "ccip" in n:
        return "ccip"
    if "bus" in n:
        return "stargate_bus"
    if "stargate" in n:
        return "stargate_oft"
    if "debridge" in n or "dln" in n:
        return "debridge"
    return n


async def fetch_gas_price() -> float | None:
    """Fetch current Ethereum gas price in gwei."""
    try:
        key = os.getenv("ETHERSCAN_API_KEY", "")
        url = "https://api.etherscan.io/api?module=gastracker&action=gasoracle"
        if key:
            url += f"&apikey={key}"
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "1":
                    return float(data["result"].get("ProposeGasPrice", 15))
    except Exception as e:
        log.warning(f"Gas price fetch failed: {e}")
    return None


async def fetch_eth_price() -> float | None:
    """Fetch current ETH/USD price."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": "ethereum", "vs_currencies": "usd"},
            )
            if resp.status_code == 200:
                return resp.json().get("ethereum", {}).get("usd")
    except Exception as e:
        log.warning(f"ETH price fetch failed: {e}")
    return None


def count_collected_rows() -> int:
    """Count rows in collected_live_data.csv."""
    if not COLLECTED_DATA.exists():
        return 0
    try:
        with open(COLLECTED_DATA) as f:
            return max(0, sum(1 for _ in f) - 1)
    except Exception:
        return 0
