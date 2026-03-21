"""
Automated Data Pipeline for Bridge Transaction Cost Prediction
==============================================================
Fetches new bridge transactions, enriches with ETH prices & gas data,
computes ML features, validates quality, and appends to training CSVs.

Usage:
    python data_pipeline.py                    # Run for all bridges
    python data_pipeline.py --bridge across    # Run for one bridge
    python data_pipeline.py --dry-run          # Validate without saving

Schedule with cron (every 6 hours):
    0 */6 * * * cd /home/hp/Documents/web3 && .venv/bin/python data_pipeline.py >> pipeline.log 2>&1
"""

import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np
import requests
from dune_client.client import DuneClient
from dune_client.query import QueryBase
from dune_client.types import QueryParameter

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — Edit these values
# ─────────────────────────────────────────────────────────────────────────────

DUNE_API_KEY = "yQuSpPvC3rSMsuLc5X4SypC4p7FNPeC3"
ALCHEMY_KEY = "C8zWXOtflxcAq-nbck_cO"

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cleaned_split_data")
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_logs")

DUNE_GAS_QUERY_ID = 6663059

# Dune query IDs for bridge transactions.
# You MUST create these queries on dune.com using the SQL templates in
# dune_queries/ folder, save them, and paste the query IDs here.
BRIDGE_QUERY_IDS = {
    "across": None,        # <-- Set after creating query on dune.com
    "cctp": None,          # <-- Set after creating query on dune.com
    "stargate_bus": None,  # <-- Set after creating query on dune.com
    "stargate_oft": None,  # <-- Set after creating query on dune.com
}

# Columns the ML model actually needs — these MUST NOT be blank
CRITICAL_COLUMNS = {
    "across": [
        "amount_usd", "src_fee_usd", "dst_fee_usd", "src_timestamp",
        "dst_timestamp", "src_blockchain", "dst_blockchain", "src_symbol",
        "user_cost", "latency",
    ],
    "cctp": [
        "amount_usd", "src_fee_usd", "dst_fee_usd", "src_timestamp",
        "dst_timestamp", "src_blockchain", "dst_blockchain", "src_symbol",
        "user_cost", "latency",
    ],
    "stargate_bus": [
        "amount_usd", "user_fee_usd", "dst_fee_usd", "src_timestamp",
        "dst_timestamp", "src_blockchain", "dst_blockchain", "src_symbol",
        "user_cost", "latency",
    ],
    "stargate_oft": [
        "amount_usd", "src_fee_usd", "dst_fee_usd", "src_timestamp",
        "dst_timestamp", "src_blockchain", "dst_blockchain", "src_symbol",
        "user_cost", "latency",
    ],
}

# Expected Dune query output columns → dataset column mapping per bridge.
# Adjust these if your Dune query uses different column names.
DUNE_COLUMN_MAP = {
    "across": {
        "src_tx_hash": "src_transaction_hash",
        "dst_tx_hash": "dst_transaction_hash",
        "src_block_time": "src_timestamp",
        "dst_block_time": "dst_timestamp",
        "src_chain": "src_blockchain",
        "dst_chain": "dst_blockchain",
        "token_symbol": "src_symbol",
        "input_amount_usd": "amount_usd",
    },
    "cctp": {
        "src_tx_hash": "src_transaction_hash",
        "dst_tx_hash": "dst_transaction_hash",
        "src_block_time": "src_timestamp",
        "dst_block_time": "dst_timestamp",
        "src_chain": "src_blockchain",
        "dst_chain": "dst_blockchain",
        "token_symbol": "src_symbol",
    },
    "stargate_bus": {
        "src_tx_hash": "user_transaction_hash",
        "dst_tx_hash": "dst_transaction_hash",
        "src_block_time": "src_timestamp",
        "dst_block_time": "dst_timestamp",
        "src_chain": "src_blockchain",
        "dst_chain": "dst_blockchain",
        "token_symbol": "src_symbol",
        "amount_sent_usd": "amount_usd",
    },
    "stargate_oft": {
        "src_tx_hash": "src_transaction_hash",
        "dst_tx_hash": "dst_transaction_hash",
        "src_block_time": "src_timestamp",
        "dst_block_time": "dst_timestamp",
        "src_chain": "src_blockchain",
        "dst_chain": "dst_blockchain",
        "token_symbol": "src_symbol",
    },
}

# How many rows of existing data to load for computing rolling features
ROLLING_CONTEXT_ROWS = 100

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(LOG_DIR, f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ),
    ],
)
log = logging.getLogger("data_pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────

class DuneFetcher:
    """Fetches bridge transaction data and gas data from Dune Analytics."""

    def __init__(self, api_key: str):
        self.client = DuneClient(api_key)

    def fetch_bridge_transactions(self, bridge: str, start_timestamp: int) -> pd.DataFrame:
        query_id = BRIDGE_QUERY_IDS.get(bridge)
        if query_id is None:
            log.warning(f"[{bridge}] No Dune query ID configured — skipping Dune fetch")
            return pd.DataFrame()

        log.info(f"[{bridge}] Fetching transactions from Dune query {query_id} since {start_timestamp}")

        start_str = datetime.fromtimestamp(start_timestamp, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        query = QueryBase(
            query_id=query_id,
            params=[QueryParameter.text_type("start_date", start_str)],
        )

        try:
            df = self.client.run_query_dataframe(query)
            log.info(f"[{bridge}] Dune returned {len(df)} rows")

            col_map = DUNE_COLUMN_MAP.get(bridge, {})
            existing_cols = {k: v for k, v in col_map.items() if k in df.columns}
            if existing_cols:
                df = df.rename(columns=existing_cols)

            return df
        except Exception as e:
            log.error(f"[{bridge}] Dune fetch failed: {e}")
            return pd.DataFrame()

    def fetch_gas_data(self, start_str: str, end_str: str) -> pd.DataFrame:
        log.info(f"Fetching gas data from Dune: {start_str} to {end_str}")

        query = QueryBase(
            query_id=DUNE_GAS_QUERY_ID,
            params=[
                QueryParameter.text_type("start_date", start_str),
                QueryParameter.text_type("end_date", end_str),
            ],
        )

        try:
            gas_df = self.client.run_query_dataframe(query)
            log.info(f"Gas data: {len(gas_df)} hourly records")

            rename_map = {
                "block_hour": "join_key",
                "avg_gas_price_gwei": "dune_hourly_gas_gwei",
            }
            if "dt" in gas_df.columns:
                rename_map["dt"] = "join_key"
            if "gas_price" in gas_df.columns:
                rename_map["gas_price"] = "dune_hourly_gas_gwei"

            gas_df = gas_df.rename(columns=rename_map)
            gas_df["join_key"] = (
                pd.to_datetime(gas_df["join_key"]).dt.tz_localize(None).dt.floor("h")
            )
            gas_df = gas_df.drop_duplicates(subset=["join_key"])
            return gas_df
        except Exception as e:
            log.error(f"Gas data fetch failed: {e}")
            return pd.DataFrame()


class AlchemyFetcher:
    """Fetches historical ETH prices from Alchemy."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_eth_prices(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        log.info(f"Fetching ETH prices: {start_date.date()} to {end_date.date()}")
        all_prices = []
        current = start_date
        chunk_days = 28

        while current < end_date:
            chunk_end = min(current + pd.Timedelta(days=chunk_days), end_date)

            url = f"https://api.g.alchemy.com/prices/v1/{self.api_key}/tokens/historical"
            payload = {
                "symbol": "ETH",
                "startTime": current.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "endTime": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "interval": "1h",
                "withMarketData": True,
            }

            try:
                r = requests.post(url, json=payload, headers={"accept": "application/json"})
                if r.status_code == 200:
                    data = r.json().get("data", [])
                    all_prices.extend(data)
                else:
                    log.warning(f"Alchemy returned status {r.status_code} for {current.date()}")
            except Exception as e:
                log.error(f"Alchemy fetch error: {e}")

            current = chunk_end
            time.sleep(0.2)

        if not all_prices:
            log.warning("No ETH price data fetched")
            return pd.DataFrame()

        price_df = pd.DataFrame(all_prices)
        price_df["timestamp"] = pd.to_datetime(price_df["timestamp"], format="mixed", utc=True)
        price_df.rename(columns={"value": "eth_price_at_src"}, inplace=True)
        price_df = price_df.drop_duplicates(subset=["timestamp"])
        price_df["join_key"] = price_df["timestamp"].dt.floor("h")

        log.info(f"Loaded {len(price_df)} hourly ETH price points")
        return price_df


# ─────────────────────────────────────────────────────────────────────────────
# ENRICHMENT & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure src_timestamp and dst_timestamp are Unix seconds (numeric)."""
    for col in ["src_timestamp", "dst_timestamp"]:
        if col not in df.columns:
            continue
        if df[col].dtype == "object" or hasattr(df[col].dtype, "tz"):
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
            df[col] = df[col].astype("int64") // 10**9
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def enrich_eth_prices(df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    if price_df.empty or "src_timestamp" not in df.columns:
        return df

    df["_temp_ts"] = pd.to_datetime(
        pd.to_numeric(df["src_timestamp"], errors="coerce"), unit="s", utc=True
    )
    df["join_key"] = df["_temp_ts"].dt.floor("h")
    df = df.merge(price_df[["join_key", "eth_price_at_src"]], on="join_key", how="left")
    df.drop(columns=["join_key", "_temp_ts"], inplace=True, errors="ignore")
    return df


def enrich_gas_data(df: pd.DataFrame, gas_df: pd.DataFrame) -> pd.DataFrame:
    if gas_df.empty or "src_timestamp" not in df.columns:
        return df

    df["join_key"] = pd.to_datetime(
        pd.to_numeric(df["src_timestamp"], errors="coerce"), unit="s"
    ).dt.floor("h")
    df = df.merge(gas_df[["join_key", "dune_hourly_gas_gwei"]], on="join_key", how="left")
    df.drop(columns=["join_key"], inplace=True, errors="ignore")
    return df


def compute_user_cost(df: pd.DataFrame, bridge: str) -> pd.DataFrame:
    """
    Compute user_cost based on bridge-specific logic.
    user_cost represents the TOTAL cost the user pays for the bridge transfer.
    """
    if "user_cost" in df.columns and df["user_cost"].notna().all():
        return df

    if bridge == "across":
        src_fee = pd.to_numeric(df.get("adjusted_src_fee_usd", df.get("src_fee_usd", 0)), errors="coerce").fillna(0)
        input_usd = pd.to_numeric(df.get("amount_usd", df.get("input_amount_usd", 0)), errors="coerce").fillna(0)
        output_usd = pd.to_numeric(df.get("output_amount_usd", 0), errors="coerce").fillna(0)
        spread = np.maximum(input_usd - output_usd, 0)
        df["user_cost"] = src_fee + spread

    elif bridge == "cctp":
        src_fee = pd.to_numeric(df.get("adjusted_src_fee_usd", df.get("src_fee_usd", 0)), errors="coerce").fillna(0)
        dst_fee = pd.to_numeric(df.get("adjusted_dst_fee_usd", df.get("dst_fee_usd", 0)), errors="coerce").fillna(0)
        df["user_cost"] = src_fee + dst_fee

    elif bridge == "stargate_bus":
        user_fee = pd.to_numeric(df.get("adjusted_user_fee_usd", df.get("user_fee_usd", 0)), errors="coerce").fillna(0)
        bus_fee = pd.to_numeric(df.get("adjusted_bus_fee_usd", df.get("bus_fee_usd", 0)), errors="coerce").fillna(0)
        df["user_cost"] = user_fee + bus_fee

    elif bridge == "stargate_oft":
        src_fee = pd.to_numeric(df.get("adjusted_src_fee_usd", df.get("src_fee_usd", 0)), errors="coerce").fillna(0)
        dst_fee = pd.to_numeric(df.get("adjusted_dst_fee_usd", df.get("dst_fee_usd", 0)), errors="coerce").fillna(0)
        df["user_cost"] = src_fee + dst_fee

    return df


def compute_latency(df: pd.DataFrame) -> pd.DataFrame:
    if "src_timestamp" in df.columns and "dst_timestamp" in df.columns:
        src = pd.to_numeric(df["src_timestamp"], errors="coerce")
        dst = pd.to_numeric(df["dst_timestamp"], errors="coerce")
        df["latency"] = dst - src
    return df


def compute_features(df: pd.DataFrame, existing_tail: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Compute all ML features. Uses existing_tail (last N rows from CSV)
    as context for rolling calculations so the first rows aren't blank.
    """
    if existing_tail is not None and not existing_tail.empty:
        context_len = len(existing_tail)
        combined = pd.concat([existing_tail, df], ignore_index=True)
    else:
        context_len = 0
        combined = df.copy()

    combined["src_timestamp"] = pd.to_numeric(combined["src_timestamp"], errors="coerce")
    combined["datetime"] = pd.to_datetime(combined["src_timestamp"], unit="s", utc=True)
    combined = combined.sort_values("datetime").reset_index(drop=True)

    # Temporal features
    combined["hour_of_day"] = combined["datetime"].dt.hour
    combined["day_of_week"] = combined["datetime"].dt.dayofweek
    combined["is_weekend"] = (combined["day_of_week"] >= 5).astype(int)
    combined["month"] = combined["datetime"].dt.month

    # Route
    if "src_blockchain" in combined.columns and "dst_blockchain" in combined.columns:
        combined["route"] = (
            combined["src_blockchain"].astype(str) + "→" + combined["dst_blockchain"].astype(str)
        )

    # Gas rolling features
    if "dune_hourly_gas_gwei" in combined.columns:
        combined["gas_1h_lag"] = combined["dune_hourly_gas_gwei"].shift(1)
        combined["gas_6h_avg"] = combined["dune_hourly_gas_gwei"].rolling(6, min_periods=1).mean()
        combined["gas_24h_avg"] = combined["dune_hourly_gas_gwei"].rolling(24, min_periods=1).mean()
        combined["gas_volatility_24h"] = (
            combined["dune_hourly_gas_gwei"].rolling(24, min_periods=1).std()
        )

    # ETH price features
    if "eth_price_at_src" in combined.columns:
        combined["eth_price_change_1h"] = combined["eth_price_at_src"].pct_change()
        combined["eth_price_24h_avg"] = (
            combined["eth_price_at_src"].rolling(24, min_periods=1).mean()
        )

    # Bridge hourly volume
    combined["hour_bucket"] = combined["datetime"].dt.floor("h")
    hourly_vol = combined.groupby("hour_bucket").size().rename("bridge_hourly_volume")
    combined = combined.merge(hourly_vol, on="hour_bucket", how="left")
    combined.drop(columns=["hour_bucket"], inplace=True, errors="ignore")

    # Fee rate
    amt_col = "amount_usd"
    if amt_col in combined.columns:
        combined["fee_rate_bps"] = np.where(
            pd.to_numeric(combined[amt_col], errors="coerce") > 0,
            (
                pd.to_numeric(
                    combined.get("user_cost", combined.get("src_fee_usd", 0)), errors="coerce"
                )
                / pd.to_numeric(combined[amt_col], errors="coerce")
            )
            * 10000,
            np.nan,
        )

    # Operator cost (if computable)
    if "operator_cost" not in combined.columns:
        src = pd.to_numeric(combined.get("adjusted_src_fee_usd", combined.get("src_fee_usd", 0)), errors="coerce").fillna(0)
        dst = pd.to_numeric(combined.get("adjusted_dst_fee_usd", combined.get("dst_fee_usd", 0)), errors="coerce").fillna(0)
        combined["operator_cost"] = src + dst

    # Strip the context rows — return only new data
    if context_len > 0:
        result = combined.iloc[context_len:].reset_index(drop=True)
    else:
        result = combined

    return result


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_dataframe(df: pd.DataFrame, bridge: str) -> tuple[pd.DataFrame, dict]:
    """
    Validate new data. Returns (cleaned_df, report_dict).
    Drops rows where critical columns are blank rather than appending bad data.
    """
    report = {
        "bridge": bridge,
        "input_rows": len(df),
        "dropped_rows": 0,
        "blank_columns": {},
        "warnings": [],
    }

    if df.empty:
        report["warnings"].append("Empty DataFrame — nothing to validate")
        return df, report

    critical = CRITICAL_COLUMNS.get(bridge, [])

    # Check which critical columns exist
    missing_cols = [c for c in critical if c not in df.columns]
    if missing_cols:
        report["warnings"].append(f"Missing critical columns: {missing_cols}")

    present_critical = [c for c in critical if c in df.columns]

    # Count blanks per column before dropping
    for col in present_critical:
        n_blank = df[col].isna().sum()
        if n_blank > 0:
            report["blank_columns"][col] = int(n_blank)

    # Drop rows where ANY critical column is blank
    initial_len = len(df)
    if present_critical:
        df = df.dropna(subset=present_critical)
    report["dropped_rows"] = initial_len - len(df)

    # Drop negative latency
    if "latency" in df.columns:
        bad_latency = (df["latency"] < 0).sum()
        if bad_latency > 0:
            df = df[df["latency"] >= 0]
            report["warnings"].append(f"Dropped {bad_latency} rows with negative latency")

    # Drop negative user_cost
    if "user_cost" in df.columns:
        bad_cost = (df["user_cost"] < 0).sum()
        if bad_cost > 0:
            df = df[df["user_cost"] >= 0]
            report["warnings"].append(f"Dropped {bad_cost} rows with negative user_cost")

    report["output_rows"] = len(df)
    return df, report


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class Pipeline:
    def __init__(self, dry_run: bool = False):
        self.dune = DuneFetcher(DUNE_API_KEY)
        self.alchemy = AlchemyFetcher(ALCHEMY_KEY)
        self.dry_run = dry_run
        self.reports = []

    def get_last_timestamp(self, bridge: str) -> int:
        """Get the latest src_timestamp from existing CSV."""
        csv_path = os.path.join(DATA_DIR, f"{bridge}_cleaned.csv")
        if not os.path.exists(csv_path):
            log.warning(f"[{bridge}] No existing CSV found at {csv_path}")
            return 0

        df = pd.read_csv(csv_path, usecols=["src_timestamp"])
        ts = pd.to_numeric(df["src_timestamp"], errors="coerce").dropna()
        ts = ts[ts > 1600000000]

        if ts.empty:
            return 0

        last = int(ts.max())
        log.info(
            f"[{bridge}] Last timestamp: {last} "
            f"({datetime.fromtimestamp(last, tz=timezone.utc).isoformat()})"
        )
        return last

    def get_existing_tail(self, bridge: str) -> pd.DataFrame:
        """Load last N rows from existing CSV for rolling feature context."""
        csv_path = os.path.join(DATA_DIR, f"{bridge}_cleaned.csv")
        if not os.path.exists(csv_path):
            return pd.DataFrame()

        df = pd.read_csv(csv_path)
        return df.tail(ROLLING_CONTEXT_ROWS)

    def run_bridge(self, bridge: str) -> dict:
        log.info(f"\n{'═' * 70}")
        log.info(f"  PIPELINE: {bridge.upper()}")
        log.info(f"{'═' * 70}")

        # 1. Determine time range
        last_ts = self.get_last_timestamp(bridge)
        if last_ts == 0:
            log.error(f"[{bridge}] Cannot determine start time — no existing data")
            return {"bridge": bridge, "status": "error", "reason": "no existing data"}

        start_ts = last_ts + 1
        now_ts = int(datetime.now(timezone.utc).timestamp())

        hours_gap = (now_ts - start_ts) / 3600
        if hours_gap < 1:
            log.info(f"[{bridge}] Data is already up-to-date (gap: {hours_gap:.1f}h)")
            return {"bridge": bridge, "status": "up_to_date"}

        log.info(f"[{bridge}] Fetching data for gap of {hours_gap:.1f} hours")

        # 2. Fetch bridge transactions from Dune
        new_df = self.dune.fetch_bridge_transactions(bridge, start_ts)

        if new_df.empty:
            log.warning(f"[{bridge}] No new transactions returned")
            return {"bridge": bridge, "status": "no_new_data"}

        # 3. Normalize timestamps
        new_df = normalize_timestamps(new_df)
        new_df["bridge"] = bridge

        # 4. Compute latency
        new_df = compute_latency(new_df)

        # 5. Compute user_cost
        new_df = compute_user_cost(new_df, bridge)

        # 6. Determine enrichment time range
        valid_ts = pd.to_numeric(new_df["src_timestamp"], errors="coerce").dropna()
        valid_ts = valid_ts[valid_ts > 1600000000]
        if valid_ts.empty:
            log.error(f"[{bridge}] No valid timestamps in new data")
            return {"bridge": bridge, "status": "error", "reason": "no valid timestamps"}

        enrich_start = pd.to_datetime(valid_ts.min(), unit="s", utc=True).floor("D") - pd.Timedelta(days=1)
        enrich_end = pd.to_datetime(valid_ts.max(), unit="s", utc=True).ceil("D") + pd.Timedelta(days=1)

        # 7. Fetch ETH prices
        price_df = self.alchemy.fetch_eth_prices(enrich_start, enrich_end)
        new_df = enrich_eth_prices(new_df, price_df)

        # 8. Fetch gas data
        gas_start = enrich_start.strftime("%Y-%m-%d %H:%M:%S")
        gas_end = enrich_end.strftime("%Y-%m-%d %H:%M:%S")
        gas_df = self.dune.fetch_gas_data(gas_start, gas_end)
        new_df = enrich_gas_data(new_df, gas_df)

        # 9. Compute features (with context from existing data)
        existing_tail = self.get_existing_tail(bridge)
        new_df = compute_features(new_df, existing_tail)

        # 10. Validate
        new_df, report = validate_dataframe(new_df, bridge)
        self.reports.append(report)

        log.info(f"[{bridge}] Validation: {report['input_rows']} in → {report.get('output_rows', 0)} out")
        if report["dropped_rows"] > 0:
            log.warning(f"[{bridge}] Dropped {report['dropped_rows']} rows with blank critical columns")
        if report["blank_columns"]:
            log.warning(f"[{bridge}] Blank columns before cleanup: {report['blank_columns']}")
        for w in report.get("warnings", []):
            log.warning(f"[{bridge}] {w}")

        if new_df.empty:
            log.warning(f"[{bridge}] No valid rows after validation")
            return {"bridge": bridge, "status": "all_filtered"}

        # 11. Append to CSV
        if self.dry_run:
            log.info(f"[{bridge}] DRY RUN — would append {len(new_df)} rows")
            log.info(f"[{bridge}] Sample columns: {list(new_df.columns)}")
            log.info(f"[{bridge}] Sample data:\n{new_df.head(2).to_string()}")
        else:
            csv_path = os.path.join(DATA_DIR, f"{bridge}_cleaned.csv")

            if os.path.exists(csv_path):
                existing_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()

                # Add any missing columns with NaN
                for col in existing_cols:
                    if col not in new_df.columns:
                        new_df[col] = np.nan

                # Reorder to match existing schema
                new_df = new_df.reindex(columns=existing_cols)

                new_df.to_csv(csv_path, mode="a", header=False, index=False)
                log.info(f"[{bridge}] Appended {len(new_df)} rows to {csv_path}")
            else:
                new_df.to_csv(csv_path, index=False)
                log.info(f"[{bridge}] Created new file {csv_path} with {len(new_df)} rows")

        return {"bridge": bridge, "status": "success", "rows_added": len(new_df)}

    def run_all(self, bridges: list[str] | None = None):
        if bridges is None:
            bridges = list(BRIDGE_QUERY_IDS.keys())

        log.info("=" * 70)
        log.info(f"  DATA PIPELINE — {datetime.now(timezone.utc).isoformat()}")
        log.info(f"  Bridges: {bridges}")
        log.info(f"  Dry run: {self.dry_run}")
        log.info("=" * 70)

        results = []
        for bridge in bridges:
            try:
                result = self.run_bridge(bridge)
                results.append(result)
            except Exception as e:
                log.error(f"[{bridge}] Pipeline failed: {e}", exc_info=True)
                results.append({"bridge": bridge, "status": "error", "reason": str(e)})

        # Summary
        log.info("\n" + "=" * 70)
        log.info("  PIPELINE SUMMARY")
        log.info("=" * 70)
        for r in results:
            status = r.get("status", "unknown")
            rows = r.get("rows_added", "-")
            reason = r.get("reason", "")
            extra = f" ({reason})" if reason else ""
            log.info(f"  {r['bridge']:<15} | {status:<15} | rows: {rows}{extra}")
        log.info("=" * 70)

        # Save reports
        report_path = os.path.join(
            LOG_DIR, f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, "w") as f:
            json.dump(self.reports, f, indent=2, default=str)
        log.info(f"Validation reports saved to {report_path}")

        return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bridge transaction data pipeline")
    parser.add_argument(
        "--bridge",
        choices=["across", "cctp", "stargate_bus", "stargate_oft"],
        help="Run pipeline for a specific bridge only",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate without saving to CSVs",
    )
    args = parser.parse_args()

    pipeline = Pipeline(dry_run=args.dry_run)

    bridges = [args.bridge] if args.bridge else None
    pipeline.run_all(bridges)


if __name__ == "__main__":
    main()
