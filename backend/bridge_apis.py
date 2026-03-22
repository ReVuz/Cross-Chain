"""
Live bridge quote fetching.

- Across: Native suggested-fees API (detailed breakdown)
- CCTP, Stargate, CCIP, deBridge: LiFi aggregator (advanced/routes)
"""

import asyncio
import logging

import httpx

log = logging.getLogger(__name__)

CHAIN_IDS = {
    "ethereum": 1,
    "arbitrum": 42161,
    "optimism": 10,
    "base": 8453,
    "polygon": 137,
}

USDC_ADDRESSES = {
    1: "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    42161: "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
    10: "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85",
    8453: "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
    137: "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359",
}

USDC_DECIMALS = 6
# LiFi validates addresses; use a well-known EOA
QUOTE_ADDRESS = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"


# ---------------------------------------------------------------------------
# Across Protocol — native API
# ---------------------------------------------------------------------------

async def _get_across_quote(
    src_chain_id: int, dst_chain_id: int, amount_raw: int, timeout: int = 15
) -> dict | None:
    src_token = USDC_ADDRESSES.get(src_chain_id)
    dst_token = USDC_ADDRESSES.get(dst_chain_id)
    if not src_token or not dst_token:
        return None

    params = {
        "inputToken": src_token,
        "outputToken": dst_token,
        "originChainId": src_chain_id,
        "destinationChainId": dst_chain_id,
        "amount": str(amount_raw),
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(
                "https://app.across.to/api/suggested-fees", params=params
            )
            resp.raise_for_status()
            data = resp.json()

        total_relay = int(data.get("totalRelayFee", {}).get("total", "0"))
        lp_fee_pct = float(data.get("lpFee", {}).get("pct", "0")) / 1e18
        relay_gas = int(data.get("relayerGasFee", {}).get("total", "0"))
        relay_cap = int(data.get("relayerCapitalFee", {}).get("total", "0"))

        amount_usd = amount_raw / 10**USDC_DECIMALS
        fee_usd = total_relay / 10**USDC_DECIMALS

        return {
            "protocol": "Across",
            "normalized_usd_fee": round(fee_usd, 6),
            "estimated_time_seconds": data.get("estimatedFillTimeSec", 30),
            "fee_breakdown": [
                {
                    "name": "LP Fee",
                    "usd": round(lp_fee_pct * amount_usd, 6),
                    "description": "Liquidity provider fee",
                },
                {
                    "name": "Relayer Gas",
                    "usd": round(relay_gas / 10**USDC_DECIMALS, 6),
                    "description": "Destination gas cost paid by relayer",
                },
                {
                    "name": "Capital Fee",
                    "usd": round(relay_cap / 10**USDC_DECIMALS, 6),
                    "description": "Capital lockup cost for relayer",
                },
            ],
        }
    except Exception as e:
        log.error(f"Across API error: {e}")
        return None


# ---------------------------------------------------------------------------
# LiFi aggregator — covers CCTP, Stargate, CCIP, deBridge
# ---------------------------------------------------------------------------

_RELEVANT_TOOLS = {
    "across", "cctp", "celercircle", "celercirclefast",
    "stargatev2", "stargatev2bus", "stargate",
    "ccip", "dln", "debridge",
}


async def _get_lifi_quotes(
    src_chain_id: int, dst_chain_id: int, amount_raw: int, timeout: int = 25
) -> list[dict]:
    src_token = USDC_ADDRESSES.get(src_chain_id)
    dst_token = USDC_ADDRESSES.get(dst_chain_id)
    if not src_token or not dst_token:
        return []

    quotes: list[dict] = []

    body = {
        "fromChainId": src_chain_id,
        "toChainId": dst_chain_id,
        "fromTokenAddress": src_token,
        "toTokenAddress": dst_token,
        "fromAmount": str(amount_raw),
        "fromAddress": QUOTE_ADDRESS,
        "options": {"order": "CHEAPEST"},
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                "https://li.quest/v1/advanced/routes", json=body
            )
            if resp.status_code == 200:
                seen: set[str] = set()
                for route in resp.json().get("routes", []):
                    tool = (route.get("steps") or [{}])[0].get("tool", "")
                    if tool.lower() not in _RELEVANT_TOOLS:
                        continue
                    q = _parse_lifi_route(route, amount_raw)
                    if q and q["protocol"] not in seen:
                        seen.add(q["protocol"])
                        quotes.append(q)
    except Exception as e:
        log.error(f"LiFi routes error: {e}")

    if not quotes:
        try:
            params = {
                "fromChain": src_chain_id,
                "toChain": dst_chain_id,
                "fromToken": src_token,
                "toToken": dst_token,
                "fromAmount": str(amount_raw),
                "fromAddress": QUOTE_ADDRESS,
                "order": "CHEAPEST",
            }
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get("https://li.quest/v1/quote", params=params)
                if resp.status_code == 200:
                    q = _parse_lifi_single(resp.json(), amount_raw)
                    if q:
                        quotes.append(q)
        except Exception as e:
            log.error(f"LiFi single-quote error: {e}")

    return quotes


def _parse_lifi_route(route: dict, amount_raw: int) -> dict | None:
    try:
        steps = route.get("steps", [])
        if not steps:
            return None

        from_amount = int(route.get("fromAmount", amount_raw))
        to_amount = int(route.get("toAmount", amount_raw))
        fee_usd = max((from_amount - to_amount) / 10**USDC_DECIMALS, 0)

        gas_usd = float(route.get("gasCostUSD") or 0)
        tool = steps[0].get("tool", "")
        bridge_name = _normalise_name(tool)

        exec_time = sum(
            s.get("estimate", {}).get("executionDuration", 60) for s in steps
        )

        breakdown = []
        if gas_usd > 0:
            breakdown.append(
                {"name": "Gas Cost", "usd": round(gas_usd, 6), "description": "Source chain gas fees"}
            )
        protocol_fee = fee_usd - gas_usd
        if protocol_fee > 0.0001:
            breakdown.append(
                {"name": "Bridge Fee", "usd": round(protocol_fee, 6), "description": "Protocol fee + spread"}
            )

        return {
            "protocol": bridge_name,
            "normalized_usd_fee": round(fee_usd, 6),
            "estimated_time_seconds": exec_time or 120,
            "fee_breakdown": breakdown,
        }
    except Exception as e:
        log.error(f"LiFi route parse error: {e}")
        return None


def _parse_lifi_single(data: dict, amount_raw: int) -> dict | None:
    try:
        estimate = data.get("estimate", {})
        from_amount = int(estimate.get("fromAmount", amount_raw))
        to_amount = int(estimate.get("toAmount", amount_raw))
        fee_usd = max((from_amount - to_amount) / 10**USDC_DECIMALS, 0)

        gas_costs = estimate.get("gasCosts", [])
        gas_usd = sum(float(g.get("amountUSD", "0")) for g in gas_costs)

        fee_costs = estimate.get("feeCosts", [])
        protocol_usd = sum(float(f.get("amountUSD", "0")) for f in fee_costs)

        tool = data.get("tool", "")
        bridge_name = _normalise_name(tool)

        breakdown = []
        if gas_usd > 0:
            breakdown.append(
                {"name": "Gas Cost", "usd": round(gas_usd, 6), "description": "Source chain gas"}
            )
        if protocol_usd > 0:
            breakdown.append(
                {"name": "Protocol Fee", "usd": round(protocol_usd, 6), "description": "Bridge protocol fee"}
            )
        spread = fee_usd - gas_usd - protocol_usd
        if spread > 0.0001:
            breakdown.append(
                {"name": "Spread", "usd": round(spread, 6), "description": "Liquidity spread"}
            )

        return {
            "protocol": bridge_name,
            "normalized_usd_fee": round(fee_usd, 6),
            "estimated_time_seconds": estimate.get("executionDuration", 300),
            "fee_breakdown": breakdown,
        }
    except Exception as e:
        log.error(f"LiFi single parse error: {e}")
        return None


def _normalise_name(tool: str) -> str:
    t = tool.lower()
    if "celercirclefast" in t:
        return "CCTP (Fast)"
    if "celercircle" in t or "cctp" in t:
        return "CCTP (Standard)"
    if "stargateV2Bus" in tool or "stargatev2bus" in t:
        return "Stargate V2 (Bus)"
    if "stargate" in t:
        return "Stargate V2"
    if "ccip" in t:
        return "CCIP"
    if "debridge" in t or "dln" in t:
        return "deBridge"
    if "across" in t:
        return "Across"
    return tool or "Unknown"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def get_all_quotes(
    source_chain: str, dest_chain: str, token: str, amount_raw: int
) -> list[dict]:
    src_id = CHAIN_IDS.get(source_chain.lower())
    dst_id = CHAIN_IDS.get(dest_chain.lower())
    if not src_id or not dst_id:
        return []

    across_result, lifi_results = await asyncio.gather(
        _get_across_quote(src_id, dst_id, amount_raw),
        _get_lifi_quotes(src_id, dst_id, amount_raw),
    )

    quotes: list[dict] = []
    if across_result:
        quotes.append(across_result)
    quotes.extend(lifi_results)

    seen: set[str] = set()
    unique: list[dict] = []
    for q in quotes:
        if q["protocol"] not in seen:
            seen.add(q["protocol"])
            unique.append(q)
    return unique
