-- ============================================================================
-- Stargate V2 Bus (Batched) Transfers
-- ============================================================================
-- Parameter: start_date (text) — e.g. '2024-12-01 00:00:00'
--
-- Bus mode batches multiple user transfers into a single LayerZero message.
-- Users pay: user_fee (submission gas) + share of bus_fare.
-- Operator pays: executor_fee + dvn_fee for the batch.
--
-- NOTE: Table names may vary. Check Dune schema browser if this fails.
--       Search for: stargate, BusRode, TokenMessaging, bus
-- ============================================================================

WITH rides AS (
    SELECT
        evt_tx_hash                                   AS user_tx_hash,
        evt_block_time                                AS src_block_time,
        CAST("guid" AS VARCHAR)                       AS bus_guid,
        CAST(dstEid AS VARCHAR)                       AS dst_eid,
        passenger,
        CAST(ticketId AS VARCHAR)                     AS bus_ticket_id,
        CAST(amountSD AS DOUBLE)                      AS amount_sd,
        CAST(fare AS DOUBLE)                          AS bus_fare_raw
    FROM stargate_v2_ethereum.TokenMessaging_evt_BusRode
    WHERE evt_block_time >= CAST('{{start_date}}' AS TIMESTAMP)
),

eid_to_chain AS (
    SELECT 30101 AS eid, 'ethereum'  AS chain_name UNION ALL
    SELECT 30110 AS eid, 'arbitrum'  AS chain_name UNION ALL
    SELECT 30111 AS eid, 'optimism'  AS chain_name UNION ALL
    SELECT 30109 AS eid, 'polygon'   AS chain_name UNION ALL
    SELECT 30184 AS eid, 'base'      AS chain_name UNION ALL
    SELECT 30106 AS eid, 'avalanche' AS chain_name UNION ALL
    SELECT 30102 AS eid, 'bsc'       AS chain_name
)

SELECT
    r.user_tx_hash                                    AS src_tx_hash,
    TO_UNIXTIME(r.src_block_time)                     AS src_timestamp,
    'ethereum'                                         AS src_blockchain,
    COALESCE(ec.chain_name, r.dst_eid)                AS dst_blockchain,
    'USDC'                                             AS token_symbol,
    -- Stargate SD (shared decimals) = 6 for USDC pools
    r.amount_sd / 1e6                                  AS amount_sent,
    (r.amount_sd / 1e6) * p.price                      AS amount_sent_usd,
    -- Bus fare in native token (ETH)
    r.bus_fare_raw / 1e18                              AS bus_fare,
    (r.bus_fare_raw / 1e18) * ep.price                 AS bus_fare_usd,
    -- User gas fee for the ride submission
    tx.gas_used * tx.gas_price / 1e18 * ep.price       AS user_fee_usd,
    r.bus_guid,
    r.passenger,
    r.bus_ticket_id
FROM rides r
LEFT JOIN eid_to_chain ec
    ON CAST(r.dst_eid AS BIGINT) = ec.eid
LEFT JOIN prices.usd p
    ON p.symbol = 'USDC'
    AND p.blockchain = 'ethereum'
    AND p.minute = DATE_TRUNC('minute', r.src_block_time)
LEFT JOIN prices.usd ep
    ON ep.symbol = 'WETH'
    AND ep.blockchain = 'ethereum'
    AND ep.minute = DATE_TRUNC('minute', r.src_block_time)
LEFT JOIN ethereum.transactions tx
    ON tx.hash = r.user_tx_hash
    AND tx.block_time >= CAST('{{start_date}}' AS TIMESTAMP)
ORDER BY r.src_block_time
