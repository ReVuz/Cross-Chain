-- ============================================================================
-- Stargate V2 OFT (Omnichain Fungible Token) Transfers
-- ============================================================================
-- Parameter: start_date (text) — e.g. '2024-12-01 00:00:00'
--
-- Stargate OFT transfers use the LayerZero OFT standard.
-- Fees include: executor_fee + dvn_fee (LayerZero infrastructure).
--
-- NOTE: Table names may vary. Check Dune schema browser if this fails.
--       Search for: stargate, OFTSent, layerzero
-- ============================================================================

WITH sends AS (
    SELECT
        evt_tx_hash                                  AS src_tx_hash,
        evt_block_time                               AS src_block_time,
        CAST("guid" AS VARCHAR)                      AS transfer_guid,
        CAST(dstEid AS VARCHAR)                      AS dst_eid,
        "to"                                         AS recipient,
        CAST(amountSentLD AS DOUBLE)                 AS amount_sent_raw,
        CAST(amountReceivedLD AS DOUBLE)             AS amount_received_raw
    FROM stargate_v2_ethereum.OFT_evt_OFTSent
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
    s.src_tx_hash,
    TO_UNIXTIME(s.src_block_time)                    AS src_timestamp,
    'ethereum'                                        AS src_blockchain,
    COALESCE(ec.chain_name, s.dst_eid)               AS dst_blockchain,
    -- Token detection: adjust based on which Stargate pool this is
    'ETH'                                             AS token_symbol,
    s.amount_sent_raw / 1e18                          AS amount,
    (s.amount_sent_raw / 1e18) * p.price              AS amount_usd,
    (s.amount_received_raw / 1e18) * p.price          AS amount_received_usd,
    -- Src gas fee
    tx.gas_used * tx.gas_price / 1e18 * p.price       AS src_fee_usd,
    s.transfer_guid                                    AS deposit_id,
    s.recipient
FROM sends s
LEFT JOIN eid_to_chain ec
    ON CAST(s.dst_eid AS BIGINT) = ec.eid
LEFT JOIN prices.usd p
    ON p.symbol = 'WETH'
    AND p.blockchain = 'ethereum'
    AND p.minute = DATE_TRUNC('minute', s.src_block_time)
LEFT JOIN ethereum.transactions tx
    ON tx.hash = s.src_tx_hash
    AND tx.block_time >= CAST('{{start_date}}' AS TIMESTAMP)
ORDER BY s.src_block_time
