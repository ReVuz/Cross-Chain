-- ============================================================================
-- Circle CCTP Bridge Transfers
-- ============================================================================
-- Parameter: start_date (text) — e.g. '2024-12-01 00:00:00'
--
-- CCTP burns USDC on source chain and mints on destination chain.
-- Protocol fee for standard transfers is 0; fee = gas cost only.
--
-- NOTE: Table names may vary. Check Dune schema browser if this fails.
--       Search for: cctp, TokenMessenger, DepositForBurn, MintAndWithdraw
-- ============================================================================

WITH burns AS (
    SELECT
        evt_tx_hash                                 AS src_tx_hash,
        evt_block_time                              AS src_block_time,
        CAST(nonce AS VARCHAR)                      AS deposit_id,
        burnToken                                   AS token_address,
        CAST(amount AS DOUBLE) / 1e6                AS amount,
        depositor                                   AS depositor,
        mintRecipient                                AS recipient,
        CAST(destinationDomain AS VARCHAR)          AS dst_domain
    FROM cctp_ethereum.TokenMessenger_evt_DepositForBurn
    WHERE evt_block_time >= CAST('{{start_date}}' AS TIMESTAMP)
),

domain_names AS (
    SELECT 0 AS domain_id, 'ethereum'  AS chain_name UNION ALL
    SELECT 1 AS domain_id, 'avalanche' AS chain_name UNION ALL
    SELECT 2 AS domain_id, 'optimism'  AS chain_name UNION ALL
    SELECT 3 AS domain_id, 'arbitrum'  AS chain_name UNION ALL
    SELECT 6 AS domain_id, 'base'      AS chain_name UNION ALL
    SELECT 7 AS domain_id, 'polygon'   AS chain_name
)

SELECT
    b.src_tx_hash,
    TO_UNIXTIME(b.src_block_time)                   AS src_timestamp,
    'ethereum'                                       AS src_blockchain,
    COALESCE(dn.chain_name, b.dst_domain)           AS dst_blockchain,
    'USDC'                                           AS token_symbol,
    b.amount,
    b.amount * p.price                               AS amount_usd,
    -- Src gas fee
    tx.gas_used * tx.gas_price / 1e18 * ep.price    AS src_fee_usd,
    b.deposit_id,
    b.depositor,
    b.recipient
FROM burns b
LEFT JOIN domain_names dn
    ON CAST(b.dst_domain AS BIGINT) = dn.domain_id
LEFT JOIN prices.usd p
    ON p.symbol = 'USDC'
    AND p.blockchain = 'ethereum'
    AND p.minute = DATE_TRUNC('minute', b.src_block_time)
LEFT JOIN prices.usd ep
    ON ep.symbol = 'WETH'
    AND ep.blockchain = 'ethereum'
    AND ep.minute = DATE_TRUNC('minute', b.src_block_time)
LEFT JOIN ethereum.transactions tx
    ON tx.hash = b.src_tx_hash
    AND tx.block_time >= CAST('{{start_date}}' AS TIMESTAMP)
ORDER BY b.src_block_time
