-- ============================================================================
-- Across V3 Bridge Transfers
-- ============================================================================
-- Parameter: start_date (text) — e.g. '2024-12-01 00:00:00'
--
-- Returns completed Across bridge deposits with matched fills.
-- Fee = input_amount - output_amount (the relayer spread).
--
-- NOTE: Table names may vary. Check Dune schema browser if this fails.
--       Search for: across, SpokePool, V3FundsDeposited, FilledV3Relay
-- ============================================================================

WITH deposits AS (
    SELECT
        evt_tx_hash                                  AS src_tx_hash,
        evt_block_time                               AS src_block_time,
        CAST(depositId AS VARCHAR)                   AS deposit_id,
        depositor,
        recipient,
        CAST(originChainId AS VARCHAR)               AS src_chain_id,
        CAST(destinationChainId AS VARCHAR)          AS dst_chain_id,
        inputToken                                   AS input_token,
        outputToken                                  AS output_token,
        CAST(inputAmount AS DOUBLE)                  AS input_amount_raw,
        CAST(outputAmount AS DOUBLE)                 AS output_amount_raw,
        quoteTimestamp                               AS quote_timestamp
    FROM across_v3_ethereum.SpokePool_evt_V3FundsDeposited
    WHERE evt_block_time >= CAST('{{start_date}}' AS TIMESTAMP)
),

fills AS (
    SELECT
        evt_tx_hash                                  AS dst_tx_hash,
        evt_block_time                               AS dst_block_time,
        CAST(depositId AS VARCHAR)                   AS deposit_id,
        relayer
    FROM across_v3_ethereum.SpokePool_evt_FilledV3Relay
    WHERE evt_block_time >= CAST('{{start_date}}' AS TIMESTAMP)
),

chain_names AS (
    SELECT 1    AS chain_id, 'ethereum'  AS chain_name UNION ALL
    SELECT 10   AS chain_id, 'optimism'  AS chain_name UNION ALL
    SELECT 137  AS chain_id, 'polygon'   AS chain_name UNION ALL
    SELECT 42161 AS chain_id, 'arbitrum' AS chain_name UNION ALL
    SELECT 8453  AS chain_id, 'base'     AS chain_name UNION ALL
    SELECT 324   AS chain_id, 'zksync'   AS chain_name UNION ALL
    SELECT 59144  AS chain_id, 'linea'   AS chain_name
),

matched AS (
    SELECT
        d.*,
        f.dst_tx_hash,
        f.dst_block_time,
        f.relayer,
        src_cn.chain_name AS src_chain,
        dst_cn.chain_name AS dst_chain
    FROM deposits d
    INNER JOIN fills f ON d.deposit_id = f.deposit_id
    LEFT JOIN chain_names src_cn ON CAST(d.src_chain_id AS BIGINT) = src_cn.chain_id
    LEFT JOIN chain_names dst_cn ON CAST(d.dst_chain_id AS BIGINT) = dst_cn.chain_id
)

SELECT
    m.src_tx_hash,
    m.dst_tx_hash,
    TO_UNIXTIME(m.src_block_time)                   AS src_timestamp,
    TO_UNIXTIME(m.dst_block_time)                   AS dst_timestamp,
    COALESCE(m.src_chain, m.src_chain_id)           AS src_blockchain,
    COALESCE(m.dst_chain, m.dst_chain_id)           AS dst_blockchain,
    t.symbol                                         AS src_symbol,
    m.input_amount_raw / POWER(10, t.decimals)      AS input_amount,
    m.output_amount_raw / POWER(10, t.decimals)     AS output_amount,
    (m.input_amount_raw / POWER(10, t.decimals)) * p.price   AS input_amount_usd,
    (m.output_amount_raw / POWER(10, t.decimals)) * p.price  AS output_amount_usd,
    -- Src gas fee in ETH × ETH price = USD
    tx.gas_used * tx.gas_price / 1e18 * ep.price    AS src_fee_usd,
    m.deposit_id,
    m.depositor,
    m.recipient,
    m.quote_timestamp,
    m.relayer                                        AS exclusive_relayer
FROM matched m
LEFT JOIN tokens.erc20 t
    ON t.contract_address = m.input_token
    AND t.blockchain = 'ethereum'
LEFT JOIN prices.usd p
    ON p.contract_address = m.input_token
    AND p.blockchain = 'ethereum'
    AND p.minute = DATE_TRUNC('minute', m.src_block_time)
LEFT JOIN prices.usd ep
    ON ep.symbol = 'WETH'
    AND ep.blockchain = 'ethereum'
    AND ep.minute = DATE_TRUNC('minute', m.src_block_time)
LEFT JOIN ethereum.transactions tx
    ON tx.hash = m.src_tx_hash
    AND tx.block_time >= CAST('{{start_date}}' AS TIMESTAMP)
ORDER BY m.src_block_time
