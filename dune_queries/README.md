# Dune SQL Query Templates for Bridge Data Pipeline

## Setup Instructions

For each bridge below:
1. Go to [dune.com](https://dune.com) and log in
2. Click "New Query"
3. Paste the SQL from the corresponding `.sql` file
4. Click the `{ }` Parameters button and add the parameter: `start_date` (type: text)
5. Save the query and note the **Query ID** from the URL (e.g., `https://dune.com/queries/1234567` → ID is `1234567`)
6. Put the query ID into `data_pipeline.py` → `BRIDGE_QUERY_IDS`

## Important Notes

- These queries use Dune's **Spellbook** decoded tables. Table names may change over time.
- Test each query manually on Dune first with a recent `start_date` (e.g., `2024-12-01 00:00:00`)
- If a table doesn't exist, search Dune's schema browser for the correct name
- All queries return timestamps as Unix seconds and amounts in USD using `prices.usd`
- The `start_date` parameter filters for transactions AFTER this timestamp

## Files

| File | Bridge | Expected Rows/Day |
|------|--------|-------------------|
| `across.sql` | Across V3 | ~15-50 |
| `cctp.sql` | Circle CCTP | ~5-20 |
| `stargate_bus.sql` | Stargate V2 Bus | ~10-40 |
| `stargate_oft.sql` | Stargate V2 OFT | ~10-40 |

## Verifying Table Names

If queries fail, use Dune's schema browser to find the correct tables:
- Search for `across` → look for `SpokePool` event tables
- Search for `cctp` → look for `TokenMessenger` event tables  
- Search for `stargate` → look for `Pool`, `OFT`, `TokenMessaging` tables
- Search for `layerzero` → look for `Endpoint` tables

The `prices.usd` table is a stable Dune Spellbook table and unlikely to change.
