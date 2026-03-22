# Cross-Chain Bridge Cost Prediction System — v2

End-to-end ML platform: data enrichment → EDA → per-bridge models → backend API (live quotes + predictions) → frontend → real-time retraining pipeline → deployment.

## Current State

| Item | Status |
|------|--------|
| Raw dataset | [all_ccc_sampled.csv](file:///home/hp/Documents/web3/all_ccc_sampled.csv) — 10K rows, 73 cols, May–Dec 2024 |
| Cleaned data | `cleaned_split_data/` — per-bridge CSVs with enriched features |
| Bridges | Across (3,411), Stargate Bus (3,130), OFT (2,898), CCTP (519), CCIP (11) |
| Target | `user_cost` (USD) — decomposed into gas + spread + operator |
| ML Models | **Trained** — Stargate OFT R²=0.78, CCTP R²=0.71, Across R²=0.59, Bus R²=0.38, CCIP=median |
| Frontend | Next.js 16, wired to `/quotes` and `/predict` — **working** |
| Backend | FastAPI — **working** with live quotes (Across + LiFi) + ML predictions |
| Deployment | Render (Docker) + Vercel configs ready |

---

## User Review Required

> [!IMPORTANT]
> **Enriching the dataset with real gas data**: We'll use the **Etherscan Gas Oracle API** (free, 5 req/s) to back-fill hourly gas prices for each transaction's timestamp. This populates `dune_hourly_gas_gwei`, `gas_1h_lag`, `gas_6h_avg`, `gas_24h_avg`, and `gas_volatility_24h` with real values. ETH price data will come from **CoinGecko's free API**.

> [!IMPORTANT]
> **Alternative ML models**: If XGBoost R² < 0.6, we'll try **LightGBM** and **LSTM**. See [Research Papers](#research-papers) section below.

> [!WARNING]
> **CCIP (13 samples)**: Median-based fallback — insufficient data for ML training.

> [!WARNING]
> **Deployment**: Backend on **Render** (free Docker), frontend on **Vercel** (free). Render free tier spins down after inactivity (~50s cold start). We can consider **Railway** if spin-down is unacceptable. ML model is deployed with the backend (serialized via joblib).

---

## Proposed Changes

### Phase 1: Data Enrichment

#### [NEW] [enrich_dataset.py](file:///home/hp/Documents/web3/backend/enrich_dataset.py)

Enriches [all_ccc_sampled.csv](file:///home/hp/Documents/web3/all_ccc_sampled.csv) with **real-world** gas and ETH price data:

1. **Gas prices** — Etherscan gas oracle historical data (`gastracker` module) → populates:
   - `dune_hourly_gas_gwei`, `gas_1h_lag`, `gas_6h_avg`, `gas_24h_avg`, `gas_volatility_24h`
2. **ETH price** — CoinGecko `/coins/ethereum/market_chart/range` → populates:
   - `eth_price_at_src`, `eth_price_change_1h`, `eth_price_24h_avg`
3. **Temporal features** from `src_timestamp`:
   - `hour_of_day`, `day_of_week`, `is_weekend`, `month`
4. **Route + token encoding**:
   - `route` = `src_blockchain→dst_blockchain`
   - `bridge_hourly_volume` = count of txs per bridge per hour window
5. Output: `all_ccc_enriched.csv`

---

### Phase 2: EDA (added to notebook)

#### [MODIFY] [Dataset_Cleaned (2).ipynb](file:///home/hp/Documents/web3/Dataset_Cleaned%20(2).ipynb)

New cells appended per bridge:

| Analysis | Details |
|----------|---------|
| Cost distributions | Histograms + box plots of `user_cost` per bridge |
| Fee decomposition | Stacked bar: gas vs spread vs operator per bridge |
| Time patterns | Hour-of-day and day-of-week cost heatmaps |
| Route analysis | Top 10 routes by volume and average cost |
| Correlation matrix | Feature vs `user_cost` correlations |
| Outlier analysis | IQR-based detection, before/after cleaning stats |
| Amount vs cost | Scatter plots with regression lines |

---

### Phase 3: Prediction Models

#### [NEW] [train_models.py](file:///home/hp/Documents/web3/backend/train_models.py)

Per-bridge model training pipeline:

- **Target**: `log1p(user_cost)` — handles skewed distribution
- **Primary model**: XGBoost (tuned per bridge)
- **Fallback models**: LightGBM, Random Forest
- **Features**: `amount_usd`, `route_encoded`, `src_symbol_encoded`, `hour_of_day`, `day_of_week`, `is_weekend`, `month`, `dune_hourly_gas_gwei`, `gas_1h_lag`, `gas_6h_avg`, `gas_24h_avg`, `gas_volatility_24h`, `eth_price_at_src`, `eth_price_change_1h`, `bridge_hourly_volume`, `latency`
- **Split**: Time-based 80/20 (no future leakage)
- **Evaluation**: MAE, RMSE, R² per bridge; compare XGBoost vs LightGBM
- **Output**: `backend/models/{bridge}_model.joblib` + `{bridge}_meta.json`

---

### Phase 4: FastAPI Backend

#### [NEW] [main.py](file:///home/hp/Documents/web3/backend/main.py)

| Endpoint | Description |
|----------|-------------|
| `GET /quotes` | Live bridge quotes (fee + time + breakdown) |
| `GET /predict` | ML predictions per bridge with confidence |
| `GET /eda` | Pre-computed EDA stats |
| `POST /retrain` | Trigger model retraining with new data |

#### [NEW] [bridge_apis.py](file:///home/hp/Documents/web3/backend/bridge_apis.py)

Live quote fetching (verified working ✅):

| Bridge | Source | Status |
|--------|--------|--------|
| **Across** | `https://app.across.to/api/suggested-fees` | ✅ Working — returns LP/relayer/gas fee breakdown |
| **Stargate** | Stargate V2 API (`mainnet.stargate-api.com`) + on-chain `quoteOFT()`/`quoteSend()` | ✅ Metadata working, quotes via RPC |
| **CCTP** | LiFi aggregator API (`li.quest/v1/quote`) | ✅ Working via LiFi |
| **CCIP** | On-chain `getFee()` via RPC | ⚠️ On-chain only |

#### [NEW] [predictor.py](file:///home/hp/Documents/web3/backend/predictor.py)

Model loading, prediction, confidence scoring, fee decomposition.

---

### Phase 5: Real-Time Data Pipeline

#### [NEW] [data_pipeline.py](file:///home/hp/Documents/web3/backend/data_pipeline.py)

Continuous data fetching + model retraining pipeline:

```
[Dune Analytics API] ─── Query bridge tx data ───┐
[Etherscan Gas Oracle] ─── Current gas prices ────┤──> Clean → Enrich → Append to CSV → Retrain
[CoinGecko API] ─── ETH price data ──────────────┘
```

1. **Fetch new transactions** from Dune Analytics (SQL queries for each bridge)
2. **Enrich** with gas + ETH price data
3. **Append** to `all_ccc_enriched.csv`
4. **Retrain** models if >100 new rows accumulated
5. **Schedule**: Runs daily via cron or on-demand via `/retrain` endpoint

---

### Phase 6: Frontend Updates

#### [MODIFY] [page.js](file:///home/hp/Documents/web3/frontend/app/page.js)

- EDA visualizations section (dynamic charts from `/eda` endpoint)
- Enhanced prediction panel with fee decomposition (gas vs spread vs operator)
- "Predicted vs Live" comparison view
- Model info badges (last trained, accuracy, sample count)

#### [MODIFY] [globals.css](file:///home/hp/Documents/web3/frontend/app/globals.css)

- EDA chart styles, enhanced prediction cards

---

### Phase 7: Deployment

#### [NEW] [Dockerfile](file:///home/hp/Documents/web3/backend/Dockerfile)

Python 3.11, FastAPI, uvicorn, trained models baked in.

#### [NEW] [render.yaml](file:///home/hp/Documents/web3/render.yaml)

Render deployment config. ML model ships with the Docker image and is retrainable via API.

---

## Research Papers

If XGBoost accuracy is low, these papers propose alternative approaches:

| Paper | Model(s) | Relevance |
|-------|----------|-----------|
| [Blockchain Transaction Fee Forecasting (2023)](https://arxiv.org/abs/2301.13714) | Direct Recursive Hybrid LSTM, CNN-LSTM, Attention LSTM | Gas fee prediction using wavelet denoising + matrix profile. Hybrid models excel for short lookaheads. |
| [An Empirical Study on Cross-chain Transactions (2025)](https://cczgroup.github.io/) | N/A (empirical) | Foundational cost decomposition across bridges — directly relevant to fee structure understanding. |
| [Optimizing Cost-Efficient Payment Transactions (2026)](https://www.preprints.org/) | Embedding-based AI routing | AI-driven routing to reduce cross-chain costs via path optimization. |
| [Ethereum Gas Price Prediction with GBM (2023)](https://doi.org/10.3390/math11092230) | LightGBM, CatBoost | Gradient boosting for Ethereum gas — relevant as gas is major cost component. |

---

## Verification Plan

### Automated Tests

1. **Enrichment** — Verify enriched CSV has filled gas/price columns:
   ```bash
   python3 backend/enrich_dataset.py && python3 -c "import pandas as pd; df=pd.read_csv('all_ccc_enriched.csv'); print(df[['dune_hourly_gas_gwei','eth_price_at_src']].describe())"
   ```

2. **Model training** — Train and check R² > 0.5 per bridge:
   ```bash
   python3 backend/train_models.py
   ```

3. **API test** — Backend endpoints return valid JSON:
   ```bash
   curl "http://localhost:8000/predict?source_chain=Ethereum&dest_chain=Arbitrum&token=USDC&amount=1000000000"
   curl "http://localhost:8000/quotes?source_chain=Ethereum&dest_chain=Arbitrum&token=USDC&amount=1000000000"
   ```

4. **Frontend build** — `npm run build` passes

### Browser Test

5. **E2E** — Select chains, click Compare, verify live quotes + predictions render
