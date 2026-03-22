# Cross-Chain Bridge Cost Prediction

A full-stack ML system that fetches live bridge quotes, predicts transaction costs using per-bridge models, and lets users compare fees across five major cross-chain protocols.

## Supported Bridges

| Protocol | Architecture | Model |
|----------|-------------|-------|
| **Across** | Optimistic relayer — relayers front liquidity, reimbursed on source chain | LightGBM (R²=0.64) |
| **Stargate V2 (OFT)** | LayerZero Omnichain Fungible Token — individual cross-chain transfers | XGBoost (R²=0.47) |
| **Stargate V2 (Bus)** | LayerZero batched mode — multiple transfers in one cross-chain message | XGBoost (R²=0.38) |
| **CCTP** | Circle's native USDC burn-and-mint — no wrapped tokens | XGBoost |
| **CCIP** | Chainlink oracle-secured bridge — institutional-grade transfers | Median fallback (11 samples) |

## Architecture

```
┌─────────────────────┐      ┌──────────────────────────────────┐
│   Next.js Frontend  │◄────►│        FastAPI Backend           │
│                     │      │                                  │
│  • Bridge Comparator│      │  GET  /quotes    ← live fees     │
│  • Predictions View │      │  GET  /predict   ← ML inference  │
│  • Training Data    │      │  GET  /eda       ← dataset stats │
│    Dashboard        │      │  GET  /data/*    ← recent rows   │
│                     │      │  GET  /model/*   ← model info    │
└─────────────────────┘      │  POST /model/retrain             │
                             │                                  │
                             │  bridge_apis.py  → Across API    │
                             │                  → LiFi (CCTP,   │
                             │                    Stargate,CCIP)│
                             │  predictor.py    → joblib models │
                             │  train_models.py → XGBoost/LGBM  │
                             └──────────────────────────────────┘
```

## Project Structure

```
web3/
├── backend/
│   ├── main.py              # FastAPI app — all endpoints
│   ├── bridge_apis.py       # Live quote fetching (Across + LiFi aggregator)
│   ├── predictor.py         # Model loading and inference
│   ├── train_models.py      # Per-bridge training pipeline
│   ├── data_pipeline.py     # Live data collection to CSV
│   ├── fetch_recent_data.py # Bulk historical data fetcher
│   ├── models/              # Trained model artifacts (.joblib + meta.json)
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   └── app/
│       ├── page.js          # Main comparison page
│       ├── data/page.js     # Training data dashboard
│       ├── globals.css
│       └── layout.js
├── cleaned_split_data/      # Per-bridge cleaned CSVs for training
├── all_ccc_sampled.csv      # Raw dataset (10K rows, May–Dec 2024)
├── Dataset_Cleaned (2).ipynb # EDA notebook
├── render.yaml              # Render deployment config
└── .env.example
```

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+

### Backend

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

# Train models (required on first run)
python -m backend.train_models

# Start API server
uvicorn backend.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend expects the backend at `http://localhost:8000` by default. Set `NEXT_PUBLIC_API_URL` in `frontend/.env.local` to override.

### Environment Variables

Copy `.env.example` to `.env` and fill in:

| Variable | Required | Description |
|----------|----------|-------------|
| `ETHERSCAN_API_KEY` | No | Etherscan Gas Oracle — falls back to defaults |
| `CACHE_TTL_SECONDS` | No | Quote cache TTL (default: 30s) |
| `NEXT_PUBLIC_API_URL` | For prod | Backend URL for the frontend |

## ML Pipeline

### Features

Each model uses 20 features including:

- **Transaction**: `amount_usd`, `src_symbol`, `route`
- **Market**: `dune_hourly_gas_gwei`, `eth_price_at_src`, gas lag/avg/volatility
- **Temporal**: `hour_of_day`, `day_of_week`, `is_weekend`, `month`
- **Engineered**: `log_amount`, `amount_x_gas`, `hour_sin`, `hour_cos`

### Training

- **Target**: `log1p(user_cost)` to handle skewed fee distributions
- **Split**: Time-based 80/20 (no future leakage)
- **Outlier handling**: Fees capped at 99th percentile per bridge
- **Evaluation**: MAE and R² on held-out test set; best of XGBoost vs LightGBM is saved

### Data Collection

```bash
# Fetch recent real transactions (Across) + market quotes (LiFi)
python -m backend.fetch_recent_data

# Fetch only Across deposits
python -m backend.fetch_recent_data across

# Fetch only LiFi quotes (CCTP, Stargate, CCIP)
python -m backend.fetch_recent_data lifi

# Retrain after new data
python -m backend.train_models
```

## API Reference

All endpoints accept query parameters.

| Endpoint | Method | Parameters | Description |
|----------|--------|------------|-------------|
| `/quotes` | GET | `source_chain`, `dest_chain`, `token`, `amount` | Live quotes from bridge APIs |
| `/predict` | GET | `source_chain`, `dest_chain`, `token`, `amount` | ML cost predictions per bridge |
| `/eda` | GET | — | Dataset statistics and distributions |
| `/data/recent` | GET | `bridge`, `limit` | Recent data rows (live + seed) |
| `/data/stats` | GET | — | Per-bridge row counts and date ranges |
| `/model/status` | GET | — | Model performance metrics |
| `/model/retrain` | POST | — | Trigger model retraining |

### Example

```bash
# Live quotes: 1000 USDC from Ethereum to Arbitrum
curl "http://localhost:8000/quotes?source_chain=Ethereum&dest_chain=Arbitrum&token=USDC&amount=1000000000"

# ML predictions for the same transfer
curl "http://localhost:8000/predict?source_chain=Ethereum&dest_chain=Arbitrum&token=USDC&amount=1000000000"
```

## Deployment

### Backend (Render)

The `render.yaml` deploys the backend as a Docker service. Models are trained at build time inside the container.

```bash
# Build locally to test
docker build -f backend/Dockerfile -t bridgecompare-api .
docker run -p 8000:8000 bridgecompare-api
```

### Frontend (Vercel)

```bash
cd frontend
npx vercel --prod
```

Set `NEXT_PUBLIC_API_URL` to your Render backend URL in the Vercel environment settings.

## Data Sources

| Source | What it provides |
|--------|-----------------|
| [Across Protocol API](https://app.across.to/api/deposits) | Real filled deposit history + suggested fees |
| [LiFi Aggregator](https://li.quest/v1/advanced/routes) | Quotes for CCTP, Stargate V2, CCIP routes |
| [Etherscan Gas Oracle](https://api.etherscan.io/api?module=gastracker&action=gasoracle) | Current Ethereum gas prices |
| [CoinGecko](https://api.coingecko.com/api/v3/simple/price) | ETH/USD price |

## License

MIT
