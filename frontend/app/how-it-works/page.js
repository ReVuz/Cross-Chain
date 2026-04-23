"use client";

import Image from "next/image";
import { useState } from "react";

const SECTIONS = [
  { id: "overview", label: "Overview" },
  { id: "architecture", label: "Architecture" },
  { id: "pipeline", label: "Data Pipeline" },
  { id: "methodology", label: "ML Methodology" },
  { id: "features", label: "Features" },
  { id: "performance", label: "Model Performance" },
  { id: "bridges", label: "Supported Bridges" },
  // { id: "references", label: "References" },
];

function Figure({ src, alt, caption }) {
  const [loaded, setLoaded] = useState(false);
  return (
    <figure className="doc-figure">
      <div className={`doc-figure-frame ${loaded ? "loaded" : ""}`}>
        <Image
          src={src}
          alt={alt}
          width={1400}
          height={900}
          className="doc-figure-img"
          onLoad={() => setLoaded(true)}
          sizes="(max-width: 900px) 100vw, 900px"
        />
      </div>
      {caption && <figcaption className="doc-figure-caption">{caption}</figcaption>}
    </figure>
  );
}

function Callout({ title, children, tone = "info" }) {
  return (
    <div className={`doc-callout doc-callout-${tone}`}>
      <div className="doc-callout-title">{title}</div>
      <div className="doc-callout-body">{children}</div>
    </div>
  );
}

export default function HowItWorksPage() {
  return (
    <div className="docs-container">
      <aside className="docs-sidebar">
        <div className="docs-sidebar-inner">
          <div className="docs-sidebar-label">On this page</div>
          <nav className="docs-toc">
            {SECTIONS.map((s) => (
              <a key={s.id} href={`#${s.id}`} className="docs-toc-link">
                {s.label}
              </a>
            ))}
          </nav>
        </div>
      </aside>

      <main className="docs-main">
        <header className="docs-header">
          <div className="header-label">
            <span className="dot" />
            System Documentation
          </div>
          <h1>How BridgeCompare Works</h1>
          <p className="docs-subtitle">
            A full-stack machine-learning platform that combines live bridge
            quotes with historical transaction data to predict cross-chain
            transfer costs across five major protocols.
          </p>
        </header>

        {/* Overview */}
        <section id="overview" className="docs-section">
          <h2>1. Overview</h2>
          <p>
            Cross-chain bridges don&apos;t charge a single fixed fee &mdash; the cost a
            user pays is the sum of source gas, relayer spread, liquidity
            provider margin, and destination-gas reimbursement. These
            components fluctuate with network congestion, token volatility, and
            liquidity depth, so quoted fees can differ by 3&ndash;10x across
            protocols at any moment.
          </p>
          <p>
            BridgeCompare solves this in two complementary ways:
          </p>
          <ul className="docs-list">
            <li>
              <strong>Live quoting.</strong> The backend queries the Across API
              and the LiFi aggregator (for Stargate V2, CCTP, CCIP, deBridge)
              in parallel and returns a normalized USD fee plus a
              component-level breakdown for each bridge.
            </li>
            <li>
              <strong>ML prediction.</strong> Per-bridge XGBoost / LightGBM
              models trained on 354,897 augmented transactions (from 9,964 seed records) predict the
              expected end-user cost, so users can tell whether a current quote
              is reasonable compared to typical fees on that route.
            </li>
          </ul>

          <div className="stat-row">
            <div className="stat-chip"><span>354,897</span>Training transactions</div>
            <div className="stat-chip"><span>5</span>Bridge protocols</div>
            <div className="stat-chip"><span>20</span>Engineered features</div>
            <div className="stat-chip"><span>R² = 0.917</span>Best bridge model</div>
          </div>
        </section>

        {/* Architecture */}
        <section id="architecture" className="docs-section">
          <h2>2. System Architecture</h2>
          <p>
            The system is split into a Next.js frontend and a FastAPI backend.
            The backend wraps three responsibilities behind a single REST API:
            live-quote fetching, ML inference, and training-data collection.
          </p>

          <Figure
            src="/docs/architecture_diagram.png"
            alt="High-level system architecture showing frontend, backend, bridge APIs, and ML models"
            caption="Figure 1 — High-level architecture. The Next.js frontend talks to a FastAPI backend, which fans out to bridge APIs for live quotes and loads serialized XGBoost/LightGBM models for predictions."
          />

          <h3>Request flow</h3>
          <ol className="docs-list docs-list-ordered">
            <li>The user selects source chain, destination chain, token, and amount in the frontend.</li>
            <li>The frontend dispatches two parallel calls: <code>GET /quotes</code> and <code>GET /predict</code>.</li>
            <li><code>/quotes</code> queries bridge APIs concurrently (Across, LiFi) and normalizes every fee to USD.</li>
            <li><code>/predict</code> loads the per-bridge joblib artifacts and returns expected fees with confidence bands.</li>
            <li>Each live quote is appended to <code>training_data.csv</code>, growing the dataset for future retraining.</li>
          </ol>

          <Callout title="Why a per-bridge model?" tone="info">
            Each bridge has a fundamentally different fee structure &mdash;
            Across is a relayer market, Stargate is liquidity-pool priced, CCTP
            is Circle&apos;s flat burn-and-mint, CCIP is oracle-secured with
            fixed premiums. A single unified model would under-fit all of
            them; per-bridge models isolate each distribution and yield
            materially higher R².
          </Callout>
        </section>

        {/* Pipeline */}
        <section id="pipeline" className="docs-section">
          <h2>3. Data Pipeline</h2>
          <p>
            The training pipeline enriches raw transactions with market data
            and temporal features before per-bridge splitting and model
            training. Data flows continuously: every live quote served by the
            API is also persisted, so the corpus grows with usage.
          </p>

          <Figure
            src="/docs/pipeline_architecture.png"
            alt="Data pipeline diagram: raw CSV -> enrichment -> per-bridge split -> training -> serialized models"
            caption="Figure 2 — Data pipeline. Raw bridge transactions are enriched with Etherscan gas and CoinGecko ETH price data, split per bridge, trained independently, and serialized for online inference."
          />

          <h3>Stages</h3>
          <ol className="docs-list docs-list-ordered">
            <li>
              <strong>Ingestion.</strong> Historical transactions (<code>all_ccc_sampled.csv</code>,
              9,964 seed rows, May&ndash;Dec 2024) augmented to 354,897 samples
              via continuous live data collection from the Across API and LiFi aggregator.
            </li>
            <li>
              <strong>Enrichment.</strong> Each row is joined against hourly
              Ethereum gas prices (Etherscan Gas Oracle) and ETH/USD (CoinGecko)
              at its timestamp.
            </li>
            <li>
              <strong>Feature engineering.</strong> Temporal signals
              (<code>hour_sin</code>, <code>hour_cos</code>, <code>is_weekend</code>),
              gas lag/volatility windows, and amount-gas interaction terms.
            </li>
            <li>
              <strong>Per-bridge split.</strong> Saved under <code>cleaned_split_data/</code>.
              Fees are capped at the 99th percentile per bridge to suppress
              outliers.
            </li>
            <li>
              <strong>Training &amp; selection.</strong> XGBoost and LightGBM
              are trained in parallel; the better of the two (by R²) is
              persisted with its metadata.
            </li>
            <li>
              <strong>Online retraining.</strong> The <code>POST /model/retrain</code>
              endpoint rebuilds models on demand from the latest CSV state.
            </li>
          </ol>
        </section>

        {/* Methodology */}
        <section id="methodology" className="docs-section">
          <h2>4. ML Methodology</h2>
          <p>
            We treat fee prediction as a regression problem with a log-transformed
            target and a time-based split, which avoids future leakage and
            reflects how the model will be used in production (predicting an
            unknown future cost from today&apos;s market state).
          </p>

          <Figure
            src="/docs/methodology_architecture.png"
            alt="ML methodology overview: feature engineering, log-transform target, time-split, XGBoost/LightGBM selection"
            caption="Figure 3 — Modeling methodology. Log1p target smooths the heavy-tailed fee distribution; a time-based 80/20 split keeps evaluation honest; and the best of XGBoost vs LightGBM is selected per bridge."
          />

          <div className="method-grid">
            <div className="method-card">
              <h4>Target</h4>
              <code>log1p(user_cost)</code>
              <p>Bridge fees are heavy-tailed (a few whale transfers cost 100x the median). A log transform compresses that tail so the gradient-boosted learner isn&apos;t dominated by outliers.</p>
            </div>
            <div className="method-card">
              <h4>Split</h4>
              <code>time-based 80/20</code>
              <p>Rows sorted by <code>src_timestamp</code>; the oldest 80% trains, the newest 20% tests. Prevents the model from peeking at future gas shocks.</p>
            </div>
            <div className="method-card">
              <h4>Outlier handling</h4>
              <code>clip at p99 per bridge</code>
              <p>Extreme fees (often reverted or MEV-inflated) are capped so the loss surface isn&apos;t distorted.</p>
            </div>
            <div className="method-card">
              <h4>Model selection</h4>
              <code>best of {"{XGBoost, LightGBM}"}</code>
              <p>Both are trained and the one with higher held-out R² wins. Empirically XGBoost wins on Stargate OFT and CCTP; LightGBM on Across.</p>
            </div>
          </div>
        </section>

        {/* Features */}
        <section id="features" className="docs-section">
          <h2>5. Feature Set</h2>
          <p>
            Each per-bridge model consumes 20 features that span transaction,
            market, and temporal dimensions. Feature-importance analysis (below)
            shows that the log-amount, Ethereum gas price, and the route
            encoding dominate the predictions &mdash; which matches intuition:
            fees scale roughly with size, spike with L1 congestion, and are
            route-specific.
          </p>

          <Figure
            src="/docs/feature_importance_xgboost.png"
            alt="Bar chart of feature importances from the XGBoost models, dominated by log_amount, gas, and route features"
            caption="Figure 4 — Feature importance aggregated across per-bridge XGBoost models. Amount magnitude and Ethereum gas dominate, followed by route and temporal signals."
          />

          <div className="feature-cats">
            <div className="feature-cat">
              <h4>Transaction</h4>
              <ul>
                <li><code>amount_usd</code>, <code>log_amount</code></li>
                <li><code>src_symbol</code> (token)</li>
                <li><code>route</code> (<code>src→dst</code> pair)</li>
              </ul>
            </div>
            <div className="feature-cat">
              <h4>Market</h4>
              <ul>
                <li><code>dune_hourly_gas_gwei</code></li>
                <li><code>gas_1h_lag</code>, <code>gas_6h_avg</code>, <code>gas_24h_avg</code></li>
                <li><code>gas_volatility_24h</code></li>
                <li><code>eth_price_at_src</code>, <code>eth_price_change_1h</code></li>
              </ul>
            </div>
            <div className="feature-cat">
              <h4>Temporal</h4>
              <ul>
                <li><code>hour_of_day</code>, <code>hour_sin</code>, <code>hour_cos</code></li>
                <li><code>day_of_week</code>, <code>is_weekend</code>, <code>month</code></li>
              </ul>
            </div>
            <div className="feature-cat">
              <h4>Engineered</h4>
              <ul>
                <li><code>amount_x_gas</code> (interaction)</li>
                <li><code>bridge_hourly_volume</code></li>
                <li><code>latency</code> (historical, excluded at inference)</li>
              </ul>
            </div>
          </div>

          <Figure
            src="/docs/feature_correlation_per_bridge.png"
            alt="Heatmap of feature-to-user-cost correlations per bridge"
            caption="Figure 5 — Per-bridge feature correlation with user cost. Correlation structure differs noticeably between Across and Stargate, reinforcing the per-bridge modeling choice."
          />
        </section>

        {/* Performance */}
        <section id="performance" className="docs-section">
          <h2>6. Model Performance</h2>
          <p>
            Model quality is evaluated on a held-out, time-forward test split.
            We report three metrics: R² (variance explained), MAE (typical
            dollar error), and the empirical residual distribution.
          </p>

          <Figure
            src="/docs/actual_vs_predicted.png"
            alt="Actual vs predicted fee scatter plot with diagonal reference line"
            caption="Figure 6 — Actual vs predicted fee on held-out data. Points hug the diagonal across three orders of magnitude in transfer size, with small systematic underestimation at the extreme tail."
          />

          <Figure
            src="/docs/prediction_error_distribution.png"
            alt="Histogram of prediction residuals roughly centered around zero"
            caption="Figure 7 — Residual distribution. Errors are approximately zero-centered with a narrow central mass, confirming the log-transform was effective."
          />

          <div className="perf-table-wrap">
            <table className="perf-table">
              <thead>
                <tr>
                  <th>Bridge</th>
                  <th>Best model</th>
                  <th>R²</th>
                  <th>Samples</th>
                  <th>Notes</th>
                </tr>
              </thead>
              <tbody>
                <tr><td>Across</td><td>XGBoost</td><td>0.917</td><td>354,897</td><td>Best performer; expanded dataset via live collection</td></tr>
                <tr><td>Stargate OFT</td><td>XGBoost</td><td>0.465</td><td>2,898</td><td>Moderate; volume-driven fee structure</td></tr>
                <tr><td>Stargate Bus</td><td>XGBoost</td><td>0.382</td><td>3,126</td><td>Low; batch composition is unpredictable</td></tr>
                <tr><td>CCTP</td><td>XGBoost</td><td>0.025</td><td>519</td><td>Low; limited data, gas-dominated fees</td></tr>
                <tr><td>CCIP</td><td>Median fallback</td><td>&mdash;</td><td>11</td><td>Insufficient data for ML</td></tr>
              </tbody>
            </table>
          </div>

          <Callout title="How to read the confidence badge" tone="info">
            The <code>/predict</code> endpoint returns a <code>confidence</code>
            of <em>high</em>, <em>medium</em>, or <em>low</em> derived from the
            model&apos;s R² and the number of training samples on that route.
            Low-confidence predictions are visually dimmed in the UI so users
            know to trust the live quote over the model.
          </Callout>
        </section>

        {/* Bridges */}
        <section id="bridges" className="docs-section">
          <h2>7. Supported Bridges</h2>
          <p>
            BridgeCompare currently covers five protocols that together
            represent the majority of USDC cross-chain volume. Each protocol
            has a distinct fee mechanic reflected in both its live quote path
            and its trained model.
          </p>

          <div className="bridge-grid">
            <div className="bridge-info">
              <h4>Across</h4>
              <p className="bridge-arch">Optimistic relayer market</p>
              <p>Relayers front liquidity on the destination chain and are reimbursed on the source. Fees = LP fee + relayer capital fee + destination gas.</p>
            </div>
            <div className="bridge-info">
              <h4>Stargate V2 (OFT)</h4>
              <p className="bridge-arch">LayerZero Omnichain Fungible Token</p>
              <p>Direct cross-chain transfers priced against unified liquidity pools. Fees depend on pool depth and LayerZero messaging cost.</p>
            </div>
            <div className="bridge-info">
              <h4>Stargate V2 (Bus)</h4>
              <p className="bridge-arch">Batched LayerZero mode</p>
              <p>Multiple transfers ride a single cross-chain message. Cheaper per-user but introduces wait-time until the bus departs.</p>
            </div>
            <div className="bridge-info">
              <h4>CCTP</h4>
              <p className="bridge-arch">Circle native burn-and-mint</p>
              <p>Burns USDC on source, attests via Circle, mints on destination. No wrapped tokens and no liquidity provider spread.</p>
            </div>
            <div className="bridge-info">
              <h4>CCIP</h4>
              <p className="bridge-arch">Chainlink oracle network</p>
              <p>Oracle-secured cross-chain messaging with defense-in-depth. Institutional-grade but higher premium.</p>
            </div>
          </div>

          <Figure
            src="/docs/stargate_latency_analysis.png"
            alt="Box plot of Stargate transfer latency by route"
            caption="Figure 8 — Stargate latency analysis. The long upper tail on bus transfers reflects batching delay and motivates modeling OFT and Bus as separate bridges."
          />

          <Figure
            src="/docs/across_variable_proof.png"
            alt="Scatter showing Across fee varying with amount and gas"
            caption="Figure 9 — Empirical evidence that Across fees genuinely depend on transfer size and gas conditions (not flat pricing), justifying per-bridge ML."
          />

          <Figure
            src="/docs/cost_evolution_analysis.png"
            alt="Time series of average bridge fees over several months"
            caption="Figure 10 — Fee evolution over time. Shifting levels motivate the retraining endpoint and the &quot;recent median&quot; fallback used for routes with very fresh data."
          />
        </section>

        {/* References */}
        {/* <section id="references" className="docs-section">
          <h2>8. References &amp; Data Sources</h2>

          <h3>Data sources</h3>
          <div className="doc-sources">
            <a className="doc-source" href="https://app.across.to/api/deposits" target="_blank" rel="noreferrer">
              <div className="doc-source-name">Across Protocol API</div>
              <div className="doc-source-desc">Historical filled deposits + live suggested fees</div>
            </a>
            <a className="doc-source" href="https://li.quest/v1/advanced/routes" target="_blank" rel="noreferrer">
              <div className="doc-source-name">LiFi Aggregator</div>
              <div className="doc-source-desc">Quotes for CCTP, Stargate V2, CCIP, deBridge</div>
            </a>
            <a className="doc-source" href="https://docs.etherscan.io/api-endpoints/gas-tracker" target="_blank" rel="noreferrer">
              <div className="doc-source-name">Etherscan Gas Oracle</div>
              <div className="doc-source-desc">Ethereum gas price history for enrichment</div>
            </a>
            <a className="doc-source" href="https://www.coingecko.com/en/api" target="_blank" rel="noreferrer">
              <div className="doc-source-name">CoinGecko</div>
              <div className="doc-source-desc">ETH/USD price series</div>
            </a>
          </div> */}

          {/* <h3>Academic background</h3>
          <div className="doc-sources">
            <a className="doc-source" href="https://arxiv.org/abs/2301.13714" target="_blank" rel="noreferrer">
              <div className="doc-source-name">Blockchain Transaction Fee Forecasting (2023)</div>
              <div className="doc-source-desc">Hybrid LSTM / CNN-LSTM / Attention-LSTM approaches to gas fee prediction.</div>
            </a>
            <a className="doc-source" href="https://cczgroup.github.io/" target="_blank" rel="noreferrer">
              <div className="doc-source-name">Empirical Study on Cross-chain Transactions (2025)</div>
              <div className="doc-source-desc">Cost decomposition across bridges &mdash; foundational for our fee breakdown.</div>
            </a>
            <a className="doc-source" href="https://doi.org/10.3390/math11092230" target="_blank" rel="noreferrer">
              <div className="doc-source-name">Ethereum Gas Price Prediction with GBM (2023)</div>
              <div className="doc-source-desc">LightGBM / CatBoost for gas &mdash; supports the gradient-boosting choice here.</div>
            </a>
          </div> */}

          {/* <h3>Internal documents</h3>
          <div className="doc-sources">
            <a className="doc-source" href="https://github.com/" target="_blank" rel="noreferrer">
              <div className="doc-source-name">README.md</div>
              <div className="doc-source-desc">Setup, API reference, and deployment instructions.</div>
            </a>
            <a className="doc-source" href="https://github.com/" target="_blank" rel="noreferrer">
              <div className="doc-source-name">implementation_plan.md</div>
              <div className="doc-source-desc">Phase-by-phase development plan with status per component.</div>
            </a>
          </div> */}
        {/* </section> */}

        <footer className="docs-footer">
          Questions or corrections? The source data, notebooks, and training
          scripts all ship with the repository.
        </footer>
      </main>
    </div>
  );
}
