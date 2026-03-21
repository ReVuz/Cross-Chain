"use client";

import { useState, useCallback } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const CHAINS = ["Ethereum", "Arbitrum", "Optimism", "Base", "Polygon"];

const PROTOCOL_META = {
  Across: { icon: "A", desc: "Intent-based relayer bridge", color: "#2563eb" },
  "CCTP (Standard)": { icon: "C", desc: "Native burn/mint (standard)", color: "#db2777" },
  "CCTP (Fast)": { icon: "C", desc: "Native burn/mint (fast)", color: "#db2777" },
  CCIP: { icon: "⬡", desc: "Chainlink CCIP", color: "#059669" },
  Stargate: { icon: "S", desc: "Omnichain liquidity", color: "#7c3aed" },
  "Stargate V2": { icon: "S", desc: "LayerZero Stargate V2", color: "#7c3aed" },
  "Stargate V2 (Taxi)": { icon: "S", desc: "Stargate V2 instant", color: "#7c3aed" },
  "Stargate V2 (Bus)": { icon: "S", desc: "Stargate V2 batched", color: "#7c3aed" },
  "Stargate (Estimate)": { icon: "S", desc: "Stargate (estimated)", color: "#a78bfa" },
  deBridge: { icon: "D", desc: "DLN cross-chain solver", color: "#f59e0b" },
};

const BRIDGE_DISPLAY = {
  across: "Across",
  cctp: "CCTP",
  ccip: "CCIP",
  stargate: "Stargate",
  debridge: "deBridge",
};

function formatFee(usd) {
  if (usd === null || usd === undefined) return "—";
  if (usd === 0) return "FREE";
  if (usd < 0.001) return "<$0.001";
  if (usd < 1) return `$${usd.toFixed(4)}`;
  return `$${usd.toFixed(2)}`;
}

function formatTime(seconds) {
  if (seconds < 60) return `${seconds}s`;
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return secs > 0 ? `${mins}m ${secs}s` : `${mins}m`;
}

function ConfidenceBadge({ level }) {
  const colors = { high: "conf-high", medium: "conf-medium", low: "conf-low" };
  return (
    <span className={`conf-badge ${colors[level] || "conf-low"}`}>
      {level}
    </span>
  );
}

function FeeBreakdown({ breakdown }) {
  if (!breakdown || breakdown.length === 0) return null;

  return (
    <div className="fee-breakdown">
      <div className="fee-breakdown-label">Fee Split</div>
      <div className="fee-breakdown-items">
        {breakdown.map((item, i) => (
          <div key={i} className="fee-item" title={item.description}>
            <span className="fee-item-name">{item.name}</span>
            <span className="fee-item-value">
              {item.usd === null || item.usd === undefined
                ? "varies"
                : formatFee(item.usd)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function QuoteCard({ quote, cheapest, fastest }) {
  const [showBreakdown, setShowBreakdown] = useState(false);
  const meta = PROTOCOL_META[quote.protocol] || {
    icon: "?",
    desc: "Cross-chain bridge",
    color: "#7AAACE",
  };

  const isRecommended = cheapest || fastest;
  const tags = [];
  if (cheapest) tags.push("cheapest");
  if (fastest) tags.push("fastest");

  const hasBreakdown = quote.fee_breakdown && quote.fee_breakdown.length > 0;

  return (
    <div className={`quote-card ${isRecommended ? "recommended" : ""}`}>
      <div className="quote-card-main">
        <div className="protocol-id">
          <div
            className="protocol-avatar"
            style={{ borderColor: meta.color, color: meta.color }}
          >
            {meta.icon}
          </div>
          <div className="protocol-details">
            <h3>{quote.protocol}</h3>
            <p>{meta.desc}</p>
          </div>
        </div>

        <div className="metric">
          <span className="metric-label">Total Fee</span>
          <span
            className={`metric-value ${cheapest ? "highlight-fee" : ""}`}
          >
            {formatFee(quote.normalized_usd_fee)}
          </span>
        </div>

        <div className="metric">
          <span className="metric-label">Transfer Time</span>
          <span
            className={`metric-value ${fastest ? "highlight-time" : ""}`}
          >
            {formatTime(quote.estimated_time_seconds)}
          </span>
        </div>

        <div className="tags-wrap">
          {tags.map((t) => (
            <span key={t} className={`status-tag ${t}`}>
              {t === "cheapest" ? "Best Fee" : "Fastest"}
            </span>
          ))}
          {hasBreakdown && (
            <button
              className="breakdown-toggle"
              onClick={() => setShowBreakdown(!showBreakdown)}
            >
              {showBreakdown ? "Hide Fees" : "Fee Split"}
            </button>
          )}
        </div>
      </div>

      {showBreakdown && hasBreakdown && (
        <FeeBreakdown breakdown={quote.fee_breakdown} />
      )}
    </div>
  );
}

function PredictionPanel({ predictions }) {
  if (!predictions || predictions.length === 0) return null;

  function fmtDate(iso) {
    if (!iso) return null;
    try {
      return new Date(iso).toLocaleDateString(undefined, {
        month: "short", day: "numeric", year: "numeric",
      });
    } catch {
      return null;
    }
  }

  const hasRecentMedian = predictions.some(
    (p) => p.prediction_source === "recent_median"
  );

  return (
    <section className="prediction-panel">
      <div className="results-header">
        <h2 className="results-title">Fee Estimates</h2>
        <span className="badge-ml">Historical Data · XGBoost</span>
      </div>
      <p className="pred-disclaimer">
        Estimates based on past bridge transactions (bridge fee + source gas).{" "}
        {hasRecentMedian
          ? "Routes marked Recent use a time-weighted median of the last 90 days — more accurate than the model when fee levels have shifted."
          : "These routes use the ML model trained on historical data — actual fees may differ."
        }{" "}
        Always confirm with the live quotes above.
      </p>
      <div className="prediction-grid">
        {predictions.map((p) => {
          const trainedOn = fmtDate(p.last_trained);
          const isRecent = p.prediction_source === "recent_median";
          const maeStr = !isRecent && p.mae != null ? `±${formatFee(p.mae)}` : null;
          return (
            <div key={p.bridge} className={`prediction-card conf-card-${p.confidence}`}>
              <div className="pred-source-row">
                <h4>{BRIDGE_DISPLAY[p.bridge] || p.bridge}</h4>
                {isRecent ? (
                  <span className="pred-source-badge recent" title="Based on last 90 days of actual transactions">
                    Recent
                  </span>
                ) : (
                  <span className="pred-source-badge model" title="XGBoost model prediction from training data">
                    Model
                  </span>
                )}
              </div>
              <span className="pred-fee">{formatFee(p.predicted_fee_usd)}</span>
              {maeStr && (
                <span className="pred-mae" title="Mean Absolute Error on test data — typical prediction error">
                  {maeStr} typical error
                </span>
              )}
              <div className="pred-meta">
                <ConfidenceBadge level={p.confidence} />
                {p.model_r2 != null && !isRecent && (
                  <span className="pred-r2" title="R² fit on held-out test data">
                    R²={p.model_r2.toFixed(3)}
                  </span>
                )}
                <span className="pred-samples" title={isRecent ? "Recent samples used for this estimate" : "Total training samples"}>
                  {(p.n_samples ?? 0).toLocaleString()} {isRecent ? "recent" : "samples"}
                </span>
              </div>
              {trainedOn && !isRecent && (
                <span className="pred-trained">Trained {trainedOn}</span>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}

export default function Home() {
  const [sourceChain, setSourceChain] = useState("Ethereum");
  const [destChain, setDestChain] = useState("Arbitrum");
  const [amount, setAmount] = useState("1000");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [predictions, setPredictions] = useState(null);

  const handleCompare = useCallback(async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    setPredictions(null);

    const rawAmount = String(Math.floor(parseFloat(amount) * 1e6));
    const qs = `source_chain=${encodeURIComponent(sourceChain)}&dest_chain=${encodeURIComponent(destChain)}&token=USDC&amount=${rawAmount}`;

    try {
      const [quotesResp, predictResp] = await Promise.allSettled([
        fetch(`${API_BASE}/quotes?${qs}`),
        fetch(`${API_BASE}/predict?${qs}`),
      ]);

      if (quotesResp.status === "fulfilled" && quotesResp.value.ok) {
        const data = await quotesResp.value.json();
        setResult(data);
      } else if (quotesResp.status === "fulfilled") {
        const body = await quotesResp.value.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${quotesResp.value.status}`);
      } else {
        throw new Error("Failed to fetch quotes. Is the API running?");
      }

      if (predictResp.status === "fulfilled" && predictResp.value.ok) {
        const pdata = await predictResp.value.json();
        setPredictions(pdata.predictions || []);
      }
    } catch (err) {
      setError(err.message || "Failed to fetch quotes. Is the API running?");
    } finally {
      setLoading(false);
    }
  }, [sourceChain, destChain, amount]);

  const quotes = result?.quotes || [];
  const minFee = quotes.length
    ? Math.min(...quotes.map((q) => q.normalized_usd_fee))
    : 0;
  const minTime = quotes.length
    ? Math.min(...quotes.map((q) => q.estimated_time_seconds))
    : 0;


  return (
    <div className="app-container">
      <header className="header">
        <div className="header-label">
          <span className="dot" />
          Live Network Pricing
        </div>
        <h1>Bridge Compare</h1>
        <p>
          Analyze real-time transfer fees with detailed cost breakdowns
          across leading cross-chain protocols.
        </p>
      </header>

      <section className="config-panel">
        <div className="config-rows">
          <div className="config-group">
            <label htmlFor="source-chain">Origin Network</label>
            <select
              id="source-chain"
              className="input-field"
              value={sourceChain}
              onChange={(e) => setSourceChain(e.target.value)}
            >
              {CHAINS.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>

          <div className="config-group">
            <label htmlFor="dest-chain">Destination Network</label>
            <select
              id="dest-chain"
              className="input-field"
              value={destChain}
              onChange={(e) => setDestChain(e.target.value)}
            >
              {CHAINS.filter((c) => c !== sourceChain).map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>

          <div className="config-group">
            <label htmlFor="amount">Transfer Amount (USDC)</label>
            <input
              id="amount"
              type="number"
              className="input-field"
              min="1"
              step="1"
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
              placeholder="1000"
            />
          </div>
        </div>

        <div className="config-action">
          <button
            className="btn-primary"
            onClick={handleCompare}
            disabled={loading || !amount || sourceChain === destChain}
          >
            {loading ? "Scanning Bridges..." : "Compare Bridges"}
          </button>
        </div>
      </section>

      {error && <div className="error-msg">{error}</div>}

      {loading && (
        <div className="loading-view">
          <div className="loader" />
          <p>Querying bridge contracts and solver networks...</p>
        </div>
      )}

      {!loading && result && quotes.length > 0 && (
        <>
          <section className="results-panel">
            <div className="results-header">
              <h2 className="results-title">Available Routes</h2>
              <span
                className={
                  result.source === "cache" ? "badge-cached" : "badge-live"
                }
              >
                {result.source === "cache"
                  ? "Cached Quotes"
                  : "Live Data"}
              </span>
            </div>

            <div className="quote-list">
              {[...quotes]
                .sort((a, b) => a.normalized_usd_fee - b.normalized_usd_fee)
                .map((q) => (
                  <QuoteCard
                    key={q.protocol}
                    quote={q}
                    cheapest={q.normalized_usd_fee === minFee}
                    fastest={q.estimated_time_seconds === minTime}
                  />
                ))}
            </div>
          </section>

          {predictions && predictions.length > 0 && (
            <PredictionPanel predictions={predictions} />
          )}
        </>
      )}

      {!loading && !error && !result && (
        <div className="empty-state">
          Configure your transfer above to see real-time bridge quotes
          with detailed fee breakdowns.
        </div>
      )}

      <footer className="footer">
        <span>&copy; 2026 BridgeCompare</span>
        <span>
          Quotes are estimates based on current on-chain activity.
        </span>
      </footer>
    </div>
  );
}
