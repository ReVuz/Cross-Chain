"use client";

import { useState, useCallback, useEffect, useMemo } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  AreaChart, Area, Cell, ScatterChart, Scatter, ZAxis, Legend,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
} from "recharts";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const CHAINS = ["Ethereum", "Arbitrum", "Optimism", "Base", "Polygon"];

const BRIDGE_COLORS = {
  across: "#2563eb",
  cctp: "#db2777",
  ccip: "#059669",
  stargate_bus: "#7c3aed",
  stargate_oft: "#a78bfa",
};

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
  across: "Across", cctp: "CCTP", ccip: "CCIP",
  stargate: "Stargate", stargate_bus: "Stargate Bus",
  stargate_oft: "Stargate OFT", debridge: "deBridge",
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
              {item.usd === null || item.usd === undefined ? "varies" : formatFee(item.usd)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function QuoteCard({ quote, cheapest, fastest }) {
  const [showBreakdown, setShowBreakdown] = useState(false);
  const meta = PROTOCOL_META[quote.protocol] || { icon: "?", desc: "Cross-chain bridge", color: "#7AAACE" };
  const isRecommended = cheapest || fastest;
  const tags = [];
  if (cheapest) tags.push("cheapest");
  if (fastest) tags.push("fastest");
  const hasBreakdown = quote.fee_breakdown && quote.fee_breakdown.length > 0;

  return (
    <div className={`quote-card ${isRecommended ? "recommended" : ""}`}>
      <div className="quote-card-main">
        <div className="protocol-id">
          <div className="protocol-avatar" style={{ borderColor: meta.color, color: meta.color }}>
            {meta.icon}
          </div>
          <div className="protocol-details">
            <h3>{quote.protocol}</h3>
            <p>{meta.desc}</p>
          </div>
        </div>
        <div className="metric">
          <span className="metric-label">Total Fee</span>
          <span className={`metric-value ${cheapest ? "highlight-fee" : ""}`}>
            {formatFee(quote.normalized_usd_fee)}
          </span>
        </div>
        <div className="metric">
          <span className="metric-label">Transfer Time</span>
          <span className={`metric-value ${fastest ? "highlight-time" : ""}`}>
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
            <button className="breakdown-toggle" onClick={() => setShowBreakdown(!showBreakdown)}>
              {showBreakdown ? "Hide Fees" : "Fee Split"}
            </button>
          )}
        </div>
      </div>
      {showBreakdown && hasBreakdown && <FeeBreakdown breakdown={quote.fee_breakdown} />}
    </div>
  );
}

function PredictionPanel({ predictions }) {
  if (!predictions || predictions.length === 0) return null;
  function fmtDate(iso) {
    if (!iso) return null;
    try { return new Date(iso).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" }); }
    catch { return null; }
  }
  const hasRecentMedian = predictions.some((p) => p.prediction_source === "recent_median");
  return (
    <section className="prediction-panel">
      <div className="results-header">
        <h2 className="results-title">Fee Estimates</h2>
        <span className="badge-ml">Historical Data · XGBoost</span>
      </div>
      <p className="pred-disclaimer">
        Estimates based on past bridge transactions (bridge fee + source gas).{" "}
        {hasRecentMedian
          ? "Routes marked Recent use a time-weighted median of the last 90 days."
          : "These routes use the ML model trained on historical data — actual fees may differ."}{" "}
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
                  <span className="pred-source-badge recent" title="Based on last 90 days">Recent</span>
                ) : (
                  <span className="pred-source-badge model" title="XGBoost model prediction">Model</span>
                )}
              </div>
              <span className="pred-fee">{formatFee(p.predicted_fee_usd)}</span>
              {maeStr && <span className="pred-mae" title="Mean Absolute Error">{maeStr} typical error</span>}
              <div className="pred-meta">
                <ConfidenceBadge level={p.confidence} />
                {p.model_r2 != null && !isRecent && (
                  <span className="pred-r2" title="R² fit">R²={p.model_r2.toFixed(3)}</span>
                )}
                <span className="pred-samples" title={isRecent ? "Recent samples" : "Total training samples"}>
                  {(p.n_samples ?? 0).toLocaleString()} {isRecent ? "recent" : "samples"}
                </span>
              </div>
              {trainedOn && !isRecent && <span className="pred-trained">Trained {trainedOn}</span>}
            </div>
          );
        })}
      </div>
    </section>
  );
}

/* ─── Chart Tooltip ───────────────────────────────────────────────── */
function ChartTooltip({ active, payload, label, prefix = "$" }) {
  if (!active || !payload || !payload.length) return null;
  return (
    <div className="chart-tooltip">
      <p className="chart-tooltip-label">{label}</p>
      {payload.map((entry, i) => (
        <p key={i} style={{ color: entry.color || "var(--color-text-primary)" }}>
          {entry.name}: {prefix}{typeof entry.value === "number" ? entry.value.toFixed(4) : entry.value}
        </p>
      ))}
    </div>
  );
}

/* ─── Analytics Dashboard ─────────────────────────────────────────── */
function AnalyticsDashboard({ eda, modelStatus }) {
  const [selectedBridge, setSelectedBridge] = useState("across");

  const bridges = useMemo(() => eda ? Object.keys(eda.bridges) : [], [eda]);

  // Hourly cost data for the selected bridge
  const hourlyData = useMemo(() => {
    if (!eda?.bridges?.[selectedBridge]?.hourly_cost) return [];
    const hourly = eda.bridges[selectedBridge].hourly_cost;
    return Array.from({ length: 24 }, (_, h) => ({
      hour: `${h.toString().padStart(2, "0")}:00`,
      cost: hourly[String(h)] ?? 0,
    }));
  }, [eda, selectedBridge]);

  // Cost distribution across bridges
  const costDistribution = useMemo(() => {
    if (!eda?.bridges) return [];
    return Object.entries(eda.bridges).map(([bridge, data]) => ({
      bridge: BRIDGE_DISPLAY[bridge] || bridge,
      median: data.cost_stats?.["50%"] ?? 0,
      p25: data.cost_stats?.["25%"] ?? 0,
      p75: data.cost_stats?.["75%"] ?? 0,
      mean: data.cost_stats?.mean ?? 0,
      color: BRIDGE_COLORS[bridge] || "#7AAACE",
    }));
  }, [eda]);

  // Fee decomposition for selected bridge
  const feeDecomp = useMemo(() => {
    if (!eda?.bridges?.[selectedBridge]?.fee_decomposition) return [];
    const decomp = eda.bridges[selectedBridge].fee_decomposition;
    const labels = {
      adjusted_src_fee_usd: "Source Gas",
      adjusted_dst_fee_usd: "Destination Gas",
      operator_cost: "Operator Fee",
    };
    return Object.entries(decomp).map(([key, value]) => ({
      name: labels[key] || key,
      value: value,
      fill: key === "adjusted_src_fee_usd" ? "#2563eb"
        : key === "adjusted_dst_fee_usd" ? "#059669"
        : "#f59e0b",
    }));
  }, [eda, selectedBridge]);

  // Model performance radar
  const modelRadar = useMemo(() => {
    if (!modelStatus?.metrics) return [];
    return Object.entries(modelStatus.metrics).map(([bridge, m]) => ({
      bridge: BRIDGE_DISPLAY[bridge] || bridge,
      r2: Math.max(0, (m.r2 ?? 0) * 100),
      samples: Math.min(100, Math.log10(Math.max(1, m.n_samples ?? 0)) * 20),
      confidence: m.confidence === "high" ? 100 : m.confidence === "medium" ? 60 : 30,
    }));
  }, [modelStatus]);

  // Route distribution for selected bridge
  const routeData = useMemo(() => {
    if (!eda?.bridges?.[selectedBridge]?.top_routes) return [];
    const routes = eda.bridges[selectedBridge].top_routes;
    return Object.entries(routes).map(([route, count]) => ({
      route: route.length > 20 ? route.slice(0, 18) + "…" : route,
      count,
    })).slice(0, 6);
  }, [eda, selectedBridge]);

  if (!eda) return null;

  return (
    <section className="analytics-dashboard">
      <div className="analytics-header">
        <div>
          <h2 className="analytics-title">
            <span className="analytics-icon">📊</span>
            Network Analytics
          </h2>
          <p className="analytics-subtitle">
            Historical cost patterns, fee decomposition, and model performance across {bridges.length} bridge protocols
          </p>
        </div>
        <div className="bridge-selector">
          {bridges.map((b) => (
            <button
              key={b}
              className={`bridge-pill ${selectedBridge === b ? "active" : ""}`}
              onClick={() => setSelectedBridge(b)}
              style={selectedBridge === b ? { background: BRIDGE_COLORS[b] || "#7AAACE", borderColor: BRIDGE_COLORS[b] || "#7AAACE" } : {}}
            >
              {BRIDGE_DISPLAY[b] || b}
            </button>
          ))}
        </div>
      </div>

      {/* Summary stat cards */}
      <div className="analytics-stats">
        {eda.bridges[selectedBridge] && (
          <>
            <div className="analytics-stat-card">
              <span className="analytics-stat-label">Transactions</span>
              <span className="analytics-stat-value">{eda.bridges[selectedBridge].n_rows?.toLocaleString()}</span>
            </div>
            <div className="analytics-stat-card">
              <span className="analytics-stat-label">Median Cost</span>
              <span className="analytics-stat-value highlight-green">
                {formatFee(eda.bridges[selectedBridge].cost_stats?.["50%"])}
              </span>
            </div>
            <div className="analytics-stat-card">
              <span className="analytics-stat-label">Mean Cost</span>
              <span className="analytics-stat-value">
                {formatFee(eda.bridges[selectedBridge].cost_stats?.mean)}
              </span>
            </div>
            <div className="analytics-stat-card">
              <span className="analytics-stat-label">Amount→Cost Corr.</span>
              <span className="analytics-stat-value">
                {eda.bridges[selectedBridge].amount_cost_corr?.correlation?.toFixed(3) ?? "—"}
              </span>
            </div>
          </>
        )}
      </div>

      {/* Charts grid */}
      <div className="charts-grid">
        {/* Hourly cost pattern */}
        <div className="chart-card chart-wide">
          <div className="chart-card-header">
            <h3>Hourly Cost Pattern (UTC)</h3>
            <span className="chart-badge">Median Fee</span>
          </div>
          <div className="chart-body">
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={hourlyData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="costGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={BRIDGE_COLORS[selectedBridge] || "#7AAACE"} stopOpacity={0.3} />
                    <stop offset="100%" stopColor={BRIDGE_COLORS[selectedBridge] || "#7AAACE"} stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(53,88,114,0.08)" />
                <XAxis dataKey="hour" tick={{ fontSize: 11, fill: "#7AAACE" }} interval={2} />
                <YAxis tick={{ fontSize: 11, fill: "#7AAACE" }} tickFormatter={(v) => `$${v.toFixed(2)}`} width={55} />
                <Tooltip content={<ChartTooltip />} />
                <Area
                  type="monotone" dataKey="cost" name="Median Cost"
                  stroke={BRIDGE_COLORS[selectedBridge] || "#7AAACE"}
                  fill="url(#costGrad)" strokeWidth={2}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Fee decomposition */}
        <div className="chart-card">
          <div className="chart-card-header">
            <h3>Fee Decomposition</h3>
            <span className="chart-badge">Median</span>
          </div>
          <div className="chart-body">
            {feeDecomp.length > 0 ? (
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={feeDecomp} layout="vertical" margin={{ top: 10, right: 20, left: 10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(53,88,114,0.08)" horizontal={false} />
                  <XAxis type="number" tick={{ fontSize: 11, fill: "#7AAACE" }} tickFormatter={(v) => `$${v.toFixed(3)}`} />
                  <YAxis dataKey="name" type="category" tick={{ fontSize: 12, fill: "#355872" }} width={100} />
                  <Tooltip content={<ChartTooltip />} />
                  <Bar dataKey="value" name="Fee (USD)" radius={[0, 6, 6, 0]} maxBarSize={28}>
                    {feeDecomp.map((entry, i) => (
                      <Cell key={i} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="chart-empty">No decomposition data available</div>
            )}
          </div>
        </div>

        {/* Cross-bridge cost comparison */}
        <div className="chart-card">
          <div className="chart-card-header">
            <h3>Cost Comparison</h3>
            <span className="chart-badge">All Bridges</span>
          </div>
          <div className="chart-body">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={costDistribution} margin={{ top: 10, right: 20, left: 10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(53,88,114,0.08)" />
                <XAxis dataKey="bridge" tick={{ fontSize: 11, fill: "#355872" }} />
                <YAxis tick={{ fontSize: 11, fill: "#7AAACE" }} tickFormatter={(v) => `$${v.toFixed(2)}`} width={55} />
                <Tooltip content={<ChartTooltip />} />
                <Bar dataKey="median" name="Median Cost" radius={[6, 6, 0, 0]} maxBarSize={40}>
                  {costDistribution.map((entry, i) => (
                    <Cell key={i} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Top routes */}
        <div className="chart-card chart-wide">
          <div className="chart-card-header">
            <h3>Popular Routes</h3>
            <span className="chart-badge">{BRIDGE_DISPLAY[selectedBridge] || selectedBridge}</span>
          </div>
          <div className="chart-body">
            {routeData.length > 0 ? (
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={routeData} layout="vertical" margin={{ top: 10, right: 20, left: 10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(53,88,114,0.08)" horizontal={false} />
                  <XAxis type="number" tick={{ fontSize: 11, fill: "#7AAACE" }} />
                  <YAxis dataKey="route" type="category" tick={{ fontSize: 11, fill: "#355872" }} width={160} />
                  <Tooltip />
                  <Bar dataKey="count" name="Transactions" fill={BRIDGE_COLORS[selectedBridge] || "#7AAACE"} radius={[0, 6, 6, 0]} maxBarSize={22} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="chart-empty">No route data available</div>
            )}
          </div>
        </div>

        {/* Model performance */}
        {modelRadar.length > 0 && (
          <div className="chart-card">
            <div className="chart-card-header">
              <h3>Model Performance</h3>
              <span className="chart-badge">XGBoost</span>
            </div>
            <div className="chart-body">
              <ResponsiveContainer width="100%" height={240}>
                <RadarChart outerRadius={80} data={modelRadar}>
                  <PolarGrid stroke="rgba(53,88,114,0.1)" />
                  <PolarAngleAxis dataKey="bridge" tick={{ fontSize: 11, fill: "#355872" }} />
                  <PolarRadiusAxis tick={{ fontSize: 9, fill: "#7AAACE" }} domain={[0, 100]} />
                  <Radar name="R² Score" dataKey="r2" stroke="#2563eb" fill="#2563eb" fillOpacity={0.15} strokeWidth={2} />
                  <Radar name="Confidence" dataKey="confidence" stroke="#059669" fill="#059669" fillOpacity={0.1} strokeWidth={2} />
                  <Legend iconSize={8} wrapperStyle={{ fontSize: "0.75rem" }} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Model metrics table */}
        {modelStatus?.metrics && (
          <div className="chart-card">
            <div className="chart-card-header">
              <h3>Model Metrics</h3>
              <span className="chart-badge">Test Set</span>
            </div>
            <div className="chart-body model-metrics-body">
              <table className="model-metrics-table">
                <thead>
                  <tr>
                    <th>Bridge</th>
                    <th>R²</th>
                    <th>MAE</th>
                    <th>Conf.</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(modelStatus.metrics).map(([bridge, m]) => (
                    <tr key={bridge}>
                      <td className="metrics-bridge-name">{BRIDGE_DISPLAY[bridge] || bridge}</td>
                      <td>
                        <span className={`metrics-r2 ${(m.r2 ?? 0) >= 0.7 ? "good" : (m.r2 ?? 0) >= 0.4 ? "ok" : "poor"}`}>
                          {m.r2 != null ? m.r2.toFixed(3) : "—"}
                        </span>
                      </td>
                      <td>{m.mae != null ? `$${m.mae.toFixed(3)}` : "—"}</td>
                      <td><ConfidenceBadge level={m.confidence || "low"} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}


/* ─── Main Page ───────────────────────────────────────────────────── */
export default function Home() {
  const [sourceChain, setSourceChain] = useState("Ethereum");
  const [destChain, setDestChain] = useState("Arbitrum");
  const [amount, setAmount] = useState("1000");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [predictions, setPredictions] = useState(null);

  // Analytics data (loaded once on mount)
  const [eda, setEda] = useState(null);
  const [modelStatus, setModelStatus] = useState(null);
  const [analyticsLoading, setAnalyticsLoading] = useState(true);

  // Load analytics data on mount
  useEffect(() => {
    async function loadAnalytics() {
      setAnalyticsLoading(true);
      try {
        const [edaResp, modelResp] = await Promise.allSettled([
          fetch(`${API_BASE}/eda`),
          fetch(`${API_BASE}/model/status`),
        ]);
        if (edaResp.status === "fulfilled" && edaResp.value.ok) {
          setEda(await edaResp.value.json());
        }
        if (modelResp.status === "fulfilled" && modelResp.value.ok) {
          setModelStatus(await modelResp.value.json());
        }
      } catch (err) {
        console.error("Failed to load analytics:", err);
      } finally {
        setAnalyticsLoading(false);
      }
    }
    loadAnalytics();
  }, []);

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
  const minFee = quotes.length ? Math.min(...quotes.map((q) => q.normalized_usd_fee)) : 0;
  const minTime = quotes.length ? Math.min(...quotes.map((q) => q.estimated_time_seconds)) : 0;

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
              id="source-chain" className="input-field"
              value={sourceChain} onChange={(e) => setSourceChain(e.target.value)}
            >
              {CHAINS.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
          <div className="config-group">
            <label htmlFor="dest-chain">Destination Network</label>
            <select
              id="dest-chain" className="input-field"
              value={destChain} onChange={(e) => setDestChain(e.target.value)}
            >
              {CHAINS.filter((c) => c !== sourceChain).map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
          <div className="config-group">
            <label htmlFor="amount">Transfer Amount (USDC)</label>
            <input
              id="amount" type="number" className="input-field"
              min="1" step="1" value={amount}
              onChange={(e) => setAmount(e.target.value)} placeholder="1000"
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
              <span className={result.source === "cache" ? "badge-cached" : "badge-live"}>
                {result.source === "cache" ? "Cached Quotes" : "Live Data"}
              </span>
            </div>
            <div className="quote-list">
              {[...quotes]
                .sort((a, b) => a.normalized_usd_fee - b.normalized_usd_fee)
                .map((q) => (
                  <QuoteCard
                    key={q.protocol} quote={q}
                    cheapest={q.normalized_usd_fee === minFee}
                    fastest={q.estimated_time_seconds === minTime}
                  />
                ))}
            </div>
          </section>
          {predictions && predictions.length > 0 && <PredictionPanel predictions={predictions} />}
        </>
      )}

      {!loading && !error && !result && (
        <div className="empty-state">
          Configure your transfer above to see real-time bridge quotes
          with detailed fee breakdowns.
        </div>
      )}

      {/* Analytics Dashboard */}
      {analyticsLoading ? (
        <div className="analytics-loading">
          <div className="loader" />
          <p>Loading analytics...</p>
        </div>
      ) : (
        <AnalyticsDashboard eda={eda} modelStatus={modelStatus} />
      )}

      <footer className="footer">
        <span>&copy; 2026 BridgeCompare</span>
        <span>Quotes are estimates based on current on-chain activity.</span>
      </footer>
    </div>
  );
}
