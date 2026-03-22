"use client";

import { useState, useEffect, useCallback } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const BRIDGES = ["all", "across", "cctp", "ccip", "stargate_oft", "stargate_bus"];

function ts(epoch) {
  if (!epoch) return "—";
  const d = new Date(Number(epoch) * 1000);
  return d.toLocaleString();
}

function usd(v) {
  const n = parseFloat(v);
  if (isNaN(n)) return "—";
  return n < 0.01 ? `$${n.toFixed(6)}` : `$${n.toFixed(2)}`;
}

function secs(v) {
  const n = parseInt(v, 10);
  if (isNaN(n)) return "—";
  if (n < 60) return `${n}s`;
  return `${Math.floor(n / 60)}m ${n % 60}s`;
}

export default function DataPage() {
  const [stats, setStats] = useState(null);
  const [rows, setRows] = useState([]);
  const [bridge, setBridge] = useState("all");
  const [loading, setLoading] = useState(true);
  const [modelInfo, setModelInfo] = useState(null);
  const [retraining, setRetraining] = useState(false);
  const [error, setError] = useState(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const bridgeParam = bridge === "all" ? "" : `&bridge=${bridge}`;
      const [statsRes, rowsRes, modelRes] = await Promise.all([
        fetch(`${API}/data/stats`),
        fetch(`${API}/data/recent?limit=100${bridgeParam}`),
        fetch(`${API}/model/status`),
      ]);
      if (!statsRes.ok) throw new Error("Failed to fetch stats");
      setStats(await statsRes.json());
      setRows((await rowsRes.json()).rows || []);
      setModelInfo(await modelRes.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [bridge]);

  useEffect(() => { fetchData(); }, [fetchData]);

  async function handleRetrain() {
    setRetraining(true);
    try {
      const res = await fetch(`${API}/model/retrain`, { method: "POST" });
      if (!res.ok) throw new Error("Retrain failed");
      await fetchData();
    } catch (e) {
      setError(e.message);
    } finally {
      setRetraining(false);
    }
  }

  const maxBridge = stats
    ? Math.max(...Object.values(stats.bridges || {}))
    : 1;

  return (
    <div className="data-container">
      <div className="data-header">
        <h1>Training Data Dashboard</h1>
        <p>
          Live view of the data powering the ML fee prediction models.
          Data is appended every time a quote is fetched on the main page.
        </p>
      </div>

      {error && <div className="error-msg">{error}</div>}

      {loading && !stats ? (
        <div className="loading-view">
          <div className="loader" />
          <span>Loading dataset...</span>
        </div>
      ) : stats ? (
        <>
          {/* Stats cards */}
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-label">Total Rows</div>
              <div className="stat-value">{stats.total_rows?.toLocaleString()}</div>
              <div className="stat-sub">
                {stats.seed_rows?.toLocaleString()} seed + {stats.collected_rows?.toLocaleString()} collected
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Collected Live</div>
              <div className="stat-value">{stats.collected_rows?.toLocaleString()}</div>
              <div className="stat-sub">From real-time quotes</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">File Size</div>
              <div className="stat-value">{stats.file_size_mb} MB</div>
              <div className="stat-sub">training_data.csv</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Time Range</div>
              <div className="stat-value" style={{ fontSize: "1rem" }}>
                {stats.newest_timestamp ? ts(stats.newest_timestamp) : "—"}
              </div>
              <div className="stat-sub">
                Since {stats.oldest_timestamp ? ts(stats.oldest_timestamp) : "—"}
              </div>
            </div>
          </div>

          {/* Bridge breakdown */}
          <div className="data-section">
            <div className="section-head">
              <h2>Rows by Bridge</h2>
            </div>
            <div className="bridge-bars">
              {Object.entries(stats.bridges || {})
                .sort((a, b) => b[1] - a[1])
                .map(([name, count]) => (
                  <div key={name} className="bridge-bar-row">
                    <span className="bridge-bar-label">{name}</span>
                    <div className="bridge-bar-track">
                      <div
                        className="bridge-bar-fill"
                        style={{ width: `${(count / maxBridge) * 100}%` }}
                      />
                    </div>
                    <span className="bridge-bar-count">{count.toLocaleString()}</span>
                  </div>
                ))}
            </div>
          </div>

          {/* Model info */}
          {modelInfo?.metrics && (
            <div className="data-section">
              <div className="section-head">
                <h2>Model Performance</h2>
                <button
                  className="retrain-btn"
                  onClick={handleRetrain}
                  disabled={retraining}
                >
                  {retraining ? "Retraining..." : "Retrain Now"}
                </button>
              </div>
              <div className="model-cards">
                {Object.entries(modelInfo.metrics).map(([name, m]) => (
                  <div key={name} className="model-card">
                    <h4>{name}</h4>
                    <div className="model-metric">
                      <span className="model-metric-label">R²</span>
                      <span className="model-metric-val">
                        {m.r2 != null ? m.r2.toFixed(4) : "—"}
                      </span>
                    </div>
                    <div className="model-metric">
                      <span className="model-metric-label">MAE</span>
                      <span className="model-metric-val">
                        {m.mae != null ? `$${m.mae.toFixed(4)}` : "—"}
                      </span>
                    </div>
                    <div className="model-metric">
                      <span className="model-metric-label">RMSE</span>
                      <span className="model-metric-val">
                        {m.rmse != null ? `$${m.rmse.toFixed(4)}` : "—"}
                      </span>
                    </div>
                    <div className="model-metric">
                      <span className="model-metric-label">Confidence</span>
                      <span className="model-metric-val">{m.confidence || "—"}</span>
                    </div>
                    <div className="model-metric">
                      <span className="model-metric-label">Last trained</span>
                      <span className="model-metric-val" style={{ fontSize: "0.75rem" }}>
                        {m.last_trained || "—"}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Top routes */}
          {stats.top_routes && (
            <div className="data-section">
              <div className="section-head">
                <h2>Top Routes</h2>
              </div>
              <div className="bridge-bars">
                {Object.entries(stats.top_routes).map(([route, count]) => (
                  <div key={route} className="bridge-bar-row">
                    <span className="bridge-bar-label" style={{ width: 200 }}>{route}</span>
                    <div className="bridge-bar-track">
                      <div
                        className="bridge-bar-fill"
                        style={{
                          width: `${(count / Object.values(stats.top_routes)[0]) * 100}%`,
                          background: "var(--color-accent-light)",
                        }}
                      />
                    </div>
                    <span className="bridge-bar-count">{count.toLocaleString()}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Recent data table */}
          <div className="data-section">
            <div className="section-head">
              <h2>Recent Data</h2>
              <div className="filter-pills">
                {BRIDGES.map((b) => (
                  <button
                    key={b}
                    className={`pill ${bridge === b ? "active" : ""}`}
                    onClick={() => setBridge(b)}
                  >
                    {b === "all" ? "All" : b}
                  </button>
                ))}
              </div>
            </div>

            <div className="data-table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Timestamp</th>
                    <th>Source</th>
                    <th>Bridge</th>
                    <th>Route</th>
                    <th>Amount (USD)</th>
                    <th>Fee (USD)</th>
                    <th>User Cost</th>
                    <th>Latency</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.length === 0 ? (
                    <tr>
                      <td colSpan={8} style={{ textAlign: "center", padding: 40, color: "var(--color-text-muted)" }}>
                        No data rows found.
                      </td>
                    </tr>
                  ) : (
                    rows.map((r, i) => (
                      <tr key={i}>
                        <td>{ts(r.src_timestamp)}</td>
                        <td>
                          <span className={`source-tag ${r.source === "live" ? "source-live" : "source-seed"}`}>
                            {r.source === "live" ? "Live" : "Seed"}
                          </span>
                        </td>
                        <td>
                          <span className={`bridge-tag ${r.bridge}`}>{r.bridge}</span>
                        </td>
                        <td>{r.src_blockchain} → {r.dst_blockchain}</td>
                        <td>{usd(r.amount_usd)}</td>
                        <td>{usd(r.src_fee_usd)}</td>
                        <td>{usd(r.user_cost)}</td>
                        <td>{secs(r.latency)}</td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </>
      ) : null}
    </div>
  );
}
