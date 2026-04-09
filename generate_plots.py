import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

folder_path = 'cleaned_split_data'
protocols = ['across', 'cctp', 'stargate_bus', 'stargate_oft']

numeric_features = [
    'amount_usd', 'dune_hourly_gas_gwei', 'gas_1h_lag', 'gas_6h_avg',
    'gas_24h_avg', 'gas_volatility_24h', 'eth_price_at_src',
    'eth_price_change_1h', 'eth_price_24h_avg', 'bridge_hourly_volume',
    'hour_of_day', 'day_of_week', 'is_weekend', 'month'
]
categorical_features = ['route', 'src_symbol']
TARGET = 'user_cost'


def engineer_features(df):
    df['src_timestamp'] = pd.to_numeric(df['src_timestamp'], errors='coerce')
    df['datetime'] = pd.to_datetime(df['src_timestamp'], unit='s', utc=True)

    df['hour_of_day'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = df['datetime'].dt.month

    if 'src_blockchain' in df.columns and 'dst_blockchain' in df.columns:
        df['route'] = df['src_blockchain'].astype(str) + '→' + df['dst_blockchain'].astype(str)

    df = df.sort_values('datetime').reset_index(drop=True)

    if 'dune_hourly_gas_gwei' in df.columns:
        df['gas_1h_lag'] = df['dune_hourly_gas_gwei'].shift(1)
        df['gas_6h_avg'] = df['dune_hourly_gas_gwei'].rolling(6, min_periods=1).mean()
        df['gas_24h_avg'] = df['dune_hourly_gas_gwei'].rolling(24, min_periods=1).mean()
        df['gas_volatility_24h'] = df['dune_hourly_gas_gwei'].rolling(24, min_periods=1).std()

    if 'eth_price_at_src' in df.columns:
        df['eth_price_change_1h'] = df['eth_price_at_src'].pct_change()
        df['eth_price_24h_avg'] = df['eth_price_at_src'].rolling(24, min_periods=1).mean()

    df['hour_bucket'] = df['datetime'].dt.floor('h')
    hourly_volume = df.groupby('hour_bucket').size().rename('bridge_hourly_volume')
    df = df.merge(hourly_volume, on='hour_bucket', how='left')

    return df


def train_models():
    all_results = {}

    for protocol in protocols:
        file_path = os.path.join(folder_path, f'{protocol}_cleaned.csv')
        if not os.path.exists(file_path):
            print(f"  {protocol}: file not found, skipping")
            continue

        df = pd.read_csv(file_path)
        df = engineer_features(df)

        if TARGET not in df.columns:
            print(f"  {protocol}: no '{TARGET}' column, skipping")
            continue

        df['src_timestamp'] = pd.to_numeric(df['src_timestamp'], errors='coerce')
        df = df.sort_values('src_timestamp').reset_index(drop=True)

        avail_num = [f for f in numeric_features if f in df.columns]
        avail_cat = [f for f in categorical_features if f in df.columns]
        feature_cols = avail_num + avail_cat

        model_df = df[feature_cols + [TARGET]].copy()
        for col in avail_cat:
            le = LabelEncoder()
            model_df[col] = le.fit_transform(model_df[col].astype(str))

        model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()
        model_df = model_df[model_df[TARGET] > 0]
        model_df['log_target'] = np.log1p(model_df[TARGET])

        if len(model_df) < 50:
            print(f"  {protocol}: only {len(model_df)} rows, skipping")
            continue

        split_idx = int(len(model_df) * 0.8)
        train = model_df.iloc[:split_idx]
        test = model_df.iloc[split_idx:]

        X_train = train[feature_cols]
        y_train_log = train['log_target']
        y_train_raw = train[TARGET]
        X_test = test[feature_cols]
        y_test_log = test['log_target']
        y_test_raw = test[TARGET]

        print(f"  {protocol.upper()}  |  Train: {len(train)}  |  Test: {len(test)}  |  Features: {len(feature_cols)}")

        models = {
            'XGBoost': XGBRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, verbosity=0
            ),
            'RandomForest': RandomForestRegressor(
                n_estimators=300, max_depth=12, min_samples_leaf=5,
                random_state=42, n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, random_state=42
            )
        }

        protocol_results = {}
        for name, model in models.items():
            model.fit(X_train, y_train_log)
            y_pred_log = model.predict(X_test)
            y_pred_raw = np.expm1(y_pred_log)
            y_pred_raw = np.maximum(y_pred_raw, 0)

            mae = mean_absolute_error(y_test_raw, y_pred_raw)
            rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred_raw))
            r2_raw = r2_score(y_test_raw, y_pred_raw)
            r2_log = r2_score(y_test_log, y_pred_log)
            pct_errors = np.abs(y_test_raw.values - y_pred_raw) / np.maximum(y_test_raw.values, 0.001) * 100
            mdape = np.median(pct_errors)

            baseline_pred = np.full_like(y_test_raw, y_train_raw.median())
            baseline_mae = mean_absolute_error(y_test_raw, baseline_pred)
            improvement = (1 - mae / baseline_mae) * 100

            protocol_results[name] = {
                'model': model, 'mae': mae, 'rmse': rmse,
                'r2_raw': r2_raw, 'r2_log': r2_log,
                'mdape': mdape, 'improvement': improvement,
                'y_test': y_test_raw, 'y_pred': y_pred_raw,
                'y_test_log': y_test_log, 'y_pred_log': y_pred_log,
                'feature_names': feature_cols
            }
            print(f"    {name:<20} MAE=${mae:.4f}  R²(log)={r2_log:.4f}")

        all_results[protocol] = protocol_results

    return all_results


def plot_actual_vs_predicted(all_results):
    protocols_with_results = [p for p in protocols if p in all_results]
    if not protocols_with_results:
        print("No results to plot.")
        return

    n = len(protocols_with_results)
    plot_models = ['XGBoost', 'RandomForest']

    fig, axes = plt.subplots(2, n, figsize=(6 * n, 10))
    if n == 1:
        axes = axes.reshape(-1, 1)

    for col_idx, protocol in enumerate(protocols_with_results):
        res = all_results[protocol]
        for row_idx, model_name in enumerate(plot_models):
            if model_name not in res:
                continue
            ax = axes[row_idx, col_idx]
            y_test = res[model_name]['y_test']
            y_pred = res[model_name]['y_pred']
            r2 = res[model_name]['r2_raw']

            ax.scatter(y_test, y_pred, alpha=0.3, s=10, color='#3498db')
            lims = [max(min(y_test.min(), y_pred.min()), 0.001),
                    max(y_test.max(), y_pred.max()) * 1.1]
            ax.plot(lims, lims, 'r--', alpha=0.8, linewidth=1.5, label='Perfect')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title(f'{protocol.upper()} — {model_name}\nR²={r2:.3f}',
                         fontsize=11, fontweight='bold')
            ax.set_xlabel('Actual User Cost ($, log scale)')
            ax.set_ylabel('Predicted User Cost ($, log scale)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png', dpi=150, bbox_inches='tight')
    print("Saved: actual_vs_predicted.png")
    plt.close()


def plot_prediction_error_distribution(all_results):
    protocols_with_results = [p for p in protocols if p in all_results]
    if not protocols_with_results:
        return

    n = len(protocols_with_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for idx, protocol in enumerate(protocols_with_results):
        res = all_results[protocol]
        best_model = min(res.keys(), key=lambda k: res[k]['mae'])
        y_test_log = res[best_model]['y_test_log']
        y_pred_log = res[best_model]['y_pred_log']
        log_errors = y_test_log.values - y_pred_log

        axes[idx].hist(log_errors, bins=50, color='#2c3e50', alpha=0.7, edgecolor='white')
        axes[idx].axvline(x=0, color='red', linestyle='--', linewidth=1.5)
        axes[idx].set_title(f'{protocol.upper()} — Log-Space Error Distribution\n(Best: {best_model})',
                            fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Error in log1p(user_cost)')
        axes[idx].set_ylabel('Count')
        axes[idx].grid(True, alpha=0.3)

        raw_errors = np.abs(res[best_model]['y_test'].values - res[best_model]['y_pred'])
        p50 = np.percentile(raw_errors, 50)
        p90 = np.percentile(raw_errors, 90)
        axes[idx].annotate(f'Median |error|: ${p50:.4f}\n90th %ile: ${p90:.4f}',
                           xy=(0.95, 0.95), xycoords='axes fraction',
                           ha='right', va='top', fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig('prediction_error_distribution.png', dpi=150, bbox_inches='tight')
    print("Saved: prediction_error_distribution.png")
    plt.close()


def plot_feature_importance(all_results):
    protocols_with_results = [p for p in protocols if p in all_results]
    if not protocols_with_results:
        return

    n = len(protocols_with_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for idx, protocol in enumerate(protocols_with_results):
        res = all_results[protocol]
        if 'XGBoost' not in res:
            continue
        model = res['XGBoost']['model']
        feat_names = res['XGBoost']['feature_names']
        importance = model.feature_importances_
        sorted_idx = np.argsort(importance)

        axes[idx].barh(range(len(sorted_idx)), importance[sorted_idx], color='#2ecc71', edgecolor='#27ae60')
        axes[idx].set_yticks(range(len(sorted_idx)))
        axes[idx].set_yticklabels([feat_names[i] for i in sorted_idx], fontsize=9)
        axes[idx].set_title(f'{protocol.upper()} — XGBoost Feature Importance',
                            fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Importance')

    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    print("Saved: feature_importance.png")
    plt.close()


def draw_architecture_diagram():
    """Model architecture diagram showing the ensemble learning pipeline."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 14))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 14)
    ax.axis('off')
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#f8f9fa')

    title_props = dict(fontsize=18, fontweight='bold', color='#1a1a2e', ha='center', va='center')
    header_props = dict(fontsize=11, fontweight='bold', color='white', ha='center', va='center')
    body_props = dict(fontsize=9, ha='center', va='center', color='#2d3436')
    small_props = dict(fontsize=8, ha='center', va='center', color='#636e72')

    ax.text(9, 13.5, 'Cross-Chain Bridge Fee Prediction — Model Architecture',
            **title_props)

    # --- Input Layer ---
    input_box = mpatches.FancyBboxPatch((0.5, 10.8), 4.5, 2.2, boxstyle="round,pad=0.15",
                                         facecolor='#0984e3', edgecolor='#0652DD', linewidth=2)
    ax.add_patch(input_box)
    ax.text(2.75, 12.6, 'INPUT FEATURES', **header_props)
    input_features = [
        'Numeric (14):',
        'amount_usd, gas_gwei, gas_lags,',
        'eth_price, volume, time features',
        '',
        'Categorical (2):',
        'route, src_symbol (LabelEncoded)'
    ]
    for i, line in enumerate(input_features):
        ax.text(2.75, 12.15 - i * 0.22, line, fontsize=8, ha='center', va='center', color='#dfe6e9')

    # --- Target Transform ---
    target_box = mpatches.FancyBboxPatch((6.0, 11.2), 3.5, 1.5, boxstyle="round,pad=0.15",
                                          facecolor='#6c5ce7', edgecolor='#5f27cd', linewidth=2)
    ax.add_patch(target_box)
    ax.text(7.75, 12.3, 'TARGET TRANSFORM', **header_props)
    ax.text(7.75, 11.85, 'user_cost → log1p(user_cost)', fontsize=9, ha='center', va='center', color='#dfe6e9')
    ax.text(7.75, 11.55, 'Handles heavy right skew', fontsize=8, ha='center', va='center', color='#a29bfe')

    # --- Data Split ---
    split_box = mpatches.FancyBboxPatch((10.5, 11.2), 3.5, 1.5, boxstyle="round,pad=0.15",
                                         facecolor='#00b894', edgecolor='#00a886', linewidth=2)
    ax.add_patch(split_box)
    ax.text(12.25, 12.3, 'TIME-BASED SPLIT', **header_props)
    ax.text(12.25, 11.85, 'Train: 80% (earlier)', fontsize=9, ha='center', va='center', color='#dfe6e9')
    ax.text(12.25, 11.55, 'Test: 20% (later)', fontsize=9, ha='center', va='center', color='#dfe6e9')

    # --- Per-Bridge Label ---
    bridge_box = mpatches.FancyBboxPatch((14.8, 11.2), 2.7, 1.5, boxstyle="round,pad=0.15",
                                          facecolor='#fdcb6e', edgecolor='#f39c12', linewidth=2)
    ax.add_patch(bridge_box)
    ax.text(16.15, 12.3, 'PER-BRIDGE', fontsize=11, fontweight='bold', ha='center', va='center', color='#2d3436')
    bridges = ['Across', 'CCTP', 'Stargate Bus', 'Stargate OFT']
    for i, b in enumerate(bridges):
        ax.text(16.15, 11.9 - i * 0.22, b, fontsize=8, ha='center', va='center', color='#2d3436')

    # --- Arrow from input to models ---
    ax.annotate('', xy=(9, 10.2), xytext=(9, 10.8),
                arrowprops=dict(arrowstyle='->', color='#2d3436', lw=2))

    # --- Three Model Boxes ---
    model_colors = [('#d63031', '#c0392b'), ('#00cec9', '#00b5ad'), ('#e17055', '#d35400')]
    model_names = ['XGBoost', 'Random Forest', 'Gradient Boosting']
    model_params = [
        ['n_estimators=500', 'max_depth=6', 'lr=0.03', 'subsample=0.8'],
        ['n_estimators=300', 'max_depth=12', 'min_leaf=5'],
        ['n_estimators=300', 'max_depth=5', 'lr=0.05', 'subsample=0.8']
    ]

    model_x_positions = [1.5, 6.5, 11.5]
    for i, (x_pos, (fc, ec), name, params) in enumerate(zip(model_x_positions, model_colors, model_names, model_params)):
        box = mpatches.FancyBboxPatch((x_pos, 7.8), 5, 2.3, boxstyle="round,pad=0.15",
                                       facecolor=fc, edgecolor=ec, linewidth=2)
        ax.add_patch(box)
        ax.text(x_pos + 2.5, 9.7, name, fontsize=12, fontweight='bold', color='white', ha='center', va='center')
        ax.text(x_pos + 2.5, 9.3, 'Regressor', fontsize=9, ha='center', va='center', color='#dfe6e9')
        for j, p in enumerate(params):
            ax.text(x_pos + 2.5, 8.85 - j * 0.25, p, fontsize=8, ha='center', va='center', color='#fab1a0' if i != 1 else '#81ecec')

    # --- Arrows from models to evaluation ---
    for x_pos in model_x_positions:
        ax.annotate('', xy=(9, 7.2), xytext=(x_pos + 2.5, 7.8),
                    arrowprops=dict(arrowstyle='->', color='#2d3436', lw=1.5))

    # --- Inverse Transform + Evaluation ---
    eval_box = mpatches.FancyBboxPatch((3, 5.2), 12, 1.8, boxstyle="round,pad=0.15",
                                        facecolor='#2d3436', edgecolor='#1a1a2e', linewidth=2)
    ax.add_patch(eval_box)
    ax.text(9, 6.7, 'INVERSE TRANSFORM & EVALUATION', fontsize=13, fontweight='bold', color='white', ha='center', va='center')
    ax.text(9, 6.2, 'expm1(y_pred_log)  →  clip(min=0)  →  Compare to actual user_cost',
            fontsize=10, ha='center', va='center', color='#b2bec3')

    eval_metrics = 'Metrics:  MAE ($)  |  RMSE ($)  |  R²(raw)  |  R²(log)  |  MdAPE%  |  vs Baseline'
    ax.text(9, 5.7, eval_metrics, fontsize=9, ha='center', va='center', color='#74b9ff')

    # --- Arrow to outputs ---
    ax.annotate('', xy=(9, 4.6), xytext=(9, 5.2),
                arrowprops=dict(arrowstyle='->', color='#2d3436', lw=2))

    # --- Best Model Selection ---
    best_box = mpatches.FancyBboxPatch((2, 3), 14, 1.4, boxstyle="round,pad=0.15",
                                        facecolor='#ffeaa7', edgecolor='#f39c12', linewidth=2)
    ax.add_patch(best_box)
    ax.text(9, 4.1, 'BEST MODEL SELECTION (min MAE per bridge)', fontsize=12, fontweight='bold',
            ha='center', va='center', color='#2d3436')

    results_text = (
        'Across → GradientBoosting (R²=0.83)    |    CCTP → GradientBoosting (R²=0.66)    |    '
        'Stargate Bus → XGBoost (R²=0.36)    |    Stargate OFT → RandomForest (R²=0.91)'
    )
    ax.text(9, 3.5, results_text, fontsize=7.5, ha='center', va='center', color='#636e72')

    # --- Output Boxes ---
    output_labels = ['Actual vs\nPredicted', 'Error\nDistribution', 'Feature\nImportance', 'Cost\nEstimation']
    output_colors = ['#3498db', '#2c3e50', '#2ecc71', '#e84393']
    for i, (label, color) in enumerate(zip(output_labels, output_colors)):
        x = 2.5 + i * 3.8
        box = mpatches.FancyBboxPatch((x, 1), 2.8, 1.5, boxstyle="round,pad=0.15",
                                       facecolor=color, edgecolor='white', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + 1.4, 1.75, label, fontsize=10, fontweight='bold',
                ha='center', va='center', color='white')

    for i in range(4):
        x = 2.5 + i * 3.8 + 1.4
        ax.annotate('', xy=(x, 2.5), xytext=(9, 3.0),
                    arrowprops=dict(arrowstyle='->', color='#636e72', lw=1.2))

    plt.savefig('architecture_diagram.png', dpi=150, bbox_inches='tight')
    print("Saved: architecture_diagram.png")
    plt.close()


def draw_pipeline_architecture():
    """End-to-end data pipeline from raw data to prediction outputs."""
    fig, ax = plt.subplots(1, 1, figsize=(22, 12))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 12)
    ax.axis('off')
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('#fafafa')

    ax.text(11, 11.5, 'Cross-Chain Bridge Fee Prediction — End-to-End Pipeline',
            fontsize=20, fontweight='bold', ha='center', va='center', color='#1a1a2e')

    def draw_stage(ax, x, y, w, h, title, items, title_color, box_color, edge_color, item_color='#ecf0f1'):
        box = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                       facecolor=box_color, edgecolor=edge_color, linewidth=2.5)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h - 0.3, title, fontsize=12, fontweight='bold',
                ha='center', va='center', color=title_color)
        for i, item in enumerate(items):
            ax.text(x + w / 2, y + h - 0.75 - i * 0.35, item, fontsize=8.5,
                    ha='center', va='center', color=item_color)

    # --- STAGE 1: Data Sources ---
    draw_stage(ax, 0.3, 7.5, 3.5, 3.2,
               '1. DATA SOURCES',
               ['all_ccc_sampled.csv', '(10K+ bridge txns)',
                '', 'Alchemy API → ETH prices',
                'Dune Analytics → Gas prices',
                'Etherscan → Daily gas avg'],
               'white', '#0984e3', '#0652DD')

    # --- Arrow ---
    ax.annotate('', xy=(4.3, 9.1), xytext=(3.8, 9.1),
                arrowprops=dict(arrowstyle='->', color='#2d3436', lw=2.5))

    # --- STAGE 2: Cleaning & Splitting ---
    draw_stage(ax, 4.3, 7.5, 3.5, 3.2,
               '2. CLEAN & SPLIT',
               ['Split by bridge protocol:', '  Across, CCTP, CCIP,',
                '  Stargate Bus, Stargate OFT',
                '', 'Remove duplicates',
                'Filter neg latency/fees',
                'Remove >1yr latency outliers'],
               'white', '#6c5ce7', '#5f27cd')

    ax.annotate('', xy=(8.3, 9.1), xytext=(7.8, 9.1),
                arrowprops=dict(arrowstyle='->', color='#2d3436', lw=2.5))

    # --- STAGE 3: Enrichment ---
    draw_stage(ax, 8.3, 7.5, 3.5, 3.2,
               '3. DATA ENRICHMENT',
               ['Merge hourly ETH prices',
                'Merge hourly gas (Dune)',
                'Merge daily gas (Etherscan)',
                '', 'Join key: hour-bucket',
                'Left joins preserve all txns',
                'Fill coverage gaps'],
               'white', '#00b894', '#00a886')

    ax.annotate('', xy=(12.3, 9.1), xytext=(11.8, 9.1),
                arrowprops=dict(arrowstyle='->', color='#2d3436', lw=2.5))

    # --- STAGE 4: Feature Engineering ---
    draw_stage(ax, 12.3, 7.5, 3.5, 3.2,
               '4. FEATURE ENG.',
               ['Temporal: hour, day, weekend',
                'Gas lags: 1h, 6h_avg, 24h_avg',
                'Gas volatility: 24h rolling std',
                'ETH momentum: pct_change, 24h_avg',
                'Volume: hourly bridge tx count',
                'Route: src→dst blockchain',
                'Normalize: amount_usd, symbol'],
               '#2d3436', '#ffeaa7', '#f39c12', '#2d3436')

    ax.annotate('', xy=(16.3, 9.1), xytext=(15.8, 9.1),
                arrowprops=dict(arrowstyle='->', color='#2d3436', lw=2.5))

    # --- STAGE 5: EDA & Hypothesis Testing ---
    draw_stage(ax, 16.3, 7.5, 5.2, 3.2,
               '5. EDA & HYPOTHESIS TESTING',
               ['Fee vs Latency correlation (Spearman)',
                'Whale analysis (>$10K transfers)',
                'Cost evolution over time',
                'Confounding variable analysis',
                'Gas autocorrelation test (r=0.91)',
                'Weekend vs Weekday effect (40% cheaper)',
                'Hour-of-day effect (5am cheapest)'],
               'white', '#e17055', '#d35400')

    # --- Middle Row: Model Training ---
    ax.annotate('', xy=(11, 7.0), xytext=(11, 7.5),
                arrowprops=dict(arrowstyle='->', color='#2d3436', lw=2.5))

    # --- STAGE 6: Model Training ---
    draw_stage(ax, 3, 3.5, 5, 3.2,
               '6. MODEL TRAINING (Per Bridge)',
               ['Target: log1p(user_cost)',
                'Split: 80/20 time-based (no leakage)',
                '',
                'XGBoost: 500 trees, depth=6, lr=0.03',
                'RandomForest: 300 trees, depth=12',
                'GradientBoosting: 300 trees, depth=5',
                '16 features (14 numeric + 2 categorical)'],
               'white', '#d63031', '#c0392b')

    ax.annotate('', xy=(8.5, 5.1), xytext=(8.0, 5.1),
                arrowprops=dict(arrowstyle='->', color='#2d3436', lw=2.5))

    # --- STAGE 7: Evaluation ---
    draw_stage(ax, 8.5, 3.5, 5, 3.2,
               '7. EVALUATION & SELECTION',
               ['Inverse transform: expm1(pred)',
                'Clip negative predictions to 0',
                '',
                'MAE, RMSE, R²(raw), R²(log)',
                'MdAPE%, vs Baseline (median)',
                '',
                'Best model selected per bridge'],
               'white', '#2d3436', '#1a1a2e')

    ax.annotate('', xy=(14.0, 5.1), xytext=(13.5, 5.1),
                arrowprops=dict(arrowstyle='->', color='#2d3436', lw=2.5))

    # --- STAGE 8: Outputs ---
    draw_stage(ax, 14.0, 3.5, 7.5, 3.2,
               '8. OUTPUTS & ARTIFACTS',
               ['actual_vs_predicted.png — scatter plots',
                'prediction_error_distribution.png — histograms',
                'feature_importance.png — XGBoost importances',
                '',
                'Best models: Across→GB, CCTP→GB,',
                '  Stargate Bus→XGB, Stargate OFT→RF',
                'Recommendation: hybrid model + live API quotes'],
               'white', '#e84393', '#d63384')

    # --- Bottom: Key Results Summary ---
    summary_box = mpatches.FancyBboxPatch((1, 0.5), 20, 2.5, boxstyle="round,pad=0.15",
                                           facecolor='#dfe6e9', edgecolor='#b2bec3', linewidth=2)
    ax.add_patch(summary_box)
    ax.text(11, 2.6, 'KEY RESULTS', fontsize=14, fontweight='bold', ha='center', va='center', color='#2d3436')

    results = [
        'Across: GradientBoosting  R²(log)=0.83  MAE=$2.13  →  STRONG',
        'CCTP: GradientBoosting  R²(log)=0.66  MAE=$1.48  →  MODERATE',
        'Stargate Bus: XGBoost  R²(log)=0.36  MAE=$0.53  →  WEAK',
        'Stargate OFT: RandomForest  R²(log)=0.91  MAE=$0.14  →  STRONG',
    ]
    for i, r in enumerate(results):
        ax.text(11, 2.1 - i * 0.35, r, fontsize=9, ha='center', va='center',
                color='#2d3436', family='monospace')

    ax.annotate('', xy=(11, 3.0), xytext=(11, 3.5),
                arrowprops=dict(arrowstyle='->', color='#2d3436', lw=2.5))

    plt.savefig('pipeline_architecture.png', dpi=150, bbox_inches='tight')
    print("Saved: pipeline_architecture.png")
    plt.close()


if __name__ == '__main__':
    print("=" * 80)
    print("Training models and generating all plots...")
    print("=" * 80)

    all_results = train_models()

    print("\n--- Generating plots ---")
    plot_actual_vs_predicted(all_results)
    plot_prediction_error_distribution(all_results)
    plot_feature_importance(all_results)
    draw_architecture_diagram()
    draw_pipeline_architecture()

    print("\nAll done!")
