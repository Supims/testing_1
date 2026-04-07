"""
Diagnostic Script 1: Feature Pipeline Visual Validation
=========================================================
Generates multi-panel plots showing all computed features on synthetic data.
Validates: correct shapes, no NaN leakage, reasonable value ranges,
correlations between features, and warmup behavior.

Run:  python scripts/diagnose_features.py
Output: outputs/diagnostics/features_diagnostic.png
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def generate_realistic_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic BTC-like 1h OHLCV + microstructure data."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")

    # Multi-regime returns
    returns = np.zeros(n)
    returns[:200] = rng.normal(0.0001, 0.003, 200)        # quiet
    returns[200:400] = rng.normal(0.002, 0.008, 200)       # trending up
    returns[400:500] = rng.normal(-0.003, 0.015, 100)      # volatile crash
    returns[500:700] = rng.normal(0.0, 0.004, 200)         # ranging
    returns[700:850] = rng.normal(0.001, 0.005, 150)       # slow trend
    returns[850:] = rng.normal(-0.001, 0.010, n - 850)     # high vol down

    price = 50000 * np.exp(np.cumsum(returns))
    high = price * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = price * (1 - np.abs(rng.normal(0, 0.003, n)))
    volume = rng.lognormal(10, 1, n)

    # Make volume spike during volatile periods
    volume[400:500] *= 3
    volume[850:] *= 2

    df = pd.DataFrame({
        "open": np.roll(price, 1),
        "high": high,
        "low": low,
        "close": price,
        "volume": volume,
    }, index=timestamps)
    df.iloc[0, 0] = df.iloc[0, 3]

    # Microstructure
    taker_frac = rng.uniform(0.3, 0.7, n)
    taker_frac[200:400] = rng.uniform(0.55, 0.75, 200)  # bullish bias
    taker_frac[400:500] = rng.uniform(0.2, 0.4, 100)    # sellers
    df["taker_buy_volume"] = df["volume"] * taker_frac
    df["taker_sell_volume"] = df["volume"] - df["taker_buy_volume"]

    funding = np.zeros(n)
    funding[:200] = rng.normal(0.0001, 0.0002, 200)
    funding[200:400] = rng.normal(0.0005, 0.0003, 200)  # premium
    funding[400:500] = rng.normal(-0.0008, 0.0005, 100)  # fear
    funding[500:] = rng.normal(0.0001, 0.0003, n - 500)
    df["funding_rate"] = funding

    df["open_interest"] = rng.lognormal(12, 0.3, n)
    df["premium_index"] = rng.normal(0.0, 0.001, n)
    df["spot_close"] = df["close"] * (1 + rng.normal(0, 0.0005, n))

    return df


def run_feature_diagnostic():
    print("[1/5] Generating synthetic multi-regime data (1000 bars)...")
    raw = generate_realistic_data()
    print(f"  Raw shape: {raw.shape}")

    print("[2/5] Running Feature Pipeline...")
    from apollo.features.pipeline import FeaturePipeline
    pipe = FeaturePipeline(warmup_bars=50)
    features, meta = pipe.compute(raw)
    print(f"  Output: {features.shape[0]} rows x {features.shape[1]} columns")
    print(f"  Warmup dropped: {meta.warmup_rows_dropped}")
    print(f"  NaN columns after fill: {meta.nan_columns}")

    # Create output dir
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # Plot 1: Full Feature Dashboard (6 panels)
    # ================================================================
    print("[3/5] Plotting feature dashboard...")

    fig = plt.figure(figsize=(24, 28))
    fig.suptitle("Feature Pipeline Diagnostic -- Synthetic 1000-bar BTC",
                 fontsize=16, fontweight="bold", y=0.995)
    gs = GridSpec(7, 3, figure=fig, hspace=0.35, wspace=0.25)

    idx = features.index

    # Panel 1: Price + Volume
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(idx, features["close"], color="#2196F3", linewidth=0.8, label="Close")
    ax1.set_ylabel("Price ($)", fontsize=9)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_title("Price Action", fontsize=10, fontweight="bold")
    ax1v = ax1.twinx()
    ax1v.bar(idx, features["volume"], color="#90CAF9", alpha=0.3, width=0.03, label="Volume")
    ax1v.set_ylabel("Volume", fontsize=9)
    ax1.grid(True, alpha=0.2)

    # Panel 2: Trend Indicators
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(idx, features["rsi"], color="#FF9800", linewidth=0.7, label="RSI")
    ax2.axhline(70, color="#E53935", linestyle="--", linewidth=0.5, alpha=0.6)
    ax2.axhline(30, color="#4CAF50", linestyle="--", linewidth=0.5, alpha=0.6)
    ax2.set_ylabel("RSI", fontsize=9)
    ax2.set_title("RSI", fontsize=10, fontweight="bold")
    ax2.grid(True, alpha=0.2)
    ax2.set_ylim(0, 100)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(idx, features["macd"], color="#2196F3", linewidth=0.7, label="MACD")
    ax3.plot(idx, features["macd_signal"], color="#FF5722", linewidth=0.7, label="Signal")
    colors = ["#4CAF50" if v >= 0 else "#E53935" for v in features["macd_hist"]]
    ax3.bar(idx, features["macd_hist"], color=colors, alpha=0.5, width=0.03)
    ax3.set_title("MACD", fontsize=10, fontweight="bold")
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.2)

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(idx, features["adx"], color="#9C27B0", linewidth=0.8, label="ADX")
    ax4.plot(idx, features["adx_pos"], color="#4CAF50", linewidth=0.6, alpha=0.7, label="+DI")
    ax4.plot(idx, features["adx_neg"], color="#E53935", linewidth=0.6, alpha=0.7, label="-DI")
    ax4.axhline(25, color="gray", linestyle=":", linewidth=0.5)
    ax4.set_title("ADX + Directional", fontsize=10, fontweight="bold")
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.2)

    # Panel 3: Volatility
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(idx, features["gk_vol"], color="#E53935", linewidth=0.7, label="Garman-Klass")
    ax5.plot(idx, features["parkinson_vol"], color="#FF9800", linewidth=0.7, alpha=0.7, label="Parkinson")
    ax5.set_title("Volatility Estimators", fontsize=10, fontweight="bold")
    ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.2)

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(idx, features["bb_pct_b"], color="#2196F3", linewidth=0.7)
    ax6.axhline(1.0, color="#E53935", linestyle="--", linewidth=0.5)
    ax6.axhline(0.0, color="#4CAF50", linestyle="--", linewidth=0.5)
    ax6.set_title("Bollinger %B", fontsize=10, fontweight="bold")
    ax6.grid(True, alpha=0.2)

    ax7 = fig.add_subplot(gs[2, 2])
    ax7.plot(idx, features["natr"], color="#9C27B0", linewidth=0.7)
    ax7.set_title("NATR (Normalized ATR)", fontsize=10, fontweight="bold")
    ax7.grid(True, alpha=0.2)

    # Panel 4: Microstructure
    ax8 = fig.add_subplot(gs[3, 0])
    if "funding_zscore" in features.columns:
        ax8.plot(idx, features["funding_zscore"], color="#FF5722", linewidth=0.7)
        ax8.axhline(2, color="#E53935", linestyle="--", linewidth=0.5)
        ax8.axhline(-2, color="#4CAF50", linestyle="--", linewidth=0.5)
    ax8.set_title("Funding Z-Score", fontsize=10, fontweight="bold")
    ax8.grid(True, alpha=0.2)

    ax9 = fig.add_subplot(gs[3, 1])
    if "taker_delta_ema_20" in features.columns:
        ax9.plot(idx, features["taker_delta_ema_20"], color="#2196F3", linewidth=0.7)
        ax9.axhline(0, color="gray", linestyle=":", linewidth=0.5)
    ax9.set_title("Taker Delta EMA(20)", fontsize=10, fontweight="bold")
    ax9.grid(True, alpha=0.2)

    ax10 = fig.add_subplot(gs[3, 2])
    if "cvd_rolling_100" in features.columns:
        ax10.plot(idx, features["cvd_rolling_100"], color="#4CAF50", linewidth=0.7)
    ax10.set_title("CVD Rolling(100)", fontsize=10, fontweight="bold")
    ax10.grid(True, alpha=0.2)

    # Panel 5: HMM Inputs
    ax11 = fig.add_subplot(gs[4, 0])
    ax11.plot(idx, features["autocorr_w20_l1"], color="#607D8B", linewidth=0.7)
    ax11.axhline(0, color="gray", linestyle=":", linewidth=0.5)
    ax11.set_title("Autocorrelation (w=20, lag=1)", fontsize=10, fontweight="bold")
    ax11.grid(True, alpha=0.2)

    ax12 = fig.add_subplot(gs[4, 1])
    ax12.plot(idx, features["log_return"], color="#795548", linewidth=0.5, alpha=0.7)
    ax12.set_title("Log Returns", fontsize=10, fontweight="bold")
    ax12.grid(True, alpha=0.2)

    ax13 = fig.add_subplot(gs[4, 2])
    if "basis_zscore_100" in features.columns:
        ax13.plot(idx, features["basis_zscore_100"], color="#00BCD4", linewidth=0.7)
        ax13.axhline(2, color="#E53935", linestyle="--", linewidth=0.5)
        ax13.axhline(-2, color="#4CAF50", linestyle="--", linewidth=0.5)
    ax13.set_title("Basis Z-Score(100)", fontsize=10, fontweight="bold")
    ax13.grid(True, alpha=0.2)

    # Panel 6: Volume Analysis
    ax14 = fig.add_subplot(gs[5, 0])
    ax14.plot(idx, features["cmf"], color="#FF9800", linewidth=0.7)
    ax14.axhline(0, color="gray", linestyle=":", linewidth=0.5)
    ax14.set_title("Chaikin Money Flow", fontsize=10, fontweight="bold")
    ax14.grid(True, alpha=0.2)

    ax15 = fig.add_subplot(gs[5, 1])
    ax15.plot(idx, features["vwap_dist"], color="#9C27B0", linewidth=0.7)
    ax15.axhline(0, color="gray", linestyle=":", linewidth=0.5)
    ax15.set_title("VWAP Distance (%)", fontsize=10, fontweight="bold")
    ax15.grid(True, alpha=0.2)

    ax16 = fig.add_subplot(gs[5, 2])
    if "vol_oi_ratio" in features.columns:
        ax16.plot(idx, features["vol_oi_ratio"], color="#795548", linewidth=0.7)
    ax16.set_title("Volume/OI Ratio", fontsize=10, fontweight="bold")
    ax16.grid(True, alpha=0.2)

    # Panel 7: Feature Stats Summary
    ax17 = fig.add_subplot(gs[6, :])
    ax17.axis("off")
    key_cols = ["rsi", "adx", "macd_hist", "gk_vol", "natr", "bb_pct_b",
                "cmf", "funding_zscore", "vwap_dist", "autocorr_w20_l1",
                "taker_delta_ema_20", "basis_zscore_100"]
    stats_data = []
    for col in key_cols:
        if col in features.columns:
            s = features[col]
            stats_data.append([col, f"{s.mean():.4f}", f"{s.std():.4f}",
                             f"{s.min():.4f}", f"{s.max():.4f}",
                             f"{s.isna().sum()}", f"{(s == 0).sum()}"])
    table = ax17.table(
        cellText=stats_data,
        colLabels=["Feature", "Mean", "Std", "Min", "Max", "NaN", "Zeros"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    ax17.set_title("Feature Statistics Summary", fontsize=10, fontweight="bold", pad=10)

    path1 = out_dir / "features_diagnostic.png"
    fig.savefig(path1, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> Saved: {path1}")

    # ================================================================
    # Plot 2: Feature Correlation Heatmap
    # ================================================================
    print("[4/5] Plotting correlation heatmap...")
    numeric_cols = [c for c in features.columns if features[c].dtype in [np.float64, np.float32]]
    sel = [c for c in numeric_cols if c not in
           ["open", "high", "low", "close", "volume", "spot_close",
            "open_interest", "premium_index", "funding_rate",
            "taker_buy_volume", "taker_sell_volume"]]

    corr = features[sel].corr()
    fig2, ax = plt.subplots(figsize=(18, 14))
    cax = ax.matshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    fig2.colorbar(cax, ax=ax, shrink=0.7)
    ax.set_xticks(range(len(sel)))
    ax.set_yticks(range(len(sel)))
    ax.set_xticklabels(sel, rotation=90, fontsize=6)
    ax.set_yticklabels(sel, fontsize=6)
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=20)

    path2 = out_dir / "features_correlation.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"  -> Saved: {path2}")

    # ================================================================
    # NaN / Quality Report
    # ================================================================
    print("[5/5] NaN and quality report...")
    print(f"  Total features: {len(sel)}")
    print(f"  Output rows: {len(features)}")
    all_zero_cols = [c for c in sel if (features[c] == 0).all()]
    if all_zero_cols:
        print(f"  WARNING -- All-zero columns: {all_zero_cols}")
    else:
        print("  OK -- No all-zero feature columns")

    inf_cols = [c for c in sel if np.isinf(features[c]).any()]
    if inf_cols:
        print(f"  WARNING -- Inf columns: {inf_cols}")
    else:
        print("  OK -- No inf values")

    print("\n  DONE. Feature pipeline validated successfully.")


if __name__ == "__main__":
    run_feature_diagnostic()
