"""
Diagnostic Script 2: Regime + Strategies + Ensemble Visual Validation
=====================================================================
End-to-end: Features -> HMM Regime -> 10 Strategies -> Ensemble Signal.
Multi-panel plot shows regime coloring, individual strategy signals,
and the final ensemble output.

Run:  python scripts/diagnose_strategies.py
Output: outputs/diagnostics/strategies_diagnostic.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch


# Regime colors
REGIME_COLORS = {
    "High Volatility (Trending)": "#E53935",
    "Low Volatility (Trending)": "#2196F3",
    "High Volatility (Ranging)": "#FF9800",
    "Low Volatility (Quiet Range)": "#4CAF50",
    "Unknown": "#9E9E9E",
}


def generate_data(n=1000, seed=42):
    """Same as features diagnostic."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    returns = np.zeros(n)
    returns[:200] = rng.normal(0.0001, 0.003, 200)
    returns[200:400] = rng.normal(0.002, 0.008, 200)
    returns[400:500] = rng.normal(-0.003, 0.015, 100)
    returns[500:700] = rng.normal(0.0, 0.004, 200)
    returns[700:850] = rng.normal(0.001, 0.005, 150)
    returns[850:] = rng.normal(-0.001, 0.010, n - 850)
    price = 50000 * np.exp(np.cumsum(returns))
    high = price * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = price * (1 - np.abs(rng.normal(0, 0.003, n)))
    volume = rng.lognormal(10, 1, n)
    volume[400:500] *= 3
    volume[850:] *= 2
    df = pd.DataFrame({"open": np.roll(price,1), "high": high, "low": low, "close": price, "volume": volume}, index=ts)
    df.iloc[0,0] = df.iloc[0,3]
    taker_frac = rng.uniform(0.3, 0.7, n)
    taker_frac[200:400] = rng.uniform(0.55, 0.75, 200)
    taker_frac[400:500] = rng.uniform(0.2, 0.4, 100)
    df["taker_buy_volume"] = df["volume"] * taker_frac
    df["taker_sell_volume"] = df["volume"] - df["taker_buy_volume"]
    funding = np.zeros(n)
    funding[:200] = rng.normal(0.0001, 0.0002, 200)
    funding[200:400] = rng.normal(0.0005, 0.0003, 200)
    funding[400:500] = rng.normal(-0.0008, 0.0005, 100)
    funding[500:] = rng.normal(0.0001, 0.0003, n - 500)
    df["funding_rate"] = funding
    df["open_interest"] = rng.lognormal(12, 0.3, n)
    df["premium_index"] = rng.normal(0.0, 0.001, n)
    df["spot_close"] = df["close"] * (1 + rng.normal(0, 0.0005, n))
    return df


def shade_regimes(ax, idx, labels, alpha=0.15):
    """Shade background by regime label."""
    if labels is None or len(labels) == 0:
        return
    current = labels.iloc[0]
    start = idx[0]
    for i in range(1, len(labels)):
        if labels.iloc[i] != current or i == len(labels) - 1:
            color = REGIME_COLORS.get(current, "#9E9E9E")
            ax.axvspan(start, idx[i], alpha=alpha, color=color)
            current = labels.iloc[i]
            start = idx[i]


def run_strategy_diagnostic():
    print("=" * 60)
    print("  STRATEGY + ENSEMBLE DIAGNOSTIC")
    print("=" * 60)

    print("\n[1/6] Generating synthetic data...")
    raw = generate_data()

    print("[2/6] Computing features...")
    from apollo.features.pipeline import FeaturePipeline
    pipe = FeaturePipeline(warmup_bars=50)
    features, meta = pipe.compute(raw)
    # Add back OHLCV for strategies
    for col in ["open", "high", "low", "close", "volume"]:
        features[col] = raw[col].loc[features.index]
    print(f"  Features: {features.shape}")

    print("[3/6] Fitting HMM Regime Detector...")
    from apollo.models.regime import RegimeDetector
    detector = RegimeDetector()
    detector.fit(features)
    regime_df = detector.predict(features)
    print(f"  Label map: {detector.label_map}")
    print(f"  OOD rate: {regime_df['hmm_ood'].mean():.1%}")

    print("[4/6] Computing 10 strategy signals...")
    from apollo.models.strategies import compute_all
    signals = compute_all(features)
    print(f"  Signal columns: {list(signals.columns)}")
    for col in signals.columns:
        active = (signals[col].abs() > 0.01).sum()
        print(f"    {col:20s}: {active:4d} active bars ({active/len(signals)*100:.1f}%)")

    print("[5/6] Computing ensemble...")
    from apollo.models.ensemble import StaticEnsemble
    label_map = detector.label_map
    ens = StaticEnsemble(label_map)
    ensemble_signal = ens.compute(signals, regime_df)
    print(f"  Ensemble range: [{ensemble_signal.min():.3f}, {ensemble_signal.max():.3f}]")
    print(f"  Ensemble mean:  {ensemble_signal.mean():.4f}")

    print("[6/6] Plotting diagnostic...")

    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    idx = features.index
    labels = regime_df["hmm_regime_label"].reindex(idx)
    n_strats = len(signals.columns)

    fig = plt.figure(figsize=(26, 6 + n_strats * 1.5 + 6))
    gs = GridSpec(3 + n_strats, 1, figure=fig, hspace=0.30,
                  height_ratios=[3] + [1] * n_strats + [2, 2])
    fig.suptitle("Regime + Strategy + Ensemble Diagnostic",
                 fontsize=16, fontweight="bold", y=0.998)

    # Panel 1: Price with regime shading
    ax_price = fig.add_subplot(gs[0])
    ax_price.plot(idx, features["close"], color="#1565C0", linewidth=0.8)
    shade_regimes(ax_price, idx, labels)
    ax_price.set_ylabel("Price ($)", fontsize=9)
    ax_price.set_title("Price + HMM Regime Shading", fontsize=11, fontweight="bold")
    ax_price.grid(True, alpha=0.15)
    # Legend
    patches = [Patch(facecolor=c, alpha=0.3, label=n[:20]) for n, c in REGIME_COLORS.items() if n != "Unknown"]
    ax_price.legend(handles=patches, loc="upper left", fontsize=7, ncol=2)

    # Panels 2-11: Strategy signals
    strat_colors = plt.cm.tab10(np.linspace(0, 1, n_strats))
    for i, strat_name in enumerate(signals.columns):
        ax = fig.add_subplot(gs[1 + i], sharex=ax_price)
        sig = signals[strat_name]

        pos = sig.clip(lower=0)
        neg = sig.clip(upper=0)
        ax.fill_between(idx, 0, pos, color="#4CAF50", alpha=0.5)
        ax.fill_between(idx, 0, neg, color="#E53935", alpha=0.5)
        ax.axhline(0, color="gray", linewidth=0.3)
        ax.set_ylim(-1.1, 1.1)
        ax.set_ylabel(strat_name, fontsize=7, rotation=0, labelpad=60, ha="left")
        ax.tick_params(axis="y", labelsize=6)
        if i < n_strats - 1:
            ax.tick_params(axis="x", labelbottom=False)
        shade_regimes(ax, idx, labels, alpha=0.08)
        ax.grid(True, alpha=0.1)

    # Panel: Ensemble Signal
    ax_ens = fig.add_subplot(gs[1 + n_strats], sharex=ax_price)
    ax_ens.fill_between(idx, 0, ensemble_signal.clip(lower=0), color="#1565C0", alpha=0.6)
    ax_ens.fill_between(idx, 0, ensemble_signal.clip(upper=0), color="#C62828", alpha=0.6)
    ax_ens.axhline(0, color="gray", linewidth=0.5)
    ax_ens.set_ylim(-1.1, 1.1)
    ax_ens.set_ylabel("Ensemble", fontsize=9, fontweight="bold")
    ax_ens.set_title("Regime-Weighted Ensemble Signal", fontsize=10, fontweight="bold")
    shade_regimes(ax_ens, idx, labels, alpha=0.08)
    ax_ens.grid(True, alpha=0.2)

    # Panel: Regime Probabilities
    ax_prob = fig.add_subplot(gs[2 + n_strats], sharex=ax_price)
    prob_cols = [c for c in regime_df.columns if c.startswith("hmm_prob_state_")]
    bottom = np.zeros(len(idx))
    for i, col in enumerate(sorted(prob_cols)):
        sid = int(col.split("_")[-1])
        label = label_map.get(sid, f"State {sid}")
        color = REGIME_COLORS.get(label, "#9E9E9E")
        vals = regime_df[col].reindex(idx).fillna(0).values
        ax_prob.bar(idx, vals, bottom=bottom, color=color, alpha=0.7, width=0.03, label=label[:20])
        bottom += vals
    ax_prob.set_ylim(0, 1.05)
    ax_prob.set_ylabel("P(state)", fontsize=9)
    ax_prob.set_title("HMM Regime Probabilities", fontsize=10, fontweight="bold")
    ax_prob.legend(fontsize=7, loc="upper right", ncol=2)
    ax_prob.grid(True, alpha=0.15)

    path = out_dir / "strategies_diagnostic.png"
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> Saved: {path}")

    # Strategy correlation heatmap
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    corr = signals.corr()
    cax = ax2.matshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    fig2.colorbar(cax, ax=ax2, shrink=0.7)
    ax2.set_xticks(range(n_strats))
    ax2.set_yticks(range(n_strats))
    ax2.set_xticklabels(signals.columns, rotation=45, ha="left", fontsize=8)
    ax2.set_yticklabels(signals.columns, fontsize=8)
    ax2.set_title("Strategy Signal Correlation", fontsize=13, fontweight="bold", pad=20)
    path2 = out_dir / "strategy_correlation.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"  -> Saved: {path2}")

    print("\n  DONE. All strategy/ensemble diagnostics complete.")


if __name__ == "__main__":
    run_strategy_diagnostic()
