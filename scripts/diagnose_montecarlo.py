"""
Diagnostic Script 3: Monte Carlo + Risk Dashboard Visual Validation
====================================================================
Full pipeline: Features -> HMM -> MC fit -> simulate -> risk profile.
Shows price paths fan chart (projection cone), return distribution,
drawdown distribution, and risk dashboard text.

Run:  python scripts/diagnose_montecarlo.py
Output: outputs/diagnostics/montecarlo_diagnostic.png
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


def generate_data(n=1000, seed=42):
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
    df["taker_buy_volume"] = df["volume"] * rng.uniform(0.3, 0.7, n)
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


def run_mc_diagnostic():
    print("=" * 60)
    print("  MONTE CARLO + RISK DIAGNOSTIC")
    print("=" * 60)

    print("\n[1/7] Generating synthetic data...")
    raw = generate_data()

    print("[2/7] Computing features...")
    from apollo.features.pipeline import FeaturePipeline
    pipe = FeaturePipeline(warmup_bars=50)
    features, _ = pipe.compute(raw)
    for col in ["open", "high", "low", "close", "volume"]:
        features[col] = raw[col].loc[features.index]

    print("[3/7] Fitting HMM Regime Detector...")
    from apollo.models.regime import RegimeDetector
    detector = RegimeDetector()
    detector.fit(features)
    regime_df = detector.predict(features)
    print(f"  Labels: {detector.label_map}")

    print("[4/7] Fitting Monte Carlo Simulator...")
    from apollo.models.monte_carlo import MonteCarloSimulator, MCConfig
    mc = MonteCarloSimulator(MCConfig(
        horizon=48,
        n_scenarios=2000,
        seed=42
    ))
    log_rets = features["log_return"]
    regimes = regime_df["hmm_regime"].fillna(0).astype(int)
    mc.fit(log_rets, regimes)
    print(f"  Residual buckets: {list(mc._resid_by_regime.keys())}")
    for rid, resid in mc._resid_by_regime.items():
        print(f"    Regime {rid}: {len(resid)} residuals, std={resid.std():.3f}")

    print("[5/7] Simulating price paths...")
    current_price = features["close"].iloc[-1]
    current_regime = int(regimes.iloc[-1])
    paths = mc.simulate(current_price, current_regime)
    print(f"  Paths shape: {paths.shape}")
    print(f"  Current price: ${current_price:,.2f}")
    print(f"  Current regime: {current_regime} ({detector.label_map.get(current_regime, '?')})")

    stats = MonteCarloSimulator.path_statistics(paths, current_price)
    print(f"  E[return]: {stats['expected_return']*100:+.2f}%")
    print(f"  VaR 5%:    {stats['VaR_5pct']*100:+.2f}%")
    print(f"  CVaR 5%:   {stats['CVaR_5pct']*100:+.2f}%")
    print(f"  P(profit): {stats['prob_profit']*100:.1f}%")

    print("[6/7] Computing Risk Dashboard...")
    from apollo.execution.risk import RiskDashboard
    risk = RiskDashboard()
    ensemble_signal = 0.35  # simulated positive signal
    profile = risk.compute_profile(paths, current_price, ensemble_signal)
    print(RiskDashboard.format_profile(profile))

    # Dynamic sizing demo
    from apollo.execution.risk import RiskDashboard as RD
    regime_label = detector.label_map.get(current_regime, "Unknown")
    for dd in [0, 5, 10, 15, 20]:
        size = RD.dynamic_size(
            kelly=profile["kelly_fraction_pct"] / 100,
            regime_label=regime_label,
            confidence="HIGH",
            current_drawdown_pct=dd,
            capital=10000.0,
        )
        print(f"  DD={dd:2d}% -> Size=${size:,.2f}")

    print("[7/7] Plotting MC diagnostic...")
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(3, 3, figure=fig, hspace=0.30, wspace=0.25)
    fig.suptitle(f"Monte Carlo Diagnostic -- {paths.shape[0]} scenarios x {paths.shape[1]} steps",
                 fontsize=16, fontweight="bold", y=0.995)

    # Panel 1: Projection Cone (full width)
    ax1 = fig.add_subplot(gs[0, :])
    cone = MonteCarloSimulator.cone_percentiles(paths)

    # Historical tail
    hist_len = min(100, len(features))
    hist_idx = list(range(-hist_len, 0))
    hist_prices = features["close"].iloc[-hist_len:].values
    future_idx = list(range(0, paths.shape[1]))

    ax1.plot(hist_idx, hist_prices, color="#1565C0", linewidth=1.2, label="Historical")

    # Cone fills
    fills = [(5, 95, "#E3F2FD", "5-95% CI"), (10, 90, "#BBDEFB", "10-90%"),
             (25, 75, "#90CAF9", "25-75%")]
    for lo, hi, color, label in fills:
        ax1.fill_between(future_idx, cone[lo], cone[hi], color=color, alpha=0.8, label=label)
    ax1.plot(future_idx, cone[50], color="#1565C0", linewidth=1.5, label="Median", linestyle="--")

    # Sample paths
    n_sample = min(50, paths.shape[0])
    for i in range(n_sample):
        ax1.plot(future_idx, paths[i], color="#90A4AE", alpha=0.05, linewidth=0.3)

    ax1.axhline(current_price, color="#FF5722", linestyle=":", linewidth=0.8, alpha=0.6)
    ax1.axvline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    ax1.text(0, current_price * 1.01, f"${current_price:,.0f}", fontsize=8, color="#FF5722")
    ax1.set_xlabel("Bars (negative=history, positive=forecast)", fontsize=9)
    ax1.set_ylabel("Price ($)", fontsize=9)
    ax1.set_title("Price Projection Cone + Sample Paths", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.2)

    # Panel 2: Return Distribution
    ax2 = fig.add_subplot(gs[1, 0])
    final_rets = (paths[:, -1] - current_price) / current_price * 100
    bins = np.linspace(final_rets.min(), final_rets.max(), 50)
    ax2.hist(final_rets, bins=bins, color="#2196F3", alpha=0.7, edgecolor="white", linewidth=0.5)
    ax2.axvline(0, color="#FF5722", linewidth=1, linestyle="--", label="Breakeven")
    ax2.axvline(stats["VaR_5pct"] * 100, color="#E53935", linewidth=1.5, linestyle="-", label=f"VaR 5%: {stats['VaR_5pct']*100:.2f}%")
    ax2.axvline(stats["CVaR_5pct"] * 100, color="#B71C1C", linewidth=1.5, linestyle=":", label=f"CVaR 5%: {stats['CVaR_5pct']*100:.2f}%")
    ax2.axvline(stats["expected_return"] * 100, color="#4CAF50", linewidth=1.5, label=f"E[r]: {stats['expected_return']*100:.2f}%")
    ax2.set_xlabel("Return (%)", fontsize=9)
    ax2.set_ylabel("Count", fontsize=9)
    ax2.set_title("Final Return Distribution", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.2)

    # Panel 3: Drawdown Distribution
    ax3 = fig.add_subplot(gs[1, 1])
    running_max = np.maximum.accumulate(paths, axis=1)
    drawdowns = (paths - running_max) / running_max
    max_dd = np.min(drawdowns, axis=1) * 100
    ax3.hist(max_dd, bins=40, color="#E53935", alpha=0.7, edgecolor="white", linewidth=0.5)
    ax3.axvline(np.median(max_dd), color="#FF5722", linewidth=1.5, linestyle="--",
                label=f"Median MDD: {np.median(max_dd):.2f}%")
    ax3.axvline(np.percentile(max_dd, 5), color="#B71C1C", linewidth=1.5, linestyle=":",
                label=f"5th pct: {np.percentile(max_dd, 5):.2f}%")
    ax3.set_xlabel("Max Drawdown (%)", fontsize=9)
    ax3.set_ylabel("Count", fontsize=9)
    ax3.set_title("Max Drawdown Distribution", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.2)

    # Panel 4: Probability Profit Over Time
    ax4 = fig.add_subplot(gs[1, 2])
    prob_profit_at_step = np.mean(paths > current_price, axis=0) * 100
    ax4.plot(np.arange(paths.shape[1]), prob_profit_at_step, color="#4CAF50", linewidth=1.5)
    ax4.axhline(50, color="gray", linestyle=":", linewidth=0.5)
    ax4.fill_between(np.arange(paths.shape[1]), 50, prob_profit_at_step,
                     where=prob_profit_at_step > 50, color="#4CAF50", alpha=0.2)
    ax4.fill_between(np.arange(paths.shape[1]), 50, prob_profit_at_step,
                     where=prob_profit_at_step < 50, color="#E53935", alpha=0.2)
    ax4.set_xlabel("Step", fontsize=9)
    ax4.set_ylabel("P(profit) %", fontsize=9)
    ax4.set_title("Probability of Profit Over Horizon", fontsize=11, fontweight="bold")
    ax4.set_ylim(0, 100)
    ax4.grid(True, alpha=0.2)

    # Panel 5: Volatility term structure
    ax5 = fig.add_subplot(gs[2, 0])
    step_returns = np.diff(paths, axis=1) / paths[:, :-1]
    vol_per_step = np.std(step_returns, axis=0) * 100
    ax5.plot(np.arange(len(vol_per_step)), vol_per_step, color="#9C27B0", linewidth=1.2)
    ax5.set_xlabel("Step", fontsize=9)
    ax5.set_ylabel("Cross-sectional Vol (%)", fontsize=9)
    ax5.set_title("Volatility Term Structure", fontsize=11, fontweight="bold")
    ax5.grid(True, alpha=0.2)

    # Panel 6: Risk Dashboard Text
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis("off")
    dashboard_text = RiskDashboard.format_profile(profile)
    ax6.text(0.05, 0.95, dashboard_text, transform=ax6.transAxes,
             fontfamily="monospace", fontsize=9, verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5", alpha=0.8))
    ax6.set_title("Risk Dashboard", fontsize=11, fontweight="bold")

    # Panel 7: Dynamic Sizing Curve
    ax7 = fig.add_subplot(gs[2, 2])
    dds = np.linspace(0, 20, 100)
    sizes = [RD.dynamic_size(
        kelly=profile["kelly_fraction_pct"] / 100,
        regime_label=regime_label,
        confidence="HIGH",
        current_drawdown_pct=dd,
        capital=10000.0,
    ) for dd in dds]
    ax7.plot(dds, sizes, color="#1565C0", linewidth=1.5)
    ax7.fill_between(dds, 0, sizes, color="#BBDEFB", alpha=0.4)
    ax7.axvline(5, color="#FF9800", linestyle="--", linewidth=0.8, alpha=0.6, label="5% DD")
    ax7.axvline(10, color="#FF5722", linestyle="--", linewidth=0.8, alpha=0.6, label="10% DD")
    ax7.axvline(15, color="#E53935", linestyle="--", linewidth=0.8, alpha=0.6, label="15% DD (stop)")
    ax7.set_xlabel("Portfolio Drawdown (%)", fontsize=9)
    ax7.set_ylabel("Position Size ($)", fontsize=9)
    ax7.set_title("Dynamic Sizing vs Drawdown", fontsize=11, fontweight="bold")
    ax7.legend(fontsize=7)
    ax7.grid(True, alpha=0.2)

    path_out = out_dir / "montecarlo_diagnostic.png"
    fig.savefig(path_out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> Saved: {path_out}")

    # Bonus: Regime-comparison MC
    print("\n  [Bonus] Per-regime MC comparison...")
    fig3, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig3.suptitle("MC Paths by Regime", fontsize=14, fontweight="bold")

    for sid in range(4):
        ax = axes[sid // 2, sid % 2]
        label = detector.label_map.get(sid, f"State {sid}")
        try:
            p = mc.simulate(current_price, sid)
            cone_r = MonteCarloSimulator.cone_percentiles(p)
            ax.fill_between(range(p.shape[1]), cone_r[5], cone_r[95], alpha=0.3, color="#90CAF9")
            ax.fill_between(range(p.shape[1]), cone_r[25], cone_r[75], alpha=0.5, color="#42A5F5")
            ax.plot(range(p.shape[1]), cone_r[50], color="#1565C0", linewidth=1.5)
            ax.axhline(current_price, color="#FF5722", linewidth=0.8, linestyle=":")
            st = MonteCarloSimulator.path_statistics(p, current_price)
            ax.set_title(f"Regime {sid}: {label[:30]}\nE[r]={st['expected_return']*100:+.2f}%, VaR5={st['VaR_5pct']*100:+.2f}%",
                        fontsize=9)
        except Exception as e:
            ax.text(0.5, 0.5, f"Failed: {e}", transform=ax.transAxes, ha="center")
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Price ($)")

    path3 = out_dir / "montecarlo_per_regime.png"
    fig3.savefig(path3, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig3)
    print(f"  -> Saved: {path3}")

    print("\n  DONE. Monte Carlo diagnostics complete.")


if __name__ == "__main__":
    run_mc_diagnostic()
