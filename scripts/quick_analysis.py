"""Quick signal analysis + scorecard test."""
import sys
sys.path.insert(0, "src")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000
ts = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
ret = np.zeros(n)
ret[:200] = np.random.normal(0.0001, 0.003, 200)
ret[200:400] = np.random.normal(0.002, 0.008, 200)
ret[400:500] = np.random.normal(-0.003, 0.015, 100)
ret[500:700] = np.random.normal(0.0, 0.004, 200)
ret[700:850] = np.random.normal(0.001, 0.005, 150)
ret[850:] = np.random.normal(-0.001, 0.010, n - 850)
p = 50000 * np.exp(np.cumsum(ret))
h = p * (1 + np.abs(np.random.normal(0, 0.003, n)))
lo = p * (1 - np.abs(np.random.normal(0, 0.003, n)))
v = np.random.lognormal(10, 1, n)
v[400:500] *= 3
v[850:] *= 2

df = pd.DataFrame({"open": np.roll(p, 1), "high": h, "low": lo, "close": p, "volume": v}, index=ts)
df.iloc[0, 0] = df.iloc[0, 3]
tf = np.random.uniform(0.3, 0.7, n)
tf[200:400] = np.random.uniform(0.55, 0.75, 200)
tf[400:500] = np.random.uniform(0.2, 0.4, 100)
df["taker_buy_volume"] = df["volume"] * tf
df["taker_sell_volume"] = df["volume"] - df["taker_buy_volume"]
fu = np.zeros(n)
fu[:200] = np.random.normal(0.0001, 0.0002, 200)
fu[200:400] = np.random.normal(0.0005, 0.0003, 200)
fu[400:500] = np.random.normal(-0.0008, 0.0005, 100)
fu[500:] = np.random.normal(0.0001, 0.0003, n - 500)
df["funding_rate"] = fu
df["open_interest"] = np.random.lognormal(12, 0.3, n)
df["premium_index"] = np.random.normal(0.0, 0.001, n)
df["spot_close"] = df["close"] * (1 + np.random.normal(0, 0.0005, n))

from apollo.features.pipeline import FeaturePipeline
feat, _ = FeaturePipeline(warmup_bars=50).compute(df)
for c in ["open", "high", "low", "close", "volume"]:
    feat[c] = df[c].loc[feat.index]

from apollo.models.strategies import compute_all, ALL_STRATEGY_NAMES
sigs = compute_all(feat)

lines = []
lines.append("STRATEGY             | Zero% | Uniq | Std   | Range")
lines.append("-" * 65)
for s in ALL_STRATEGY_NAMES:
    sig = sigs[s]
    z = (sig == 0).mean() * 100
    u = sig.round(3).nunique()
    lines.append(f"{s:20s} | {z:5.1f} | {u:4d} | {sig.std():.3f} | [{sig.min():.3f},{sig.max():.3f}]")

# Scorecard test
from apollo.models.scorecard import StrategyScorecard
returns = feat["close"].pct_change().fillna(0)
sc = StrategyScorecard()
card = sc.compute(sigs, returns)
summary = sc.summary(card)

lines.append("")
lines.append("SCORECARD SUMMARY (last 24 bars):")
lines.append(f"{'Strategy':20s} | {'IC':>7s} | {'Hit%':>5s} | {'Conf':>6s} | {'Pers':>5s}")
lines.append("-" * 60)
for strat in ALL_STRATEGY_NAMES:
    s = summary[strat]
    lines.append(
        f"{strat:20s} | {s['recent_ic']:+.4f} | {s['recent_hit_rate']:.2f} | {s['confidence']:+.4f} | {s['persistence']:.2f}"
    )
cs = summary["_cross"]
lines.append("")
lines.append(f"Cross-Agreement:  {cs['agreement']:.2f}")
lines.append(f"Cross-Conviction: {cs['conviction']:.3f}")
lines.append(f"Dominant Dir:     {cs['dominant_direction']:+.3f}")
lines.append(f"Dispersion:       {cs['dispersion']:.3f}")

# Enrichment test
from apollo.models.enrichment import SignalEnrichment
enricher = SignalEnrichment()
enrichment = enricher.compute(sigs)
lines.append("")
lines.append(f"Enrichment columns: {len(enrichment.columns)}")
lines.append(f"Enrichment sample cols: {list(enrichment.columns[:10])}")

output = "\n".join(lines)
print(output)
with open("outputs/signal_analysis_v3.txt", "w") as f:
    f.write(output)
