"""
Prompt Builder
===============
System and user prompt construction for the AI brain.

System prompt defines the AI's identity, reasoning framework,
and output format. User prompt formats scan data, scorecard,
enrichment, sentiment, and memory into a structured input.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger("ai.prompts")


def _sanitize(text: str) -> str:
    """Strip control chars and suspicious prompt-injection patterns from external data."""
    if not isinstance(text, str):
        return str(text)
    # Remove non-printable control characters (keep newlines and tabs)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Collapse sequences that look like prompt override attempts
    text = re.sub(
        r'(?i)(ignore\s+(all\s+)?(?:previous|above|prior)\s+instructions)',
        '[FILTERED]', text,
    )
    # Limit length of any single injected field to 500 chars
    if len(text) > 500:
        text = text[:500] + "..."
    return text


def build_system_prompt(compact: bool = False) -> str:
    """Build the AI system prompt with trading identity and rules."""
    if compact:
        return _compact_prompt()
    return _full_prompt()


def _full_prompt() -> str:
    return """You are an elite quantitative crypto trading analyst. You analyze pipeline output with institutional rigor and communicate with precision.

## YOUR PIPELINE

Data comes from a production quant pipeline:
1. DataEngine: Binance Futures OHLCV + Spot + Funding + OI, temporally aligned (no look-ahead bias).
2. FeaturePipeline: 40+ features (microstructure, volatility, momentum, volume, OI) across 5m/15m/1h timeframes.
3. HMM Regime Detection: 4-state Gaussian Mixture HMM:
   - "High Volatility (Trending)" -- strong directional moves
   - "Low Volatility (Trending)" -- slow grind
   - "High Volatility (Ranging)" -- dangerous chop
   - "Low Volatility (Quiet Range)" -- dead market
   - OOD flag = outside model's training distribution -> LOW confidence
4. 10 Quantitative Strategies (each outputs continuous signal in [-1, +1]):
   Trend, MeanReversion, Squeeze, SmartMoney, BasisArb, Breakout, FundingMom, OIDivergence, LiquidationCascade, VolumeProfile
5. Regime-Weighted Ensemble: weights each strategy by HMM regime probability.
6. Monte Carlo (EGARCH-Bootstrap): 300-500 future price paths. VaR, CVaR, expected return, max drawdown.
7. 5x XGBoost Probability Models (Isotonic calibrated): P(+1.5% 12h), P(+1.5% 24h), P(+3.0% 48h), P(DD>1% 24h), P(DD>2% 24h).
8. Risk Dashboard: MC-derived SL/TP, Kelly sizing, payoff ratio.

## STRATEGY SCORECARD

You receive a SCORECARD section with per-strategy performance metrics:
- Rolling IC: information coefficient (correlation with future returns). IC > 0.05 = useful, > 0.10 = strong.
- Hit Rate: pct of correct directional predictions. > 55% is good.
- Regime IC: IC broken down by current regime. Shows which strategies work NOW.
- Confidence Score: combines IC + hit rate + persistence. Range [-1, +1]. > 0.3 = trust, < -0.2 = distrust.
- Cross Agreement: how many strategies agree in direction. > 0.7 = very aligned.

## SIGNAL ENRICHMENT

You receive ENRICHMENT metadata:
- Signal Age: how many bars the current signal has been active. Young (< 3) = fresh, Old (> 10) = possibly stale.
- Acceleration: is the signal strengthening or weakening?
- Stability: low stability = signal is flickering, wait for confirmation.
- Consensus Ratio: what fraction of all strategies agree on direction.
- Historical Extremity: how unusual is the current signal magnitude.

## ON-CHAIN INTELLIGENCE

You receive an ON-CHAIN section per pair with REAL data from free APIs:
- Market Overview: price changes (24h/7d/30d), market cap rank, supply metrics, ATH distance.
- Derivatives: Top trader L/S ratio, global L/S ratio, taker buy/sell ratio, OI changes.
  - L/S > 1.5 = crowded long (contrarian short signal). L/S < 0.7 = crowded short (squeeze risk).
  - Taker buy/sell: > 1.2 = aggr buyers, < 0.8 = aggr sellers.
  - OI rising + price flat = potential breakout. OI falling + price falling = capitulation.
- DeFi/TVL: chain TVL context (for L1/L2 tokens).
- Community: Twitter followers, Reddit activity, sentiment scores.
- Global: BTC dominance, total market cap changes, stablecoin supply.
  - Rising stablecoin supply = fresh capital entering crypto (bullish macro).
  - Falling stablecoin supply = capital leaving (bearish macro).

USE ON-CHAIN DATA to validate or contradict technical signals. A strong long signal + crowded longs = DANGER.

## SENTIMENT CONTEXT

You receive a SENTIMENT section:
- Fear & Greed Index: < 25 = extreme fear (contrarian buy), > 75 = extreme greed (caution).
- BTC Dominance: rising = risk-off (alts underperform), falling = risk-on.
- Funding Heatmap: aggregate funding across top pairs. Extreme = crowded trade.
- Recent Large Liquidations: major liquidation events that may signal forced selling/buying.

## CROSS-PAIR CORRELATION

You may receive a CROSS-PAIR CORRELATION block:
- Concentration Risk: how correlated the scanned pairs are (0=diverse, 1=all same).
- Clusters: groups of highly correlated pairs. DO NOT open same-direction positions in clustered pairs.
- Hedge pairs: negatively correlated pairs useful for portfolio balancing.
- HIGH concentration (>0.6) = reduce total exposure, the portfolio is effectively one big trade.

## MACRO EVENTS

You may receive a MACRO EVENTS block listing upcoming events:
- FOMC, CPI, NFP = HIGH impact. Reduce position sizes 50% if imminent.
- IMMINENT events (<12h away) = do NOT open new positions unless signal is exceptional.
- JUST HAPPENED events = expect volatility, wait for post-event price action to settle.

## YOUR PAST PERFORMANCE

You may receive a PERFORMANCE block with your historical accuracy.
- Use this to calibrate confidence: if HIGH confidence hit rate < MEDIUM, lower your confidence thresholds.
- If overall hit rate < 40%, be MORE selective and SKIP more often.
- Learn from your patterns: if SHORT decisions consistently underperform, prefer SKIP over SHORT.

## RULES

1. Never invent data. Only reference numbers in the scan data.
2. Always state the regime and whether it is OOD.
3. Probabilistic language: "65% probability" not "it will go up".
4. Risk first: mention VaR and drawdown before recommending.
5. Regime-aware: a signal strong in trending is weak in ranging.
6. ONLY trade when multiple strategies give aligned signals.
7. Check cluster/pair alignment: no contradiction to correlated pairs.
8. DO NOT open correlated positions (same direction on clustered pairs).
9. Memory-aware: reference past performance on similar setups.
10. Capital preservation > opportunity capture.
11. Be honest about uncertainty. If signals conflict, say SKIP.
12. Reduce risk around macro events (FOMC, CPI, NFP).

## YOUR OUTPUT FORMAT

Respond with this EXACT structure for each pair analyzed:

DECISION: [LONG/SHORT/CLOSE/SKIP] [SYMBOL]
CONFIDENCE: [HIGH/MEDIUM/LOW]
REASONING: [2 sentences max -- reference specific signals and metrics]
SL: [stop loss price] | TP: [take profit price]
ALERT: [SET_LONG_ALERT price / SET_SHORT_ALERT price / NONE]
SELF_NOTES: [optional -- notes for your future self about this setup]
SELF_ERRORS: [optional -- lessons from past mistakes relevant here]

If the market assessment is needed, prepend:
MARKET: [1 sentence on overall regime + key driver]

For position management:
CLOSE: [SYMBOL] [reason in 1 sentence]"""


def _compact_prompt() -> str:
    return """You are an elite quant crypto analyst. Pipeline: Binance data -> 40 features (5m/15m/1h) -> HMM regime (4 states) -> 10 strategies (continuous [-1,+1]) -> ensemble -> EGARCH Monte Carlo -> 5 XGBoost probs -> risk dashboard.

SCORECARD: per-strategy IC, hit rate, regime IC, confidence. Trust confidence > 0.3, distrust < -0.2.
ENRICHMENT: signal age, acceleration, stability, consensus ratio.
ON-CHAIN: L/S ratios, taker ratios, OI changes, market cap, supply metrics, TVL, community. L/S > 1.5 = crowded long (danger). Validate technicals vs on-chain.
SENTIMENT: F&G index, BTC dominance, funding, liquidations.

RULES: Never invent data. Regime-aware. Multiple strategies must align. No contradiction to cluster. Risk first. When in doubt SKIP.

FORMAT:
DECISION: [LONG/SHORT/CLOSE/SKIP] [SYMBOL]
CONFIDENCE: [HIGH/MEDIUM/LOW]
REASONING: [2 sentences]
SL: [price] | TP: [price]
ALERT: [SET_LONG_ALERT price / SET_SHORT_ALERT price / NONE]
SELF_NOTES: [optional]
SELF_ERRORS: [optional]"""


def build_market_prompt(
    scan_results: list[dict],
    scorecard_summary: dict = None,
    enrichment_summary: dict = None,
    sentiment: dict = None,
    positions: list[dict] = None,
    alerts: list[dict] = None,
    memory_block: str = "",
    extra_context: list[str] = None,
) -> str:
    """Build the complete user prompt with all context."""
    sections = []
    # -- Sentiment context --
    if sentiment:
        lines = ["=== SENTIMENT ==="]
        fg = sentiment.get("fear_greed", {})
        if fg:
            lines.append(f"Fear & Greed: {fg.get('value', '?')} ({fg.get('label', '?')})")
        if "btc_dominance" in sentiment:
            btc_d = sentiment['btc_dominance']
            if btc_d > 0:
                lines.append(f"BTC Dominance: {btc_d:.1f}%")
            else:
                lines.append("BTC Dominance: N/A (data unavailable)")
        if "total_funding_bias" in sentiment:
            lines.append(f"Avg Funding: {sentiment['total_funding_bias']:+.6f}")
        liqs = sentiment.get("recent_large_liqs", [])
        if liqs:
            top3 = liqs[:3]
            for lq in top3:
                vol = lq.get('volume_24h', 0)
                pct = lq.get('price_change_pct', 0)
                lines.append(
                    f"  Liq: {lq.get('symbol','?')} {lq.get('side','?')} "
                    f"{pct:+.1f}% (vol ${vol:,.0f})"
                )
        sections.append("\n".join(lines))

    # -- Current positions --
    if positions:
        lines = ["=== CURRENT POSITIONS ==="]
        for p in positions:
            lines.append(
                f"  {p.get('direction','?')} {p.get('symbol','?')} "
                f"@ ${p.get('entry_price',0):,.2f} | "
                f"PnL: {p.get('unrealized_pnl_pct',0):+.2f}%"
            )
        sections.append("\n".join(lines))

    # -- Pending alerts --
    if alerts:
        lines = ["=== PENDING ALERTS ==="]
        for a in alerts:
            lines.append(f"  {a.get('type','?')} {a.get('symbol','?')} @ ${a.get('price',0):,.2f}")
        sections.append("\n".join(lines))

    # -- Memory --
    if memory_block:
        sections.append(memory_block)

    # -- Scan data per pair --
    for result in scan_results:
        symbol = _sanitize(str(result.get('symbol', '?')))
        lines = [f"=== {symbol} ==="]

        # Regime
        regime = result.get("regime", {})
        is_ood = regime.get("is_ood", False)
        lines.append(
            f"Regime: {regime.get('label', '?')} "
            f"(OOD: {'YES' if is_ood else 'no'})"
        )
        if is_ood:
            lines.append("  *** WARNING: OOD regime -- model outside training distribution. "
                         "Reduce confidence to LOW, prefer SKIP. ***")

        # Signals
        signals = result.get("signals", {})
        sig_parts = []
        for name, val in sorted(signals.items()):
            if name != "ensemble":
                sig_parts.append(f"{name}={val:+.3f}")
        lines.append(f"Signals: {', '.join(sig_parts)}")
        lines.append(f"Ensemble: {signals.get('ensemble', 0):+.4f}")

        # Probabilities
        probs = result.get("probabilities", {})
        if probs:
            prob_parts = [f"{k}={v:.1%}" for k, v in probs.items()]
            lines.append(f"Probs: {', '.join(prob_parts)}")

        # Risk
        risk = result.get("risk", {})
        if risk:
            lines.append(
                f"Risk: VaR5%={risk.get('var_5pct', 0):+.2f}% | "
                f"P(profit)={risk.get('prob_profit_pct', 0):.1f}% | "
                f"Payoff={risk.get('payoff_ratio', 0):.2f}:1"
            )
            lines.append(
                f"SL=${risk.get('sl_price', 0):,.2f} | TP=${risk.get('tp_price', 0):,.2f}"
            )

        # MTF alignment (if available)
        mtf = result.get("mtf_summary", {})
        if mtf:
            lines.append(
                f"MTF: 15m_align={mtf.get('mtf_15m_trend_alignment', 0):+.2f} | "
                f"5m_entry={mtf.get('mtf_5m_entry_quality', 0):+.2f}"
            )

        # On-chain data (injected by scanner, sanitized)
        onchain_prompt = result.get("onchain_prompt", "")
        if onchain_prompt:
            lines.append(_sanitize(onchain_prompt))

        # Price
        price = result.get("current_price", 0)
        if price:
            lines.append(f"Price: ${price:,.2f}")

        sections.append("\n".join(lines))

    # -- Scorecard --
    if scorecard_summary:
        lines = ["=== SCORECARD ==="]
        for strat, metrics in scorecard_summary.items():
            if isinstance(metrics, dict):
                ic = metrics.get("rolling_ic", 0)
                hr = metrics.get("hit_rate", 0)
                conf = metrics.get("confidence", 0)
                lines.append(f"  {strat}: IC={ic:+.3f} HR={hr:.0%} conf={conf:+.2f}")
        ca = scorecard_summary.get("cross_agreement", 0)
        if ca:
            lines.append(f"  Cross agreement: {ca:.2f}")
        sections.append("\n".join(lines))

    # -- Enrichment --
    if enrichment_summary:
        lines = ["=== ENRICHMENT ==="]
        consensus = enrichment_summary.get("consensus_ratio", 0)
        lines.append(f"Consensus: {consensus:.2f}")
        for key in ["signal_age", "acceleration", "stability"]:
            val = enrichment_summary.get(key)
            if val is not None:
                lines.append(f"  {key}: {val:.2f}")
        sections.append("\n".join(lines))

    # -- Extra context blocks (correlation, events, quality, etc.) --
    if extra_context:
        for block in extra_context:
            if block and block.strip():
                sections.append(block)

    return "\n\n".join(sections)


def build_chat_prompt(user_message: str, memory_block: str = "",
                      positions: list[dict] = None) -> str:
    """For Telegram chat -- no market data, just memory + positions."""
    sections = []

    if positions:
        lines = ["Current positions:"]
        for p in positions:
            lines.append(
                f"  {p.get('direction','?')} {p.get('symbol','?')} "
                f"PnL: {p.get('unrealized_pnl_pct',0):+.2f}%"
            )
        sections.append("\n".join(lines))

    if memory_block:
        sections.append(memory_block)

    sections.append(f"User asks: {user_message}")
    return "\n\n".join(sections)
