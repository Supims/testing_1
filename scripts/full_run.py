"""
Full Tier-3 Run — All Discovered Pairs
=======================================
Runs the complete pipeline: Discovery → Train → Scan → AI Brain
Then dumps the full prompts and AI responses for analysis.
"""
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-22s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("full_run")


def main():
    t0 = time.time()

    # ── 1. Discover pairs ───────────────────────────────────────────────
    from apollo.data.discovery import PairDiscovery
    from apollo.config import Settings

    settings = Settings()
    discovery = PairDiscovery()
    max_pairs = settings.apollo_max_pairs
    symbols = discovery.discover(max_pairs=max_pairs)

    print(f"\n{'='*70}")
    print(f"  FULL TIER-3 RUN  |  {len(symbols)} pairs  |  {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}")
    print(f"  Provider: {settings.active_ai_provider}  |  Tier: {settings.apollo_default_tier}")
    print(f"  Pairs: {', '.join(symbols)}")
    print(f"{'='*70}\n")

    # ── 2. Train ────────────────────────────────────────────────────────
    from apollo.core.scanner import Scanner

    scanner = Scanner(mc_scenarios=500)
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start = (datetime.now(timezone.utc) - timedelta(days=60)).strftime("%Y-%m-%d")

    logger.info("Training on %s (%s → %s)", symbols[:2], start, end)
    scanner.train(symbols[:2], start, end)

    # ── 3. Scan all pairs ───────────────────────────────────────────────
    logger.info("Scanning %d pairs...", len(symbols))
    scan_output = scanner.scan(symbols)
    results = scan_output.get("results", [])
    ok = [r for r in results if "error" not in r]
    err = [r for r in results if "error" in r]

    logger.info("Scan done: %d OK, %d errors", len(ok), len(err))
    if err:
        for e in err:
            logger.warning("  FAILED: %s — %s", e.get("symbol"), e.get("error"))

    # ── 4. Print scan summary ───────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  SCAN RESULTS  ({len(ok)}/{len(results)} pairs)")
    print(f"{'─'*70}")
    for r in ok:
        sym = r.get("symbol", "?")
        regime = r.get("regime", {})
        signals = r.get("signals", {})
        risk = r.get("risk", {})
        ens = signals.get("ensemble", 0)
        print(
            f"  {sym:12s} | Regime: {regime.get('label','?'):30s} | "
            f"OOD: {'YES' if regime.get('is_ood') else 'no ':3s} | "
            f"Ens: {ens:+.4f} | "
            f"VaR5: {risk.get('var_5pct', 0):+.2f}% | "
            f"Payoff: {risk.get('payoff_ratio', 0):.2f}:1"
        )
    print(f"{'─'*70}\n")

    # ── 5. AI Brain ─────────────────────────────────────────────────────
    from apollo.ai.brain import Brain

    brain = Brain()

    logger.info("Calling AI Brain on %d pairs...", len(ok))
    decisions = brain.analyze_scan(
        scan_results=ok,
        scorecard_summary=scan_output.get("scorecard_summary"),
        enrichment_summary=scan_output.get("enrichment_summary"),
        scan_id=scan_output.get("scan_id"),
        correlation_prompt=scan_output.get("correlation_prompt", ""),
        events_prompt=scan_output.get("events_prompt", ""),
    )

    # ── 6. Print AI decisions ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  AI DECISIONS  ({len(decisions)} total)")
    print(f"{'='*70}")
    for d in decisions:
        actionable = "** ACTIONABLE **" if d.is_actionable else ""
        print(f"\n  [{d.action:5s}] {d.symbol:12s}  Confidence: {d.confidence:6s}  {actionable}")
        print(f"         Reasoning: {d.reasoning}")
        if d.sl_price:
            print(f"         SL: ${d.sl_price:,.2f}  |  TP: ${d.tp_price:,.2f}")
        if d.alert_type != "NONE":
            print(f"         Alert: {d.alert_type} @ ${d.alert_price:,.2f}")
        if d.self_notes:
            print(f"         Self-notes: {d.self_notes[:200]}")

    actionable_count = sum(1 for d in decisions if d.is_actionable)
    skip_count = sum(1 for d in decisions if d.action == "SKIP")
    print(f"\n  Summary: {actionable_count} actionable, {skip_count} skipped, {len(decisions)} total")

    # ── 7. Load and display the logged prompt/response ──────────────────
    print(f"\n{'='*70}")
    print(f"  PROMPT & RESPONSE ANALYSIS")
    print(f"{'='*70}")

    log_dir = Path("logs/prompts") / datetime.now().strftime("%Y-%m-%d")
    log_file = log_dir / "interactions.jsonl"

    if log_file.exists():
        lines = log_file.read_text(encoding="utf-8").strip().split("\n")
        # Get the last interaction (the one we just triggered)
        last = json.loads(lines[-1])

        sys_prompt = last.get("system_prompt", "")
        user_prompt = last.get("user_prompt", "")
        response = last.get("response", "")

        print(f"\n  Model: {last.get('model')}")
        print(f"  Tokens: {last.get('in_tokens')} in + {last.get('out_tokens')} out = {last.get('total_tokens')}")
        print(f"  Cost: ${last.get('cost_usd', 0):.4f}")

        print(f"\n  ── SYSTEM PROMPT ({len(sys_prompt)} chars) ──")
        print(f"  Sections: ", end="")
        sections = [line.strip() for line in sys_prompt.split("\n") if line.strip().startswith("## ")]
        print(", ".join(s.replace("## ", "") for s in sections))

        print(f"\n  ── USER PROMPT ({len(user_prompt)} chars) ──")
        print(f"  {'─'*60}")
        # Print with indentation
        for line in user_prompt.split("\n"):
            print(f"  {line}")
        print(f"  {'─'*60}")

        print(f"\n  ── AI RESPONSE ({len(response)} chars) ──")
        print(f"  {'─'*60}")
        for line in response.split("\n"):
            print(f"  {line}")
        print(f"  {'─'*60}")
    else:
        logger.warning("No interaction log found at %s", log_file)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  DONE in {elapsed:.1f}s  |  Cost: ${sum(last.get('cost_usd', 0) for _ in [1]):.4f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("Full run FAILED: %s", e, exc_info=True)
        sys.exit(1)
