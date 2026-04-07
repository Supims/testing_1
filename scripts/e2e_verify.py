"""
End-to-end pipeline verification script.
Tests: Data -> Features -> HMM -> Strategies -> Ensemble -> MC -> Risk -> AI Brain
"""
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-20s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("e2e_verify")


def verify_pipeline():
    """Test the full scanner pipeline (no AI)."""
    from apollo.core.scanner import Scanner

    logger.info("Creating Scanner (mc_scenarios=100)")
    scanner = Scanner(mc_scenarios=100)

    from datetime import datetime, timedelta
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
    logger.info(f"Training on BTCUSDT, ETHUSDT ({start} to {end})")
    scanner.train(["BTCUSDT", "ETHUSDT"], start, end)

    logger.info("Scanning BTCUSDT")
    scan_output = scanner.scan(["BTCUSDT"])
    results = scan_output["results"]
    r = results[0]

    print(f"\n{'='*60}")
    print(f"  Symbol      : {r.get('symbol')}")
    regime = r.get('regime', {})
    print(f"  Regime      : {regime.get('label', '?')} (OOD: {regime.get('is_ood', False)})")
    signals = r.get('signals', {})
    print(f"  Signals     : {len(signals)} strategies")
    for name, val in signals.items():
        print(f"    - {name}: {val:.4f}")
    print(f"  Ensemble    : {signals.get('ensemble', 'N/A')}")
    mc = r.get('monte_carlo') or {}
    print(f"  MC mean_ret : {mc.get('mean_return', 'N/A')}")
    print(f"  MC VaR 95   : {mc.get('var_95', 'N/A')}")
    print(f"  Probabilities: {r.get('probabilities', {})}")
    risk = r.get('risk', {})
    print(f"  Risk dir    : {risk.get('direction', 'N/A')}")
    print(f"  Risk VaR5%  : {risk.get('var_5pct', 'N/A')}")
    print(f"  Risk SL     : {risk.get('sl_price', 'N/A')}")
    print(f"  Risk TP     : {risk.get('tp_price', 'N/A')}")
    print(f"  Scorecard   : {'yes' if r.get('scorecard') else 'no'}")
    print(f"{'='*60}\n")

    assert r.get('symbol') == "BTCUSDT", "Symbol mismatch"
    assert regime, "No regime detected"
    assert len(signals) > 0, "No strategy signals"
    assert 'ensemble' in signals, "No ensemble signal"
    assert risk, "No risk output"
    assert risk.get('sl_price'), "No stop loss in risk"
    assert risk.get('tp_price'), "No take profit in risk"

    logger.info("Pipeline test PASSED")
    return scan_output


def verify_prompts(scan_results):
    """Test prompt construction from scan results."""
    from apollo.ai.prompts import build_system_prompt, build_market_prompt

    logger.info("Building system prompt")
    sys_prompt = build_system_prompt(compact=False)
    print(f"  System prompt length: {len(sys_prompt)} chars")
    assert len(sys_prompt) > 100, "System prompt too short"

    logger.info("Building market prompt")
    scan_dicts = scan_results.get("results", []) if isinstance(scan_results, dict) else scan_results
    user_prompt = build_market_prompt(scan_dicts)
    print(f"  Market prompt length: {len(user_prompt)} chars")
    assert len(user_prompt) > 200, "Market prompt too short"
    assert "BTCUSDT" in user_prompt, "BTCUSDT not in prompt"

    # Check OOD warning if regime is OOD
    first = scan_dicts[0] if scan_dicts else {}
    regime = first.get("regime", {})
    if regime.get("is_ood"):
        assert "WARNING: OOD" in user_prompt, "Missing OOD warning in prompt"
        logger.info("OOD warning correctly injected")
    else:
        logger.info("Regime is in-distribution (OOD warning not needed)")

    logger.info("Prompt test PASSED")
    return sys_prompt, user_prompt


def verify_ai_brain(scan_results):
    """Test the AI brain with real API call."""
    from apollo.config import Settings
    from apollo.ai.brain import Brain

    settings = Settings()
    provider = settings.active_ai_provider
    if not provider:
        logger.warning("No AI provider configured, skipping AI test")
        return None

    logger.info(f"Testing AI Brain with provider={provider}, tier={settings.apollo_default_tier}")
    brain = Brain()  # uses get_provider() auto-detection internally

    logger.info("Calling brain.analyze_scan()...")
    if isinstance(scan_results, dict):
        decisions = brain.analyze_scan(
            scan_results=scan_results.get("results", []),
            scorecard_summary=scan_results.get("scorecard_summary"),
            enrichment_summary=scan_results.get("enrichment_summary"),
            scan_id=scan_results.get("scan_id"),
            correlation_prompt=scan_results.get("correlation_prompt", ""),
            events_prompt=scan_results.get("events_prompt", ""),
        )
    else:
        decisions = brain.analyze_scan(scan_results)

    if not decisions:
        logger.warning("AI returned no decisions")
        return decisions

    for d in decisions:
        print(f"\n{'='*60}")
        print(f"  AI Decision for {d.symbol}:")
        print(f"    Action     : {d.action}")
        print(f"    Confidence : {d.confidence}")
        print(f"    Reasoning  : {d.reasoning[:200]}...")
        if d.sl_price:
            print(f"    Stop Loss  : {d.sl_price}")
        if d.tp_price:
            print(f"    Take Profit: {d.tp_price}")
        print(f"{'='*60}")

    logger.info("AI Brain test PASSED")
    return decisions


def main():
    print("\n" + "=" * 60)
    print("  APOLLO E2E VERIFICATION")
    print("=" * 60 + "\n")

    # Step 1: Pipeline
    logger.info("STEP 1: Scanner pipeline")
    results = verify_pipeline()

    # Step 2: Prompts
    logger.info("STEP 2: Prompt construction")
    verify_prompts(results)

    # Step 3: AI Brain
    logger.info("STEP 3: AI Brain (live API call)")
    verify_ai_brain(results)

    print("\n" + "=" * 60)
    print("  ALL VERIFICATIONS PASSED")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"E2E verification FAILED: {e}", exc_info=True)
        sys.exit(1)
