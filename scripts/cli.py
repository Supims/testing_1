"""
CLI Entry Point
=================
Production CLI for Apollo Quant.

Usage:
    python scripts/cli.py check            # Validate setup
    python scripts/cli.py scan             # Single scan cycle
    python scripts/cli.py scan --pairs 3   # Scan top 3 pairs
    python scripts/cli.py run              # Start agent loop
    python scripts/cli.py run --once       # Single agent cycle
    python scripts/cli.py status           # Show status
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
_root = Path(__file__).resolve().parent.parent / "src"
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(name)-20s] %(levelname)-7s %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")
    # Suppress noisy third-party loggers in non-verbose mode
    if not verbose:
        for name in ["urllib3", "hmmlearn", "httpx", "httpcore"]:
            logging.getLogger(name).setLevel(logging.WARNING)


def cmd_check(args):
    """Validate the environment and configuration."""
    print("=" * 50)
    print("  Apollo Quant -- Environment Check")
    print("=" * 50)

    errors = []

    # Python version
    v = sys.version_info
    print(f"\n  Python        : {v.major}.{v.minor}.{v.micro}", end="")
    if v.minor < 10 or v.minor > 12:
        print(" [!] Recommended: 3.10-3.12")
        errors.append("Python version outside recommended range")
    else:
        print(" [OK]")

    # Dependencies
    deps = ["numpy", "pandas", "sklearn", "xgboost", "hmmlearn", "ta",
            "pydantic", "openai", "scipy"]
    print("\n  Dependencies:")
    for dep in deps:
        try:
            mod = __import__(dep)
            ver = getattr(mod, "__version__", "?")
            print(f"    {dep:20s} {ver:10s} [OK]")
        except ImportError:
            print(f"    {dep:20s} {'MISSING':10s} [FAIL]")
            errors.append(f"Missing dependency: {dep}")

    # Configuration
    print("\n  Configuration:")
    try:
        from apollo.config import settings
        provider = settings.active_ai_provider
        if provider:
            print(f"    AI Provider : {provider} [OK]")
        else:
            print("    AI Provider : NONE [FAIL]")
            errors.append("No AI provider configured. Set an API key in .env")

        print(f"    Tier        : {settings.apollo_default_tier}")
        print(f"    Max Pairs   : {settings.apollo_max_pairs}")
        print(f"    Budget      : ${settings.apollo_daily_budget}/day")

        if settings.has_telegram:
            print(f"    Telegram    : configured [OK]")
        else:
            print(f"    Telegram    : not configured (optional)")
    except Exception as e:
        print(f"    Config load FAILED: {e}")
        errors.append(f"Config error: {e}")

    # Binance connectivity
    print("\n  Connectivity:")
    try:
        from apollo.data.client import BinanceClient
        client = BinanceClient()
        data = client.get_futures_klines("BTCUSDT", "1h",
            "2026-04-04 00:00:00", "2026-04-04 02:00:00")
        if data is not None and len(data) > 0:
            print(f"    Binance     : connected ({len(data)} bars) [OK]")
        else:
            print("    Binance     : no data returned [WARN]")
    except Exception as e:
        print(f"    Binance     : FAILED ({e}) [FAIL]")
        errors.append(f"Binance connection failed: {e}")

    # AI test
    if provider:
        try:
            from apollo.ai.providers import get_provider
            from apollo.ai.budget import TokenBudget, MODEL_CATALOG
            p = get_provider()
            bm = TokenBudget()
            model_key = bm.select_model()
            model_name = MODEL_CATALOG[model_key].name
            text, in_t, out_t = p.call(
                model=model_name,
                system="You are a test bot.",
                user="Reply with just: OK",
            )
            if text and len(text) > 0:
                print(f"    AI ({provider:8s}): {text.strip()[:20]} [OK]")
            else:
                print(f"    AI ({provider:8s}): empty response [WARN]")
        except Exception as e:
            print(f"    AI ({provider:8s}): FAILED ({str(e)[:50]}) [FAIL]")
            errors.append(f"AI provider test failed: {e}")

    # Summary
    print("\n" + "=" * 50)
    if errors:
        print(f"  {len(errors)} issue(s) found:")
        for e in errors:
            print(f"    - {e}")
    else:
        print("  All checks passed. Ready to run!")
    print("=" * 50)

    return len(errors) == 0


def cmd_scan(args):
    """Run a single scan with AI analysis."""
    from datetime import datetime, timezone, timedelta
    from apollo.config import settings
    from apollo.core.scanner import Scanner
    from apollo.ai.brain import Brain

    max_pairs = args.pairs or settings.apollo_max_pairs

    print("=" * 50)
    print("  Apollo Quant -- Single Scan")
    print("=" * 50)

    scanner = Scanner()
    brain = Brain()

    # Train or load models
    models_dir = settings.models_dir / "scanner"
    if (models_dir / "scanner_meta.json").exists():
        try:
            scanner = Scanner.load(str(models_dir))
            print("  Models: loaded from disk")
        except Exception as e:
            print(f"  Models: load failed ({e}), training fresh...")
            _train_scanner(scanner, settings, models_dir)
    else:
        print("  Models: not found, training fresh...")
        _train_scanner(scanner, settings, models_dir)

    # Discover pairs
    print(f"\n  Discovering top {max_pairs} pairs by volume...")
    old_max = settings.apollo_max_pairs
    settings.apollo_max_pairs = max_pairs

    t0 = time.time()
    scan_result = scanner.scan()
    elapsed = time.time() - t0

    settings.apollo_max_pairs = old_max

    results = scan_result.get("results", [])
    valid_results = [r for r in results if "error" not in r]

    print(f"  Scanned {len(results)} pairs in {elapsed:.1f}s")
    print()

    # Show results table
    if valid_results:
        print(f"  {'Pair':15s} {'Regime':30s} {'Ensemble':>10s} {'Payoff':>8s} {'OOD':>5s}")
        print("  " + "-" * 72)
        for r in valid_results:
            regime = r.get("regime", {})
            signals = r.get("signals", {})
            risk = r.get("risk", {})
            print(
                f"  {r['symbol']:15s} "
                f"{regime.get('label', '?'):30s} "
                f"{signals.get('ensemble', 0):>+10.4f} "
                f"{risk.get('payoff_ratio', 0):>7.2f}:1 "
                f"{'YES' if regime.get('is_ood') else 'no':>5s}"
            )

    if not valid_results:
        print("  No valid results to analyze.")
        return

    # AI Analysis
    print(f"\n  Running AI analysis on {len(valid_results)} pairs...")
    decisions = brain.analyze_scan(
        scan_results=valid_results,
        scorecard_summary=scan_result.get("scorecard_summary"),
        enrichment_summary=scan_result.get("enrichment_summary"),
        scan_id=scan_result.get("scan_id", ""),
        correlation_prompt=scan_result.get("correlation_prompt", ""),
        events_prompt=scan_result.get("events_prompt", ""),
    )

    # Show decisions
    print(f"\n  {'Pair':15s} {'Decision':8s} {'Conf':6s} {'Reasoning'}")
    print("  " + "-" * 72)
    for d in decisions:
        print(f"  {d.symbol:15s} {d.action:8s} {d.confidence:6s} {d.reasoning[:50]}")

    actionable = [d for d in decisions if d.is_actionable]
    print(f"\n  Total: {len(decisions)} decisions, {len(actionable)} actionable")
    print(f"  Cost: ${brain.budget.get_daily_spend():.4f}")


def _train_scanner(scanner, settings, models_dir):
    """Train scanner models from scratch."""
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    end_str = now.strftime("%Y-%m-%d %H:%M:%S")
    start_str = (now - timedelta(days=90)).strftime("%Y-%m-%d %H:%M:%S")
    scanner.train(["BTCUSDT", "ETHUSDT"], start_str, end_str)
    scanner.save_models(str(models_dir))
    print("  Models: trained and saved")


def cmd_run(args):
    """Start the autonomous agent."""
    from apollo.agent import Agent
    agent = Agent()
    if args.once:
        print("Running single cycle...")
        agent._ensure_trained()
        agent.run_once()
    else:
        print("Starting agent loop (Ctrl+C to stop)...")
        agent.run()


def cmd_status(args):
    """Show current status."""
    from apollo.config import settings
    print(settings.status())

    try:
        from apollo.ai.budget import TokenBudget
        budget = TokenBudget()
        print()
        print(budget.status())
    except Exception:
        pass

    try:
        from apollo.ai.memory import AIMemory
        memory = AIMemory()
        print()
        print(memory.format_stats())
    except Exception:
        pass

    try:
        from apollo.ai.prompt_log import PromptLogger
        plog = PromptLogger()
        print()
        print(plog.format_stats())
    except Exception:
        pass

    try:
        from apollo.trading.portfolio import Portfolio
        portfolio = Portfolio()
        print()
        print(portfolio.summary_text())
    except Exception:
        pass

    try:
        from apollo.ai.quality import DecisionTracker
        qt = DecisionTracker()
        print()
        print(qt.format_stats())
    except Exception:
        pass

    try:
        from apollo.models.retrain import ModelRetrainer
        rt = ModelRetrainer()
        print()
        print(rt.format_status())
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Apollo Quant -- AI Crypto Trading Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/cli.py check              Validate setup
  python scripts/cli.py scan               Single scan (default pairs)
  python scripts/cli.py scan --pairs 3     Scan top 3 pairs
  python scripts/cli.py run                Start agent loop
  python scripts/cli.py run --once         Single agent cycle
  python scripts/cli.py status             Show status + budget + memory
""",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("check", help="Validate environment and configuration")

    scan_parser = sub.add_parser("scan", help="Run a single scan with AI analysis")
    scan_parser.add_argument("--pairs", type=int, default=None,
                             help="Number of pairs to scan (default: from config)")

    run_parser = sub.add_parser("run", help="Start autonomous agent loop")
    run_parser.add_argument("--once", action="store_true", help="Run a single cycle only")

    sub.add_parser("status", help="Show status, budget, and memory stats")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command == "check":
        cmd_check(args)
    elif args.command == "scan":
        cmd_scan(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
