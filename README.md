# 🚀 Apollo Quant Intelligence Core

> **AI-augmented probabilistic crypto trading engine** — Fully autonomous, self-learning, zero-cost data pipeline.

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)]()
[![Tests](https://img.shields.io/badge/Tests-217%20passed-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

---

## 🎯 What is Apollo?

Apollo is an **autonomous crypto trading agent** that combines:

- **10 quantitative strategies** (trend, mean-reversion, squeeze, smart money, etc.)
- **HMM regime detection** (4 market states: trending, volatile, ranging, quiet)
- **Monte Carlo simulation** (EGARCH-powered, 500 scenarios per pair)
- **5 XGBoost probability models** (P(+1.5% 12h), P(+3% 48h), P(drawdown), etc.)
- **AI reasoning** (Gemini/GPT/Claude analyzes all signals + on-chain data)
- **On-chain intelligence** (L/S ratios, taker flows, OI, TVL — all FREE)
- **Self-learning** (tracks its own accuracy, adjusts behavior)

It scans **20 pairs** every cycle, makes autonomous decisions, manages virtual positions with SL/TP, and alerts you on Telegram.

---

## ⚡ Quick Start

### 1. Clone & Setup

```bash
# Clone the repository
cd apollo-quant

# Option A: Interactive Setup Wizard (recommended)
python setup_wizard.py

# Option B: Manual setup
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -e ".[dev]"
```

### 2. Configure

```bash
# Copy the example config
cp .env.example .env
```

Edit `.env` with your API key:

```ini
# REQUIRED: Get a FREE key at https://aistudio.google.com/app/apikey
GOOGLE_API_KEY=your-actual-key-here

# OPTIONAL: Telegram alerts
TELEGRAM_BOT_TOKEN=your-bot-token
TELEGRAM_CHAT_ID=your-chat-id
```

> 💡 **Only `GOOGLE_API_KEY` is required.** Everything else has smart defaults.

### 3. Run

```bash
# Single scan cycle (recommended first run)
python -m scripts.cli run --once

# Autonomous loop (scans every 30min-2h based on market conditions)
python -m scripts.cli run

# Check status
python -m scripts.cli status
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    APOLLO QUANT v2.0                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─ DATA LAYER ──────────────────────────────────────────┐  │
│  │  Binance Futures API  →  OHLCV + Spot + Funding + OI  │  │
│  │  CoinGecko / DeFiLlama  →  On-Chain Intelligence      │  │
│  │  Fear & Greed Index  →  Sentiment                     │  │
│  └────────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ┌─ FEATURE ENGINE ─────────────────────────────────────┐   │
│  │  40+ features  ×  3 timeframes (5m / 15m / 1h)       │   │
│  │  RSI, MACD, Bollinger, ATR, Volume Z-score, OBV...   │   │
│  └───────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  ┌─ QUANT MODELS ────────────────────────────────────────┐  │
│  │  HMM (4 states)  →  Regime Detection                  │  │
│  │  10 Strategies   →  Continuous Signals [-1, +1]        │  │
│  │  Ensemble        →  Regime-Weighted Combined Signal    │  │
│  │  EGARCH MC       →  500 Scenario VaR/CVaR/Drawdown    │  │
│  │  5x XGBoost      →  Multi-Horizon Probabilities        │  │
│  │  Risk Dashboard  →  MC-derived SL/TP + Kelly Sizing    │  │
│  └───────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  ┌─ INTELLIGENCE LAYER ─────────────────────────────────┐   │
│  │  Cross-Pair Correlation  →  Cluster/Hedge Detection   │   │
│  │  Macro Events Calendar   →  FOMC/CPI/NFP Risk Adj.   │   │
│  │  Decision Quality Track  →  Self-Performance Aware    │   │
│  │  On-Chain Analytics      →  Smart Money Validation    │   │
│  └───────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  ┌─ AI BRAIN ────────────────────────────────────────────┐  │
│  │  Gemini 2.5 Pro  (Google AI)                           │  │
│  │  Budget-aware model selection (3 tiers)                │  │
│  │  Structured prompt: quant data + on-chain + memory     │  │
│  │  Output: LONG/SHORT/SKIP + SL/TP + confidence         │  │
│  └───────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  ┌─ EXECUTION ───────────────────────────────────────────┐  │
│  │  Paper Trader  →  Virtual positions + SL/TP + alerts   │  │
│  │  Portfolio     →  Equity tracking + PnL snapshots      │  │
│  │  Telegram      →  Real-time alerts + /ask chatbot      │  │
│  └───────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Project Structure

```
apollo-quant/
├── src/apollo/
│   ├── agent.py              # Autonomous loop (adaptive 30min-2h cycles)
│   ├── config.py             # Pydantic settings + auto-detection
│   ├── types.py              # Shared data structures
│   ├── errors.py             # Custom exceptions
│   │
│   ├── ai/                   # AI Brain (2,595 lines)
│   │   ├── brain.py          #   Orchestrator: prompt → LLM → decisions
│   │   ├── prompts.py        #   System + market prompts with all context
│   │   ├── providers.py      #   Google/OpenAI/Anthropic abstraction
│   │   ├── budget.py         #   3-tier budget management ($50/day default)
│   │   ├── memory.py         #   Predictions, self-notes, position journal
│   │   ├── parser.py         #   Structured AI response parser
│   │   ├── sentiment.py      #   Fear & Greed, BTC dominance, funding
│   │   ├── prompt_log.py     #   JSONL logging for analysis
│   │   └── quality.py        #   Decision accuracy tracking (self-learning)
│   │
│   ├── core/                 # Pipeline Core (1,256 lines)
│   │   ├── scanner.py        #   Full pipeline facade (scan 20 pairs)
│   │   ├── correlation.py    #   Cross-pair correlation + cluster detection
│   │   ├── events.py         #   Macro events calendar (FOMC, CPI, NFP)
│   │   └── alignment.py      #   Signal alignment utilities
│   │
│   ├── data/                 # Data Pipeline (1,435 lines)
│   │   ├── client.py         #   Binance API client (futures + spot)
│   │   ├── provider.py       #   Enriched dataset builder
│   │   ├── discovery.py      #   Auto-discover top 20 traded pairs
│   │   ├── onchain.py        #   CoinGecko + DeFiLlama + Binance derivs
│   │   └── cache.py          #   Parquet caching layer
│   │
│   ├── features/             # Feature Engineering (649 lines)
│   │   ├── pipeline.py       #   40+ technical indicators
│   │   └── indicators.py     #   Custom indicator implementations
│   │
│   ├── models/               # Quant Models (1,872 lines)
│   │   ├── regime.py         #   HMM 4-state regime detector
│   │   ├── strategies/       #   10 trading strategies
│   │   ├── ensemble.py       #   Regime-weighted strategy combination
│   │   ├── monte_carlo.py    #   EGARCH Monte Carlo (500 scenarios)
│   │   ├── scorecard.py      #   Per-strategy IC, hit rate, confidence
│   │   ├── enrichment.py     #   Signal age, acceleration, stability
│   │   ├── optimizer.py      #   Portfolio optimization
│   │   └── retrain.py        #   Auto-retrain (rolling 90d window)
│   │
│   ├── execution/            # Probability & Risk (643 lines)
│   │   ├── probability.py    #   5x XGBoost multi-horizon models
│   │   └── risk.py           #   Risk dashboard (VaR, Kelly, SL/TP)
│   │
│   ├── trading/              # Paper Trading (573 lines)
│   │   ├── paper.py          #   SQLite-backed virtual trader
│   │   └── portfolio.py      #   Equity curves + performance stats
│   │
│   └── alerts/               # Telegram (339 lines)
│       ├── telegram.py       #   Alerter (HTML-formatted messages)
│       └── handler.py        #   Bot commands (/positions, /ask, etc.)
│
├── scripts/
│   ├── cli.py                # CLI entry point (run / status)
│   ├── diagnose_features.py  # Feature diagnostic plots
│   ├── diagnose_strategies.py # Strategy diagnostic plots
│   └── diagnose_montecarlo.py # Monte Carlo diagnostic plots
│
├── tests/                    # 217 tests (13 test files)
├── setup_wizard.py           # Interactive guided setup
├── setup.ps1                 # PowerShell setup script
├── setup.sh                  # Bash setup script
├── pyproject.toml            # Project config + dependencies
├── .env.example              # Configuration template
└── README.md                 # This file
```

**Total: 59 Python files | 11,400+ lines of code | 217 tests**

---

## 🧠 The 10 Strategies

| # | Strategy | Description | Signal Type |
|---|----------|-------------|-------------|
| 1 | **Trend** | EMA crossover + ADX filter + volume confirmation | Continuous [-1, +1] |
| 2 | **Mean Reversion** | Bollinger band + RSI extremes + volume spike | Continuous [-1, +1] |
| 3 | **Squeeze** | Keltner-inside-Bollinger breakout + momentum | Continuous [-1, +1] |
| 4 | **Smart Money** | Price-volume divergence (institutional flow proxy) | Continuous [-1, +1] |
| 5 | **Basis Arb** | Futures-spot basis with mean reversion bands | Continuous [-1, +1] |
| 6 | **Breakout** | ATR channel breakout + volume confirmation | Continuous [-1, +1] |
| 7 | **Funding Momentum** | Funding rate momentum + crowd positioning | Continuous [-1, +1] |
| 8 | **OI Divergence** | Price vs open interest divergence detection | Continuous [-1, +1] |
| 9 | **Liquidation Cascade** | Rapid OI drop + price displacement detection | Continuous [-1, +1] |
| 10 | **Volume Profile** | Volume-weighted mean reversion (VWAP-like) | Continuous [-1, +1] |

All strategies produce continuous signals on a [-1, +1] scale. The **ensemble** combines them with regime-dependent weights.

---

## 🔮 AI Decision Making

The AI receives a **massive context** for each scan cycle:

```
System Prompt (~2,000 tokens)
  ├── Role definition + expertise areas
  ├── Scorecard interpretation (IC, hit rate, confidence)
  ├── On-chain signal rules (L/S ratios, OI, TVL)
  ├── Correlation rules (cluster avoidance)
  ├── Macro event rules (FOMC risk reduction)
  ├── Self-performance feedback
  └── Output format specification

Market Prompt (~8,000 tokens per scan)
  ├── Sentiment (Fear & Greed, BTC dominance, funding)
  ├── Per-pair data (20 pairs × regime + strategies + probabilities)
  ├── On-chain intelligence (L/S ratio, taker flows, OI changes)
  ├── Cross-pair correlation (clusters, hedges, concentration)
  ├── Macro events (upcoming FOMC, CPI, NFP)
  ├── Past performance stats (hit rate, accuracy by confidence)
  ├── Active self-notes from memory
  └── Open positions (for management)
```

The AI outputs structured decisions:
```
DECISION: LONG ETHUSDT
CONFIDENCE: MEDIUM
REASONING: Strong trend signal (0.72) confirmed by rising OI (+5.2%) and
  bullish taker ratio (1.15). However, L/S ratio at 1.4 suggests moderate
  long crowding — sizing conservatively.
SL: 3450.00 | TP: 3680.00
ALERT: NONE
SELF_NOTES: ETH showing trend-OI alignment. Watch for L/S > 1.5.
```

---

## 📱 Telegram Bot Commands

| Command | Description |
|---------|-------------|
| `/positions` | View all open paper trades with PnL |
| `/portfolio` | Portfolio stats (equity, win rate, Sharpe) |
| `/memory` | AI memory (active notes, predictions) |
| `/status` | Agent status (cycle count, regime, budget) |
| `/prompts` | Prompt log stats (interactions, cost, tokens) |
| `/scan` | Force an immediate scan cycle |
| `/ask [question]` | Chat with the AI about any pair or market |

**Free-form text** is automatically treated as an `/ask` question.

---

## ⚙️ Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | *required* | Google AI API key ([get free](https://aistudio.google.com/app/apikey)) |
| `OPENAI_API_KEY` | | Optional OpenAI fallback |
| `ANTHROPIC_API_KEY` | | Optional Claude fallback |
| `APOLLO_AI_PROVIDER` | `auto` | `google`, `openai`, `anthropic`, or `auto` |
| `APOLLO_DAILY_BUDGET` | `50.0` | Max daily AI spend in USD |
| `APOLLO_WEEKLY_BUDGET` | `200.0` | Max weekly AI spend in USD |
| `APOLLO_DEFAULT_TIER` | `2` | Default model tier (1=cheap, 2=balanced, 3=best) |
| `APOLLO_MAX_PAIRS` | `15` | Max pairs to scan per cycle |
| `APOLLO_SCAN_INTERVAL_HOURS` | `2` | Base scan interval (adaptive overrides this) |
| `APOLLO_ADAPTIVE_SCAN` | `true` | Auto-adjust interval based on regime/positions |
| `TELEGRAM_BOT_TOKEN` | | Telegram bot token from @BotFather |
| `TELEGRAM_CHAT_ID` | | Your Telegram chat ID (auto-discovered if empty) |

---

## 📊 Data Sources (All FREE)

| Source | Data | API Key? |
|--------|------|----------|
| **Binance Futures** | OHLCV, Mark Price, Funding, OI, L/S Ratios | No |
| **Binance Spot** | Spot prices (for basis calculation) | No |
| **CoinGecko** | Market cap, supply, community, ATH, exchanges | No |
| **DeFiLlama** | Chain TVL, protocol TVL, stablecoin supply | No |
| **Alternative.me** | Fear & Greed Index | No |
| **Binance Derivatives** | Top trader L/S, taker buy/sell, global L/S | No |

> 💰 **Zero data costs.** All market data comes from public, no-key-required APIs.

---

## 🔄 Agent Cycle (What Happens Each Loop)

```
1. EVALUATE      Evaluate past predictions + decision quality
2. CLEANUP       Remove expired self-notes
3. RETRAIN       Auto-retrain models if > 24h since last retrain
4. SL/TP CHECK   Check stop loss / take profit on open positions
5. TELEGRAM      Poll for user commands
6. SCAN          Discover top 20 pairs → full pipeline analysis
                 (features → HMM → strategies → MC → XGBoost → risk)
7. ON-CHAIN      Fetch L/S ratios, OI, TVL from free APIs
8. CORRELATION   Compute cross-pair correlation matrix
9. EVENTS        Check macro event calendar
10. AI ANALYZE   Send everything to Gemini → get decisions
11. QUALITY      Record decisions for future accuracy evaluation
12. EXECUTE      Open/close paper trades based on decisions
13. ALERT        Send Telegram notifications
14. SNAPSHOT     Save portfolio snapshot
```

**Adaptive interval**: 30min (high volatility) → 45min (trending) → 1h (normal) → 2h (quiet)

---

## 🧪 Testing

```bash
# Run all tests (217 tests, ~65s)
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_phase7.py -v

# Run with coverage
python -m pytest tests/ --cov=apollo --cov-report=html
```

### Test Coverage

| Module | Tests | What's Tested |
|--------|-------|---------------|
| `test_features.py` | 30+ | All 40+ indicators, edge cases, NaN handling |
| `test_strategies.py` | 20+ | Signal bounds, regime interaction, edge cases |
| `test_ensemble.py` | 10+ | Weighting, combination, regime fallback |
| `test_monte_carlo.py` | 10+ | EGARCH fitting, scenario generation, VaR |
| `test_regime.py` | 8+ | HMM fitting, prediction, label mapping |
| `test_config.py` | 13 | Settings, auto-detection, validation |
| `test_integration.py` | 30+ | Full pipeline end-to-end |
| `test_phase3.py` | 40+ | Memory, budget, sentiment, parser, providers |
| `test_phase7.py` | 19 | Correlation, events, quality, retrain |
| `test_types.py` | 15+ | Data structures, serialization |
| `test_risk.py` | 8+ | VaR, Kelly sizing, SL/TP |

---

## 📈 Diagnostic Scripts

Generate visual reports for strategy analysis:

```bash
# Feature correlation analysis
python scripts/diagnose_features.py

# Strategy signal quality
python scripts/diagnose_strategies.py

# Monte Carlo simulation diagnostics
python scripts/diagnose_montecarlo.py
```

These generate Matplotlib/Plotly charts to `outputs/`.

---

## 🔐 Security Notes

- **No exchange API keys required** for paper trading — all data is public.
- **AI API key** is stored in `.env` (gitignored) and never logged.
- **Telegram bot token** is never committed — stored in `.env`.
- **No outbound connections** other than: Binance, CoinGecko, DeFiLlama, Google AI, Telegram.
- **All trade execution is virtual** (paper trading) — no real money at risk.

---

## 🗺️ Roadmap

- [x] ~~Core pipeline (features, strategies, HMM, Monte Carlo, XGBoost)~~
- [x] ~~AI brain with budget management~~
- [x] ~~Memory system (predictions, self-notes, position journal)~~
- [x] ~~On-chain intelligence (CoinGecko, DeFiLlama, Binance derivs)~~
- [x] ~~Paper trading with SL/TP~~
- [x] ~~Telegram alerts + bot commands~~
- [x] ~~Cross-pair correlation + cluster detection~~
- [x] ~~Macro events calendar (FOMC, CPI, NFP)~~
- [x] ~~Decision quality tracking (self-learning)~~
- [x] ~~Auto model retraining (rolling window)~~
- [x] ~~Prompt logging for analysis~~
- [ ] Live trading via ccxt (Binance API keys)
- [ ] Analytics dashboard (Plotly web)
- [ ] News/Twitter sentiment integration
- [ ] Multi-exchange support (Bybit, OKX)

---

## 📋 FAQ

<details>
<summary><strong>How much does it cost to run?</strong></summary>

**Data: $0.** All market data comes from free APIs.

**AI: ~$0.02–0.05 per scan cycle** using Gemini 2.5 Pro. With the default 2h interval, that's roughly **$0.25–0.60/day**. The budget system prevents overages automatically.

</details>

<details>
<summary><strong>Can I use OpenAI instead of Google?</strong></summary>

Yes. Set `OPENAI_API_KEY=sk-...` in your `.env` and `APOLLO_AI_PROVIDER=openai`. OpenAI costs ~3x more per scan than Gemini.

</details>

<details>
<summary><strong>Is this trading real money?</strong></summary>

**No.** Apollo runs in paper trading mode by default. All positions are virtual. No exchange API keys are needed for paper trading.

</details>

<details>
<summary><strong>How many pairs does it scan?</strong></summary>

By default, it auto-discovers the **top 20 most liquid Binance Futures pairs** each cycle. This includes majors (BTC, ETH, SOL) and trending alts.

</details>

<details>
<summary><strong>What happens if the Binance API goes down?</strong></summary>

The system has exponential backoff and graceful degradation:
- Failed spot data → falls back to futures-only
- Failed on-chain → continues without on-chain context
- Failed AI → skips the cycle, retries next interval
- Consecutive failures increase the retry interval

</details>

<details>
<summary><strong>How do I stop the agent?</strong></summary>

Three ways:
1. `Ctrl+C` in the terminal
2. Create a `.stop_agent` file in the project root
3. Send `/stop` via Telegram (if configured)

</details>

---

## 📝 License

MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with 🧠 by a quant who believes AI should augment, not replace, rigorous analysis.**

*Apollo doesn't predict the future. It measures probabilities, manages risk, and learns from its mistakes.*

</div>
