# Apollo Quant -- User Guide

Production setup and usage instructions.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Commands](#commands)
5. [Running in Production](#running-in-production)
6. [Understanding the Output](#understanding-the-output)
7. [Telegram Bot Setup](#telegram-bot-setup)
8. [Troubleshooting](#troubleshooting)
9. [Cost Estimates](#cost-estimates)

---

## Requirements

- **Python 3.10, 3.11, or 3.12** (3.12 recommended)
- **Internet connection** (Binance API + AI provider)
- **One AI API key** (Google Gemini recommended -- free tier available)
- **~500 MB disk space** (dependencies + cached data)

---

## Installation

### Step 1: Create a Python virtual environment

```bash
# Windows
py -3.12 -m venv .venv
.venv\Scripts\activate

# Linux / Mac
python3.12 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install dependencies

```bash
pip install -e ".[dev]"
```

This installs all required packages (numpy, pandas, scikit-learn, xgboost,
hmmlearn, openai, pydantic, ta, scipy, etc.).

### Step 3: Configure your API key

```bash
# Copy the template
cp .env.example .env
```

Edit `.env` and set your API key:

```ini
# Get a free key at: https://aistudio.google.com/app/apikey
GOOGLE_API_KEY=your-key-here
```

That is the only required setting. Everything else has sensible defaults.

### Step 4: Verify the setup

```bash
python scripts/cli.py check
```

This validates Python version, dependencies, configuration, Binance
connectivity, and AI provider access. Fix any issues it reports before
proceeding.

---

## Configuration

All settings are in `.env`. The most important ones:

| Setting                      | Default  | Description                          |
|------------------------------|----------|--------------------------------------|
| `GOOGLE_API_KEY`             | (none)   | Google Gemini API key (required)     |
| `APOLLO_AI_PROVIDER`        | `auto`   | `google`, `openai`, `anthropic`      |
| `APOLLO_DEFAULT_TIER`       | `2`      | Model quality: 1=fast, 2=balanced, 3=best |
| `APOLLO_MAX_PAIRS`          | `15`     | Max pairs to scan per cycle          |
| `APOLLO_DAILY_BUDGET`       | `50.0`   | Daily spend cap in USD               |
| `APOLLO_SCAN_INTERVAL_HOURS`| `2`      | Base hours between scans             |
| `APOLLO_ADAPTIVE_SCAN`      | `true`   | Auto-adjust interval by volatility   |
| `TELEGRAM_BOT_TOKEN`        | (none)   | Telegram bot token (optional)        |
| `TELEGRAM_CHAT_ID`          | (none)   | Your Telegram chat ID (optional)     |

### Tier System

| Tier | Google Model     | Speed    | Cost/scan | Best For                |
|------|------------------|----------|-----------|-------------------------|
| 1    | gemini-2.5-flash | ~5s      | ~$0.001   | Budget mode, monitoring  |
| 2    | gemini-2.5-pro   | ~15s     | ~$0.005   | Daily trading (default)  |
| 3    | gemini-2.5-pro   | ~15s     | ~$0.005   | Same as T2 for Google    |

---

## Commands

### `check` -- Validate your setup

```bash
python scripts/cli.py check
```

Checks: Python version, all dependencies, config file, Binance API
connectivity, and AI provider. Run this first.

### `scan` -- One-shot market scan

```bash
# Scan with default settings
python scripts/cli.py scan

# Scan top 3 pairs only (faster, cheaper)
python scripts/cli.py scan --pairs 3

# Verbose mode (shows all API calls)
python scripts/cli.py -v scan
```

This runs the full pipeline once:
1. Discovers top pairs by 24h volume
2. Fetches OHLCV + funding + OI data
3. Computes 40+ technical features
4. Detects market regime (HMM)
5. Runs 10 trading strategies
6. Monte Carlo simulation (500 paths)
7. XGBoost probability models
8. On-chain intelligence (L/S ratios, OI, TVL)
9. AI analysis (Gemini/GPT/Claude)
10. Outputs LONG/SHORT/SKIP decisions

### `run` -- Start the autonomous agent

```bash
# Continuous mode (runs until Ctrl+C)
python scripts/cli.py run

# Single cycle only
python scripts/cli.py run --once
```

The agent loop:
- Scans every 30 min to 2 hours (adaptive based on volatility)
- Manages virtual paper trading positions
- Tracks its own accuracy over time
- Sends Telegram alerts (if configured)
- Auto-retrains models when they go stale

### `status` -- Show system status

```bash
python scripts/cli.py status
```

Shows: config, budget usage, memory stats, open positions, decision quality.

---

## Running in Production

### Option 1: Background process (simple)

```bash
# Windows PowerShell
Start-Process -NoNewWindow python -ArgumentList "scripts/cli.py","run"

# Linux
nohup python scripts/cli.py run > logs/agent.log 2>&1 &
```

### Option 2: systemd service (Linux)

Create `/etc/systemd/system/apollo.service`:

```ini
[Unit]
Description=Apollo Quant Trading Agent
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/apollo-quant
ExecStart=/path/to/apollo-quant/.venv/bin/python scripts/cli.py run
Restart=on-failure
RestartSec=60

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable apollo
sudo systemctl start apollo
sudo journalctl -u apollo -f  # View logs
```

### Option 3: Screen/tmux (Linux/Mac)

```bash
screen -S apollo
python scripts/cli.py run
# Detach: Ctrl+A, D
# Reattach: screen -r apollo
```

### Graceful shutdown

- Press `Ctrl+C` (agent finishes current cycle, then stops)
- Or create a file called `.stop_agent` in the project root

---

## Understanding the Output

### Scan Results Table

```
  Pair           Regime                         Ensemble    Payoff   OOD
  BTCUSDT        Low Volatility (Quiet Range)    +0.0143  0.62:1    no
  ETHUSDT        Low Volatility (Quiet Range)    +0.0703  0.53:1   YES
```

- **Regime**: Current market state detected by HMM
- **Ensemble**: Combined signal from 10 strategies (-1 to +1)
- **Payoff**: Risk/reward ratio from Monte Carlo (want > 1:1)
- **OOD**: Out of Distribution -- model is uncertain, signals unreliable

### AI Decisions

```
DECISION: SKIP BTCUSDT
CONFIDENCE: HIGH
REASONING: Trend signal (+0.716) fires in Quiet Range regime (unreliable).
           Crowded longs (L/S: 1.65), payoff 0.62:1.
```

Decision types:
- **LONG**: Open a long position
- **SHORT**: Open a short position
- **SKIP**: Do not trade (most common and safest)
- **CLOSE**: Close an existing position

Confidence levels:
- **HIGH**: Strong conviction, multiple signals agree
- **MEDIUM**: Moderate conviction, some conflicting signals
- **LOW**: Weak conviction, mostly uncertainty

### Key Metrics to Watch

| Metric           | Good           | Caution            | Danger          |
|------------------|----------------|--------------------|-----------------|
| Payoff Ratio     | > 1.5:1        | 1.0-1.5:1          | < 1.0:1         |
| P(profit)        | > 55%          | 50-55%             | < 50%           |
| VaR 5%           | > -2%          | -2% to -4%         | < -4%           |
| L/S Ratio        | 0.8-1.2        | 1.2-1.5 or 0.7-0.8 | > 1.5 or < 0.7 |
| Cross Agreement  | > 0.7          | 0.5-0.7            | < 0.5           |

---

## Telegram Bot Setup

### Step 1: Create the bot

1. Open Telegram and search for `@BotFather`
2. Send `/newbot` and follow the prompts
3. Copy the token (looks like `123456789:ABCdef...`)

### Step 2: Get your chat ID

1. Send any message to your new bot
2. Apollo will auto-discover your chat ID on startup
3. Or find it manually: `https://api.telegram.org/bot<TOKEN>/getUpdates`

### Step 3: Configure

Add to `.env`:

```ini
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrSTUvwxYZ
```

### Bot Commands

| Command      | Description                     |
|--------------|---------------------------------|
| `/status`    | Agent status + next scan time   |
| `/positions` | Open paper trading positions    |
| `/portfolio` | Equity curve + overall PnL      |
| `/memory`    | Active AI self-notes            |
| `/prompts`   | Recent prompt stats + costs     |
| `/ask <q>`   | Ask the AI anything about markets |
| `/scan`      | Force an immediate scan cycle   |

---

## Troubleshooting

### "No AI provider configured"

You need at least one API key in `.env`. Get a free Google key at:
https://aistudio.google.com/app/apikey

### "HMM fitting failed" or covariance warnings

Normal on first run with limited data. The system uses 10 random restarts
and picks the best model. Warnings are suppressed in production.

### "Binance API 429" (rate limited)

The client has built-in retry logic with exponential backoff. If you see
frequent rate limits:
- Reduce `APOLLO_MAX_PAIRS` to 5
- Increase `APOLLO_SCAN_INTERVAL_HOURS` to 4

### "OOD" on many pairs

Out of Distribution means the current market behavior does not match what
the HMM was trained on. The system will:
1. Warn the AI in the prompt
2. The AI should reduce confidence and prefer SKIP
3. Models auto-retrain if OOD rates stay high

### Slow first run

The first run trains HMM + Monte Carlo + XGBoost models on 90 days of data.
This takes 2-5 minutes. Subsequent runs load models from disk (~1 second).

### Python version issues

Requires Python 3.10-3.12. Python 3.13+ may have compatibility issues with
hmmlearn. Check with:

```bash
python --version
```

---

## Cost Estimates

All market data (Binance, CoinGecko, DeFiLlama) is **free**. The only cost
is the AI provider.

### Google Gemini (recommended)

| Usage Pattern         | Scans/day | Daily Cost | Monthly Cost |
|-----------------------|-----------|------------|--------------|
| Conservative (T1)     | 12        | ~$0.01     | ~$0.30       |
| Standard (T2)         | 12        | ~$0.06     | ~$1.80       |
| Aggressive (T2, 15 pairs) | 24   | ~$0.25     | ~$7.50       |

### OpenAI GPT-4o

| Usage Pattern         | Scans/day | Daily Cost | Monthly Cost |
|-----------------------|-----------|------------|--------------|
| Standard (T2)         | 12        | ~$0.20     | ~$6.00       |

The budget system automatically prevents overspending. If the daily limit is
reached, Apollo downgrades to cheaper models or pauses AI calls.

---

## Data Flow Summary

```
Binance API ──> Raw OHLCV + Funding + OI
                     |
              Feature Pipeline (40+ indicators)
                     |
              HMM Regime Detection (4 states)
                     |
              10 Strategies (signals -1 to +1)
                     |
              Ensemble (regime-weighted combination)
                     |
              Monte Carlo (500 EGARCH paths)
                     |
              XGBoost Probabilities (5 multi-horizon models)
                     |
              Risk Dashboard (VaR, SL/TP, Kelly sizing)
                     |
    On-chain ──> AI Brain (Gemini/GPT/Claude) <── Memory
                     |
              Decisions: LONG / SHORT / SKIP / CLOSE
                     |
              Paper Trader + Telegram Alerts
```

---

## Files and Logs

| Path                  | Description                           |
|-----------------------|---------------------------------------|
| `.env`                | Your configuration                    |
| `models/`             | Trained model files (auto-created)    |
| `data/`               | Cached market data (parquet)          |
| `logs/prompts/`       | Full AI prompt/response logs (JSONL)  |
| `models/ai_memory.db` | AI memory database (SQLite)           |
| `models/paper_trades.db` | Paper trading database (SQLite)    |
| `outputs/`            | Analysis reports and exports          |

Prompt logs are stored as JSONL files at:
`logs/prompts/YYYY-MM-DD/interactions.jsonl`

Each line contains: timestamp, model, system prompt, user prompt, response,
token counts, and cost. Useful for debugging and analyzing AI behavior.
