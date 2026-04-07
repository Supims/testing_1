"""
On-Chain Intelligence Module
==============================
Aggregates FREE on-chain + market intelligence from multiple providers.
No API keys required for any of these endpoints.

Data sources:
  1. DeFiLlama   — TVL, protocol flows, stablecoin supply, yields
  2. CoinGecko   — Market data, community sentiment, market cap ranks
  3. Alternative  — Fear & Greed Index (already in sentiment.py)
  4. Binance      — Funding rates, OI, long/short ratios (public endpoints)
  5. Blockchain   — Mempool stats, gas prices, network hash rate

Output: structured dict injected into AI prompts for each symbol.

Tables produced (matching user spec):
  TABLE 1 — Market Overview (price, volume, volatility, trend)
  TABLE 2 — Exchange & Derivatives Flows (funding, OI, L/S ratios)
  TABLE 3 — DeFi & TVL context (protocol TVL, stablecoin flows)
  TABLE 4 — Pair Comparison (vs BTC strength)
  TABLE 5 — Community Sentiment (social scores, dev activity)
  TABLE 6 — Network Health (if applicable)
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("data.onchain")

_TIMEOUT = 12
_HEADERS = {"User-Agent": "apollo-quant/2.0", "Accept": "application/json"}

# Symbol -> CoinGecko ID mapping (top pairs)
_SYMBOL_TO_GECKO = {
    "BTCUSDT": "bitcoin", "ETHUSDT": "ethereum", "SOLUSDT": "solana",
    "BNBUSDT": "binancecoin", "XRPUSDT": "ripple", "DOGEUSDT": "dogecoin",
    "ADAUSDT": "cardano", "AVAXUSDT": "avalanche-2", "DOTUSDT": "polkadot",
    "LINKUSDT": "chainlink", "LTCUSDT": "litecoin", "ARBUSDT": "arbitrum",
    "OPUSDT": "optimism", "APTUSDT": "aptos", "NEARUSDT": "near",
    "FILUSDT": "filecoin", "ATOMUSDT": "cosmos", "TRXUSDT": "tron",
    "UNIUSDT": "uniswap", "SUIUSDT": "sui", "SEIUSDT": "sei-network",
    "INJUSDT": "injective-protocol", "PEPEUSDT": "pepe",
    "WIFUSDT": "dogwifcoin", "FETUSDT": "fetch-ai",
    "RENDERUSDT": "render-token", "TIAUSDT": "celestia",
    "JUPUSDT": "jupiter-exchange-solana", "MATICUSDT": "matic-network",
    "TONUSDT": "the-open-network", "AAVEUSDT": "aave",
}

# DeFiLlama slug for top protocols (where symbol is the governance token)
_SYMBOL_TO_DEFI = {
    "ETHUSDT": "ethereum", "SOLUSDT": "solana", "AVAXUSDT": "avalanche",
    "BNBUSDT": "bsc", "ARBUSDT": "arbitrum", "OPUSDT": "optimism",
    "SUIUSDT": "sui", "APTUSDT": "aptos", "NEARUSDT": "near",
    "ADAUSDT": "cardano", "DOTUSDT": "polkadot", "ATOMUSDT": "cosmos",
    "TONUSDT": "ton", "TRXUSDT": "tron", "LINKUSDT": None,
    "UNIUSDT": None, "AAVEUSDT": None,
}


def _fetch_json(url: str, timeout: int = _TIMEOUT) -> Optional[dict | list]:
    """Fetch JSON with timeout and error handling."""
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        logger.debug("Fetch failed: %s -> %s", url[:80], e)
        return None


# ============================================================================
# Data Fetchers (all free, no API keys)
# ============================================================================

class CoinGeckoFetcher:
    """
    CoinGecko Free API (rate limit: ~10-30 req/min, no key needed).
    Provides: market data, community metrics, market cap rank.
    """

    BASE = "https://api.coingecko.com/api/v3"

    def get_coin_data(self, gecko_id: str) -> Optional[dict]:
        """Full coin data including market, community, and sentiment."""
        url = (
            f"{self.BASE}/coins/{gecko_id}"
            f"?localization=false&tickers=false&market_data=true"
            f"&community_data=true&developer_data=false&sparkline=false"
        )
        data = _fetch_json(url)
        if not data:
            return None

        md = data.get("market_data", {})
        cd = data.get("community_data", {})
        sentiment = data.get("sentiment_votes_up_percentage", 0)

        return {
            "price_usd": md.get("current_price", {}).get("usd", 0),
            "market_cap": md.get("market_cap", {}).get("usd", 0),
            "market_cap_rank": md.get("market_cap_rank", 999),
            "total_volume_24h": md.get("total_volume", {}).get("usd", 0),
            "price_change_24h_pct": md.get("price_change_percentage_24h", 0),
            "price_change_7d_pct": md.get("price_change_percentage_7d", 0),
            "price_change_30d_pct": md.get("price_change_percentage_30d", 0),
            "ath": md.get("ath", {}).get("usd", 0),
            "ath_change_pct": md.get("ath_change_percentage", {}).get("usd", 0),
            "circulating_supply": md.get("circulating_supply", 0),
            "total_supply": md.get("total_supply", 0),
            "max_supply": md.get("max_supply"),
            # Community
            "twitter_followers": cd.get("twitter_followers", 0),
            "reddit_subscribers": cd.get("reddit_subscribers", 0),
            "reddit_active_48h": cd.get("reddit_accounts_active_48h", 0),
            "sentiment_up_pct": sentiment,
            "sentiment_down_pct": data.get("sentiment_votes_down_percentage", 0),
        }

    def get_global_market(self) -> Optional[dict]:
        """Global crypto market data."""
        data = _fetch_json(f"{self.BASE}/global")
        if not data or "data" not in data:
            return None
        d = data["data"]
        return {
            "total_market_cap_usd": d.get("total_market_cap", {}).get("usd", 0),
            "total_volume_24h_usd": d.get("total_volume", {}).get("usd", 0),
            "btc_dominance": d.get("market_cap_percentage", {}).get("btc", 0),
            "eth_dominance": d.get("market_cap_percentage", {}).get("eth", 0),
            "active_cryptos": d.get("active_cryptocurrencies", 0),
            "market_cap_change_24h": d.get("market_cap_change_percentage_24h_usd", 0),
        }


class DeFiLlamaFetcher:
    """
    DeFiLlama Free API (no key, no rate limit issues).
    Provides: chain TVL, protocol data, stablecoin supply.
    """

    def get_chain_tvl(self, chain: str) -> Optional[dict]:
        """Current TVL for a blockchain."""
        data = _fetch_json(f"https://api.llama.fi/v2/chains")
        if not data:
            return None
        for item in data:
            if item.get("name", "").lower() == chain.lower():
                return {
                    "tvl_usd": item.get("tvl", 0),
                    "chain": item.get("name", chain),
                }
        return None

    def get_stablecoin_supply(self) -> Optional[dict]:
        """Total stablecoin circulating supply."""
        data = _fetch_json("https://stablecoins.llama.fi/stablecoins?includePrices=true")
        if not data or "peggedAssets" not in data:
            return None

        total_supply = 0
        top_stables = []
        for stable in data["peggedAssets"][:10]:
            supply = stable.get("circulating", {}).get("peggedUSD", 0)
            total_supply += supply
            top_stables.append({
                "name": stable.get("name", ""),
                "symbol": stable.get("symbol", ""),
                "supply": supply,
            })

        return {
            "total_stablecoin_supply": total_supply,
            "top_stablecoins": top_stables[:5],
        }

    def get_yields_summary(self, chain: str = None) -> Optional[dict]:
        """Top yields across DeFi protocols."""
        url = "https://yields.llama.fi/pools"
        data = _fetch_json(url, timeout=15)
        if not data or "data" not in data:
            return None

        pools = data["data"]
        if chain:
            pools = [p for p in pools if p.get("chain", "").lower() == chain.lower()]

        # Sort by TVL, take top 5
        pools.sort(key=lambda p: p.get("tvlUsd", 0), reverse=True)
        top = pools[:5]

        return {
            "top_pools": [
                {
                    "protocol": p.get("project", ""),
                    "symbol": p.get("symbol", ""),
                    "chain": p.get("chain", ""),
                    "apy": round(p.get("apy", 0), 2),
                    "tvl_usd": p.get("tvlUsd", 0),
                }
                for p in top
            ]
        }


class BinanceDerivsFetcher:
    """
    Binance public futures endpoints (no key needed).
    Provides: detailed funding, OI, long/short ratios.
    """

    FAPI = "https://fapi.binance.com/futures/data"

    def get_long_short_ratio(self, symbol: str, period: str = "1h") -> Optional[dict]:
        """Top trader long/short ratio."""
        url = f"{self.FAPI}/topLongShortAccountRatio?symbol={symbol}&period={period}&limit=5"
        data = _fetch_json(url)
        if not data or len(data) == 0:
            return None
        latest = data[-1]
        return {
            "long_account_pct": float(latest.get("longAccount", 0.5)) * 100,
            "short_account_pct": float(latest.get("shortAccount", 0.5)) * 100,
            "long_short_ratio": float(latest.get("longShortRatio", 1.0)),
            "timestamp": latest.get("timestamp", 0),
        }

    def get_taker_buy_sell(self, symbol: str, period: str = "1h") -> Optional[dict]:
        """Taker buy/sell volume ratio."""
        url = f"{self.FAPI}/takerlongshortRatio?symbol={symbol}&period={period}&limit=5"
        data = _fetch_json(url)
        if not data or len(data) == 0:
            return None
        latest = data[-1]
        return {
            "buy_vol_pct": float(latest.get("buyVol", 0.5)) * 100,
            "sell_vol_pct": float(latest.get("sellVol", 0.5)) * 100,
            "buy_sell_ratio": float(latest.get("buySellRatio", 1.0)),
        }

    def get_open_interest_hist(self, symbol: str, period: str = "1h") -> Optional[dict]:
        """Open interest history."""
        url = f"https://fapi.binance.com/futures/data/openInterestHist?symbol={symbol}&period={period}&limit=10"
        data = _fetch_json(url)
        if not data or len(data) < 2:
            return None
        latest = data[-1]
        prev = data[-2]
        oi_now = float(latest.get("sumOpenInterestValue", 0))
        oi_prev = float(prev.get("sumOpenInterestValue", 0))
        change_pct = ((oi_now - oi_prev) / oi_prev * 100) if oi_prev > 0 else 0
        return {
            "open_interest_usd": oi_now,
            "oi_change_pct": round(change_pct, 2),
        }

    def get_global_long_short(self, symbol: str, period: str = "1h") -> Optional[dict]:
        """Global (all accounts) long/short ratio."""
        url = f"{self.FAPI}/globalLongShortAccountRatio?symbol={symbol}&period={period}&limit=5"
        data = _fetch_json(url)
        if not data or len(data) == 0:
            return None
        latest = data[-1]
        return {
            "global_long_pct": float(latest.get("longAccount", 0.5)) * 100,
            "global_short_pct": float(latest.get("shortAccount", 0.5)) * 100,
            "global_ls_ratio": float(latest.get("longShortRatio", 1.0)),
        }


# ============================================================================
# Main Aggregator
# ============================================================================

@dataclass
class OnChainProfile:
    """Complete on-chain profile for a symbol."""
    symbol: str
    timestamp: str = ""
    # Market
    market: dict = field(default_factory=dict)
    # Derivatives
    derivatives: dict = field(default_factory=dict)
    # DeFi / TVL
    defi: dict = field(default_factory=dict)
    # Community
    community: dict = field(default_factory=dict)
    # Global context
    global_market: dict = field(default_factory=dict)
    # Stablecoin context
    stablecoins: dict = field(default_factory=dict)

    def to_prompt_block(self) -> str:
        """Format as text block for AI prompt injection."""
        lines = [f"=== ON-CHAIN: {self.symbol} ==="]

        if self.market:
            m = self.market
            lines.append(f"Price: ${m.get('price_usd', 0):,.2f} | "
                        f"24h: {m.get('price_change_24h_pct', 0):+.1f}% | "
                        f"7d: {m.get('price_change_7d_pct', 0):+.1f}% | "
                        f"30d: {m.get('price_change_30d_pct', 0):+.1f}%")
            lines.append(f"MCap Rank: #{m.get('market_cap_rank', '?')} | "
                        f"Vol24h: ${m.get('total_volume_24h', 0):,.0f}")
            if m.get("circulating_supply") and m.get("total_supply"):
                circ_pct = m["circulating_supply"] / max(m["total_supply"], 1) * 100
                lines.append(f"Supply: {circ_pct:.0f}% circulating | "
                            f"ATH: ${m.get('ath', 0):,.2f} ({m.get('ath_change_pct', 0):+.0f}%)")

        if self.derivatives:
            d = self.derivatives
            if "long_short_ratio" in d:
                ls = d["long_short_ratio"]
                lines.append(f"Top Traders L/S: {ls:.2f} "
                            f"(L:{d.get('long_account_pct', 50):.0f}% / "
                            f"S:{d.get('short_account_pct', 50):.0f}%)")
            if "buy_sell_ratio" in d:
                lines.append(f"Taker Buy/Sell: {d['buy_sell_ratio']:.2f}")
            if "open_interest_usd" in d:
                lines.append(f"OI: ${d['open_interest_usd']:,.0f} "
                            f"({d.get('oi_change_pct', 0):+.1f}%)")
            if "global_ls_ratio" in d:
                lines.append(f"Global L/S: {d['global_ls_ratio']:.2f} "
                            f"(L:{d.get('global_long_pct', 50):.0f}%)")

        if self.defi:
            if "tvl_usd" in self.defi:
                lines.append(f"Chain TVL: ${self.defi['tvl_usd']:,.0f}")

        if self.community:
            c = self.community
            if c.get("twitter_followers"):
                lines.append(f"Twitter: {c['twitter_followers']:,} | "
                            f"Reddit: {c.get('reddit_subscribers', 0):,} | "
                            f"Sentiment: {c.get('sentiment_up_pct', 0):.0f}% bullish")

        if self.global_market:
            g = self.global_market
            lines.append(f"Global: BTC.D={g.get('btc_dominance', 0):.1f}% | "
                        f"MCap 24h: {g.get('market_cap_change_24h', 0):+.1f}%")

        if self.stablecoins:
            total = self.stablecoins.get("total_stablecoin_supply", 0)
            if total > 0:
                lines.append(f"Stablecoin Supply: ${total:,.0f}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """For JSON serialization."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "market": self.market,
            "derivatives": self.derivatives,
            "defi": self.defi,
            "community": self.community,
            "global_market": self.global_market,
            "stablecoins": self.stablecoins,
        }


class OnChainIntelligence:
    """
    Main aggregator. Fetches on-chain data for symbols in parallel.

    Usage:
        intel = OnChainIntelligence()
        profiles = intel.fetch_batch(["BTCUSDT", "ETHUSDT", "SOLUSDT"])
        for p in profiles:
            print(p.to_prompt_block())
    """

    def __init__(self, max_workers: int = 4):
        self._gecko = CoinGeckoFetcher()
        self._defi = DeFiLlamaFetcher()
        self._derivs = BinanceDerivsFetcher()
        self._max_workers = max_workers
        # Cache global data (same for all symbols)
        self._global_cache: Optional[dict] = None
        self._stable_cache: Optional[dict] = None
        self._cache_ts: float = 0

    def _refresh_global_cache(self):
        """Refresh global market + stablecoin data (max once per 5 min)."""
        if time.time() - self._cache_ts < 300 and self._global_cache:
            return
        self._global_cache = self._gecko.get_global_market()
        self._stable_cache = self._defi.get_stablecoin_supply()
        self._cache_ts = time.time()

    def fetch_single(self, symbol: str) -> OnChainProfile:
        """Fetch all on-chain data for one symbol."""
        profile = OnChainProfile(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # CoinGecko market data
        gecko_id = _SYMBOL_TO_GECKO.get(symbol)
        if gecko_id:
            coin_data = self._gecko.get_coin_data(gecko_id)
            if coin_data:
                profile.market = {
                    k: v for k, v in coin_data.items()
                    if k not in ("twitter_followers", "reddit_subscribers",
                                 "reddit_active_48h", "sentiment_up_pct",
                                 "sentiment_down_pct")
                }
                profile.community = {
                    "twitter_followers": coin_data.get("twitter_followers", 0),
                    "reddit_subscribers": coin_data.get("reddit_subscribers", 0),
                    "reddit_active_48h": coin_data.get("reddit_active_48h", 0),
                    "sentiment_up_pct": coin_data.get("sentiment_up_pct", 0),
                    "sentiment_down_pct": coin_data.get("sentiment_down_pct", 0),
                }
            # Rate limit protection
            time.sleep(0.5)

        # Binance derivatives data (always available for USDT-M pairs)
        ls = self._derivs.get_long_short_ratio(symbol)
        taker = self._derivs.get_taker_buy_sell(symbol)
        oi = self._derivs.get_open_interest_hist(symbol)
        global_ls = self._derivs.get_global_long_short(symbol)

        derivs = {}
        if ls:
            derivs.update(ls)
        if taker:
            derivs.update(taker)
        if oi:
            derivs.update(oi)
        if global_ls:
            derivs.update(global_ls)
        profile.derivatives = derivs

        # DeFi TVL (for chain tokens)
        chain = _SYMBOL_TO_DEFI.get(symbol)
        if chain:
            tvl = self._defi.get_chain_tvl(chain)
            if tvl:
                profile.defi = tvl

        # Global context (cached)
        if self._global_cache:
            profile.global_market = self._global_cache
        if self._stable_cache:
            profile.stablecoins = self._stable_cache

        return profile

    def fetch_batch(self, symbols: list[str], max_symbols: int = 10) -> list[OnChainProfile]:
        """
        Fetch on-chain data for multiple symbols in parallel.
        Limits to max_symbols to respect rate limits.
        """
        self._refresh_global_cache()

        # Limit to avoid CoinGecko rate limits
        symbols = symbols[:max_symbols]
        profiles = []

        # Use ThreadPool for parallel fetching (Binance calls are fast)
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(self.fetch_single, sym): sym
                for sym in symbols
            }
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    profile = future.result()
                    profiles.append(profile)
                    logger.debug("On-chain: %s fetched", sym)
                except Exception as e:
                    logger.warning("On-chain fetch failed for %s: %s", sym, e)
                    profiles.append(OnChainProfile(symbol=sym))

        # Sort by original order
        sym_order = {s: i for i, s in enumerate(symbols)}
        profiles.sort(key=lambda p: sym_order.get(p.symbol, 999))

        logger.info(
            "On-chain: fetched %d/%d profiles (global: %s, stables: %s)",
            len(profiles), len(symbols),
            "yes" if self._global_cache else "no",
            "yes" if self._stable_cache else "no",
        )
        return profiles

    def format_multi_prompt(self, profiles: list[OnChainProfile]) -> str:
        """Format multiple profiles into a single prompt block."""
        blocks = [p.to_prompt_block() for p in profiles if p.market or p.derivatives]
        return "\n\n".join(blocks) if blocks else ""
