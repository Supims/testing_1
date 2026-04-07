"""
Prompt Logger
==============
Logs all prompts sent to and received from the AI for analysis.

Stores:
  - Timestamp
  - Model used
  - System prompt
  - User prompt
  - AI response
  - Token counts and cost
  - Scan ID

Files stored in: logs/prompts/YYYY-MM-DD/
Format: JSON lines (.jsonl) for easy analysis
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("ai.prompt_log")


class PromptLogger:
    """
    Logs every AI interaction to structured JSONL files.

    Usage:
        plog = PromptLogger()
        plog.log_interaction(
            scan_id="abc123",
            model="gemini-2.5-pro",
            system_prompt=system,
            user_prompt=user,
            response=text,
            in_tokens=4881,
            out_tokens=347,
            cost=0.0096,
        )
    """

    def __init__(self, base_dir: Path = None, max_days: int = 30):
        if base_dir is None:
            from apollo.config import settings
            base_dir = settings.logs_dir / "prompts"
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._max_days = max_days
        self._cleanup_old_logs()

    def _cleanup_old_logs(self):
        """Remove log directories older than max_days."""
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._max_days)
        try:
            for day_dir in self._base_dir.iterdir():
                if day_dir.is_dir():
                    try:
                        dir_date = datetime.strptime(day_dir.name, "%Y-%m-%d").replace(
                            tzinfo=timezone.utc
                        )
                        if dir_date < cutoff:
                            import shutil
                            shutil.rmtree(day_dir)
                            logger.debug("Cleaned up old prompt log: %s", day_dir.name)
                    except ValueError:
                        pass  # Not a date directory
        except Exception as e:
            logger.debug("Log cleanup error: %s", e)

    def _get_log_file(self) -> Path:
        """Get today's log file path."""
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        day_dir = self._base_dir / date_str
        day_dir.mkdir(parents=True, exist_ok=True)
        return day_dir / "interactions.jsonl"

    def log_interaction(
        self,
        scan_id: str = "",
        model: str = "",
        system_prompt: str = "",
        user_prompt: str = "",
        response: str = "",
        in_tokens: int = 0,
        out_tokens: int = 0,
        cost: float = 0.0,
        task_type: str = "scan",
        decisions: list[dict] = None,
        metadata: dict = None,
    ):
        """Log a single AI interaction."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scan_id": scan_id,
            "task_type": task_type,
            "model": model,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response": response,
            "in_tokens": in_tokens,
            "out_tokens": out_tokens,
            "total_tokens": in_tokens + out_tokens,
            "cost_usd": cost,
            "prompt_chars": len(system_prompt) + len(user_prompt),
            "response_chars": len(response),
        }

        if decisions:
            entry["decisions"] = decisions
        if metadata:
            entry["metadata"] = metadata

        log_file = self._get_log_file()
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            logger.debug("Prompt logged to %s", log_file)
        except Exception as e:
            logger.warning("Failed to log prompt: %s", e)

    def get_recent_interactions(self, n: int = 10) -> list[dict]:
        """Read the N most recent interactions (today + yesterday)."""
        entries = []
        # Check today and yesterday
        for offset in range(2):
            date = datetime.now(timezone.utc)
            if offset:
                from datetime import timedelta
                date = date - timedelta(days=1)
            date_str = date.strftime("%Y-%m-%d")
            log_file = self._base_dir / date_str / "interactions.jsonl"
            if log_file.exists():
                try:
                    with open(log_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                entries.append(json.loads(line))
                except Exception:
                    pass

        # Sort by timestamp descending, return last N
        entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
        return entries[:n]

    def get_daily_stats(self, date_str: str = None) -> dict:
        """Get stats for a specific day."""
        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        log_file = self._base_dir / date_str / "interactions.jsonl"
        if not log_file.exists():
            return {"date": date_str, "interactions": 0}

        entries = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

        total_cost = sum(e.get("cost_usd", 0) for e in entries)
        total_tokens = sum(e.get("total_tokens", 0) for e in entries)
        models_used = {}
        for e in entries:
            model = e.get("model", "unknown")
            models_used[model] = models_used.get(model, 0) + 1

        return {
            "date": date_str,
            "interactions": len(entries),
            "total_cost_usd": round(total_cost, 6),
            "total_tokens": total_tokens,
            "avg_prompt_chars": int(sum(e.get("prompt_chars", 0) for e in entries) / max(len(entries), 1)),
            "avg_response_chars": int(sum(e.get("response_chars", 0) for e in entries) / max(len(entries), 1)),
            "models_used": models_used,
        }

    def format_stats(self, date_str: str = None) -> str:
        """Human-readable stats."""
        stats = self.get_daily_stats(date_str)
        return (
            f"Prompt Log [{stats['date']}]: "
            f"{stats['interactions']} interactions | "
            f"${stats['total_cost_usd']:.4f} | "
            f"{stats['total_tokens']:,} tokens | "
            f"Models: {stats.get('models_used', {})}"
        )
