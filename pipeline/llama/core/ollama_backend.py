from __future__ import annotations

import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any, Dict

SYSTEM = """You are a senior VFX matchmove supervisor.
Convert a client note + structured shot analysis into tracking guidance for mask-based tracking.

You MUST return ONLY valid JSON with:
{
  "mask_includes": [string, ...],
  "mask_excludes": [string, ...],
  "track_mode": "track_inside_mask" | "track_outside_mask" | "no_mask_needed",
  "tracking_targets": ["camera"|"object"|"rotomation"|"matchmove"|"other", ...],
  "confidence": number between 0 and 1,
  "notes": "short reasoning, max 1-2 sentences"
}

Rules:
- Keep includes/excludes SAM3-friendly: concrete visual noun phrases.
- Prefer the suggested plan unless clearly wrong.
"""

@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 512

class OllamaReasoner:
    def __init__(self, cfg: OllamaConfig) -> None:
        self.cfg = cfg

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self.cfg.base_url.rstrip("/") + path
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", errors="replace") if e.fp else str(e)
            raise RuntimeError(f"Ollama HTTP {e.code}: {msg}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to reach Ollama at {url}. Is Ollama running?") from e
        return json.loads(raw)

    @staticmethod
    def _safe_json_parse(text: str) -> Dict[str, Any]:
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            return json.loads(text)
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise ValueError("Model did not return JSON.")

    def reason_one(self, user_payload_text: str) -> Dict[str, Any]:
        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_payload_text},
            ],
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature,
                "top_p": self.cfg.top_p,
                "num_predict": self.cfg.max_tokens,
            },
        }
        resp = self._post_json("/api/chat", payload)
        content = (resp.get("message") or {}).get("content", "") or ""
        data = self._safe_json_parse(content)
        data.setdefault("mask_includes", [])
        data.setdefault("mask_excludes", [])
        data.setdefault("track_mode", "no_mask_needed")
        data.setdefault("tracking_targets", ["camera"])
        data.setdefault("confidence", 0.5)
        data.setdefault("notes", "")
        return data
