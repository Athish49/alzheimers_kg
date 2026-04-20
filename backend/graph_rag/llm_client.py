"""
graph_rag.llm_client
--------------------

Multi-provider LLM client for the Graph-RAG layer.

Supported providers (set LLM_PROVIDER in backend/.env):
    ollama    — local Ollama daemon (default for local environment)
    openai    — OpenAI API
    groq      — Groq API (OpenAI-compatible; free tier available)
    gemini    — Google Gemini (OpenAI-compatible endpoint)
    anthropic — Anthropic Claude (requires `pip install anthropic`)

The provider is selected automatically by config.py based on the environment
and available API keys. You can override it explicitly via LLM_PROVIDER.

All providers expose the same interface:
    client.simple_qa(question, context, ...) -> str
    client.chat(messages, ...) -> str

Rate-limit retry strategy (cloud providers)
-------------------------------------------
1. First 429 → wait 3 seconds, retry once.
2. Second 429 → try the fallback provider (if configured) once.
3. Fallback also fails → raise LLMUnavailableError (caught by main.py → HTTP 503).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from .config import CONFIG

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RETRY_DELAY_S = 3  # seconds to wait before the single retry on 429

# Maximum context characters passed to the LLM (~30K tokens for gemini-2.5-flash's 1M limit).
# This is a cost/latency guardrail; real graph contexts are usually much smaller.
_MAX_CONTEXT_CHARS = 120_000

# OpenAI-SDK-compatible base URLs per provider
_OPENAI_COMPAT_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "groq":   "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
}

# Default system prompt shared across providers
_DEFAULT_SYSTEM_PROMPT = (
    "You are a **grounded extraction assistant**.\n"
    "\n"
    "CRITICAL RULES:\n"
    "- You MUST use ONLY facts that appear explicitly in the provided context.\n"
    "- Do NOT introduce any biomarker, drug, gene, or concept that is not "
    "literally present in the context text.\n"
    "- Do NOT rely on your own medical knowledge or outside information.\n"
    "- If the question asks for information that is not clearly present in "
    "the context, say: \"The context is insufficient to answer this precisely.\"\n"
    "- When listing entities, copy their names EXACTLY as written in the context.\n"
    "\n"
    "Your job is to extract and summarize, not to guess or generalize."
)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class LLMUnavailableError(RuntimeError):
    """Raised when the LLM and its fallback are both unavailable (e.g. rate-limited)."""


# ---------------------------------------------------------------------------
# LLMClient — single class, three underlying protocol paths
# ---------------------------------------------------------------------------

@dataclass
class LLMClient:
    """
    Unified LLM client supporting multiple providers.

    Parameters
    ----------
    provider:
        One of: ollama | openai | groq | gemini | anthropic
    model:
        Model identifier as expected by the provider.
    api_key:
        API key for cloud providers. Unused for ollama.
    fallback_provider / fallback_model / fallback_api_key:
        Secondary provider used when the primary is rate-limited.
    ollama_base_url:
        Base URL for the Ollama HTTP API (e.g. "http://localhost:11434/api").
    default_temperature / default_top_p / default_num_ctx / timeout:
        Sampling defaults; can be overridden per call.
    """

    provider:            str   = "ollama"
    model:               str   = "llama3.2:3b"
    api_key:             str   = ""
    fallback_provider:   str   = ""
    fallback_model:      str   = ""
    fallback_api_key:    str   = ""
    ollama_base_url:     str   = "http://localhost:11434/api"
    default_temperature: float = 0.2
    default_top_p:       float = 0.9
    default_num_ctx:     int   = 4096
    timeout:             int   = 120

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        num_ctx: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send a chat-style request and return the assistant reply as a string.

        Parameters
        ----------
        messages:
            List of {"role": "user" | "assistant" | "system", "content": "..."}.
        system_prompt:
            Prepended as a system message if provided.
        temperature, top_p, num_ctx, max_tokens:
            Per-call overrides.
        """
        temp = temperature if temperature is not None else self.default_temperature
        top  = top_p if top_p is not None else self.default_top_p
        ctx  = num_ctx if num_ctx is not None else self.default_num_ctx

        if self.provider == "ollama":
            return self._chat_ollama(messages, system_prompt, temp, top, ctx, max_tokens)
        if self.provider == "anthropic":
            return self._chat_anthropic(messages, system_prompt, temp, max_tokens)
        # openai / groq / gemini — all OpenAI-SDK-compatible
        return self._chat_openai_compat(messages, system_prompt, temp, top, max_tokens)

    def simple_qa(
        self,
        question: str,
        context: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = 512,
    ) -> str:
        """
        RAG-style helper: feed context + question to the LLM and return the answer.

        The context is truncated to _MAX_CONTEXT_CHARS before sending to keep
        token usage and cost within a reasonable bound.
        """
        if system_prompt is None:
            system_prompt = _DEFAULT_SYSTEM_PROMPT

        if len(context) > _MAX_CONTEXT_CHARS:
            logger.warning(
                "Context truncated from %d to %d chars before LLM call.",
                len(context), _MAX_CONTEXT_CHARS,
            )
            context = context[:_MAX_CONTEXT_CHARS]

        user_content = (
            "Context:\n"
            "---------------------\n"
            f"{context}\n"
            "---------------------\n\n"
            f"Question: {question}\n\n"
            "Answer using the context above."
        )
        messages = [{"role": "user", "content": user_content}]
        return self.chat(
            messages,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # ------------------------------------------------------------------
    # Path 1 — Ollama (requests-based)
    # ------------------------------------------------------------------

    def _chat_ollama(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str],
        temperature: float,
        top_p: float,
        num_ctx: int,
        max_tokens: Optional[int],
    ) -> str:
        final_messages: List[Dict[str, str]] = []
        if system_prompt:
            final_messages.append({"role": "system", "content": system_prompt})
        final_messages.extend(messages)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": final_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_ctx": num_ctx,
            },
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        url = f"{self.ollama_base_url}/chat"
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {url}. Is the daemon running?\n"
                f"Run: ollama serve   (then: ollama pull {self.model})\n"
                f"Original error: {exc}"
            ) from exc

        if resp.status_code != 200:
            raise RuntimeError(
                f"Ollama returned HTTP {resp.status_code}: {resp.text[:500]}"
            )

        data = resp.json()
        return (data.get("message") or {}).get("content", "")

    # ------------------------------------------------------------------
    # Path 2 — OpenAI-compatible (openai, groq, gemini)
    # ------------------------------------------------------------------

    def _chat_openai_compat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str],
        temperature: float,
        top_p: float,
        max_tokens: Optional[int],
    ) -> str:
        try:
            from openai import OpenAI, RateLimitError  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required for provider "
                f"'{self.provider}'. Run: pip install openai"
            ) from exc

        if not self.api_key:
            key_env = {
                "openai": "OPENAI_API_KEY",
                "groq":   "GROQ_API_KEY",
                "gemini": "GEMINI_API_KEY",
            }.get(self.provider, "LLM_API_KEY")
            raise RuntimeError(
                f"No API key found for provider '{self.provider}'. "
                f"Set {key_env} in backend/.env"
            )

        base_url = _OPENAI_COMPAT_BASE_URLS.get(self.provider)
        client = OpenAI(api_key=self.api_key, base_url=base_url)

        final_messages: List[Dict[str, str]] = []
        if system_prompt:
            final_messages.append({"role": "system", "content": system_prompt})
        final_messages.extend(messages)

        kwargs: Dict[str, Any] = {
            "model":       self.model,
            "messages":    final_messages,
            "temperature": temperature,
            "top_p":       top_p,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        # Attempt 1
        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except RateLimitError:
            logger.warning(
                "[%s] Rate limit (429) on first attempt — waiting %ds before retry.",
                self.provider, _RETRY_DELAY_S,
            )
            time.sleep(_RETRY_DELAY_S)
        except Exception as exc:
            raise RuntimeError(f"[{self.provider}] API call failed: {exc}") from exc

        # Attempt 2 (after 3s wait)
        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except RateLimitError:
            logger.warning(
                "[%s] Rate limit (429) on retry — trying fallback provider.",
                self.provider,
            )
        except Exception as exc:
            raise RuntimeError(f"[{self.provider}] API call failed: {exc}") from exc

        # Fallback provider attempt
        return self._try_fallback(messages, system_prompt, temperature, top_p, max_tokens)

    # ------------------------------------------------------------------
    # Path 3 — Anthropic
    # ------------------------------------------------------------------

    def _chat_anthropic(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
    ) -> str:
        try:
            import anthropic as _anthropic  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "The 'anthropic' package is required for provider 'anthropic'. "
                "Run: pip install anthropic"
            ) from exc

        if not self.api_key:
            raise RuntimeError(
                "No API key found for provider 'anthropic'. "
                "Set ANTHROPIC_API_KEY in backend/.env"
            )

        client = _anthropic.Anthropic(api_key=self.api_key)

        anth_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m["role"] != "system"
        ]

        kwargs: Dict[str, Any] = {
            "model":       self.model,
            "messages":    anth_messages,
            "temperature": temperature,
            "max_tokens":  max_tokens or 1024,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        # Attempt 1
        try:
            response = client.messages.create(**kwargs)
            return response.content[0].text if response.content else ""
        except _anthropic.RateLimitError:
            logger.warning(
                "[anthropic] Rate limit (429) on first attempt — waiting %ds before retry.",
                _RETRY_DELAY_S,
            )
            time.sleep(_RETRY_DELAY_S)
        except Exception as exc:
            raise RuntimeError(f"[anthropic] API call failed: {exc}") from exc

        # Attempt 2
        try:
            response = client.messages.create(**kwargs)
            return response.content[0].text if response.content else ""
        except _anthropic.RateLimitError:
            logger.warning(
                "[anthropic] Rate limit (429) on retry — trying fallback provider."
            )
        except Exception as exc:
            raise RuntimeError(f"[anthropic] API call failed: {exc}") from exc

        # Fallback provider attempt
        return self._try_fallback(messages, system_prompt, temperature, None, max_tokens)

    # ------------------------------------------------------------------
    # Fallback helper
    # ------------------------------------------------------------------

    def _try_fallback(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str],
        temperature: float,
        top_p: Optional[float],
        max_tokens: Optional[int],
    ) -> str:
        """Try the fallback provider once. Raises LLMUnavailableError if unavailable."""
        if not self.fallback_provider or not self.fallback_api_key:
            raise LLMUnavailableError(
                f"[{self.provider}] Rate limit exhausted and no fallback provider configured. "
                "LLM service is currently unavailable."
            )

        logger.warning(
            "[%s] Attempting fallback provider '%s'.",
            self.provider, self.fallback_provider,
        )
        fallback = LLMClient(
            provider=self.fallback_provider,
            model=self.fallback_model,
            api_key=self.fallback_api_key,
            default_temperature=temperature,
            default_top_p=top_p if top_p is not None else self.default_top_p,
            default_num_ctx=self.default_num_ctx,
            timeout=self.timeout,
        )
        try:
            return fallback.chat(
                messages,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except (LLMUnavailableError, RuntimeError) as exc:
            raise LLMUnavailableError(
                f"[{self.provider}] Primary rate-limited; fallback '{self.fallback_provider}' "
                f"also failed. LLM service is currently unavailable. Detail: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Singleton-style accessor
# ---------------------------------------------------------------------------

_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """
    Get (and lazily create) the project-wide LLMClient.

    Provider, model, and credentials are read from graph_rag.config.CONFIG.
    """
    global _client
    if _client is None:
        _client = LLMClient(
            provider=CONFIG.llm_provider,
            model=CONFIG.llm_model,
            api_key=CONFIG.llm_api_key,
            fallback_provider=CONFIG.llm_fallback_provider,
            fallback_model=CONFIG.llm_fallback_model,
            fallback_api_key=CONFIG.llm_fallback_api_key,
            ollama_base_url=CONFIG.ollama_base_url,
            default_temperature=CONFIG.llm_temperature,
            default_top_p=CONFIG.llm_top_p,
            default_num_ctx=CONFIG.llm_num_ctx,
            timeout=CONFIG.llm_timeout,
        )
        logger.info(
            "LLMClient initialised | provider=%s | model=%s | fallback=%s",
            _client.provider,
            _client.model,
            _client.fallback_provider or "none",
        )
    return _client
