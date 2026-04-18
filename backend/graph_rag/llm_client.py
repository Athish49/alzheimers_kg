"""
graph_rag.llm_client
--------------------

Thin wrapper around a local Ollama LLM (llama3.2:3b by default).

- Uses the HTTP API exposed by the Ollama daemon.
- Default model / params are read from `graph_rag.config.CONFIG`.
- Provides:
    * low-level `.chat()` for arbitrary message lists
    * convenience `.simple_qa()` for RAG-style question answering

This module assumes Ollama is running locally, e.g.:

    ollama pull llama3.2:3b
    ollama serve

Docs: https://github.com/ollama/ollama
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from .config import CONFIG


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------


@dataclass
class LLMClient:
    """
    Client for a local Ollama LLM.

    Parameters
    ----------
    base_url:
        Base URL for the Ollama HTTP API (including /api prefix).
        Usually "http://localhost:11434/api".
    model:
        Model name as known to Ollama, e.g. "llama3.2:3b".
    default_temperature:
        Default sampling temperature.
    default_top_p:
        Default top-p value.
    default_num_ctx:
        Default context window (tokens), passed via `options.num_ctx`.
    timeout:
        HTTP timeout in seconds for each request.
    """

    base_url: str = "http://localhost:11434/api"
    model: str = "llama3.2:3b"
    default_temperature: float = 0.2
    default_top_p: float = 0.9
    default_num_ctx: int = 4096
    timeout: int = 120

    # ------------------------------------------------------------------
    # Core chat interface
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
        Send a chat-style request to the Ollama model.

        Parameters
        ----------
        messages:
            List of {"role": "user" | "assistant" | "system", "content": "..."}.
            If `system_prompt` is provided, it will be injected as the first
            system message (before your list).
        system_prompt:
            Optional system message to prepend.
        temperature, top_p, num_ctx, max_tokens:
            Optional overrides for sampling / context parameters.

        Returns
        -------
        The assistant's response content as a plain string.
        """
        # Prepend system message if provided
        final_messages: List[Dict[str, str]] = []
        if system_prompt:
            final_messages.append({"role": "system", "content": system_prompt})
        final_messages.extend(messages)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": final_messages,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.default_temperature,
                "top_p": top_p if top_p is not None else self.default_top_p,
                "num_ctx": num_ctx if num_ctx is not None else self.default_num_ctx,
            },
        }

        # Only set num_predict if explicitly requested
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        url = f"{self.base_url}/chat"
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
        except requests.RequestException as e:
            raise RuntimeError(
                f"Failed to connect to Ollama at {url}. "
                f"Is the daemon running? Original error: {e}"
            ) from e

        if resp.status_code != 200:
            raise RuntimeError(
                f"Ollama returned status {resp.status_code}: {resp.text[:500]}"
            )

        data = resp.json()
        # Non-streaming /chat returns a single message object
        message = data.get("message") or {}
        content = message.get("content", "")
        return content

    # ------------------------------------------------------------------
    # Convenience helper for RAG-style QA
    # ------------------------------------------------------------------

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
        RAG-style helper: feed `context` + `question` to the LLM and
        return an answer.

        Parameters
        ----------
        question:
            User question.
        context:
            Retrieved context (graph neighborhood, documents, etc.).
        system_prompt:
            Optional additional system instructions; if None, a default
            Graph-RAG system prompt is used.
        temperature:
            Optional override for sampling temperature.
        max_tokens:
            Optional upper bound on generated tokens.

        Returns
        -------
        Answer string.
        """
        if system_prompt is None:
            system_prompt = (
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


# ---------------------------------------------------------------------
# Singleton-style accessor
# ---------------------------------------------------------------------

_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """
    Get (and lazily create) the project-wide LLMClient.

    Uses defaults from graph_rag.config.CONFIG, but you can still override
    at the instance level if needed.
    """
    global _client
    if _client is None:
        _client = LLMClient(
            base_url=CONFIG.ollama_base_url,
            model=CONFIG.llm_model,
            default_temperature=CONFIG.llm_temperature,
            default_top_p=CONFIG.llm_top_p,
            default_num_ctx=CONFIG.llm_num_ctx,
            timeout=CONFIG.llm_timeout,
        )
    return _client