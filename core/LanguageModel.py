# ============================================================
# LanguageModel.py
# ============================================================
# Unified LLM wrapper supporting:
#   - OpenAI  (gpt-*, o1, o3, o4-mini, ...)
#   - Anthropic / Claude  (claude-*)          ← added vs original
#   - DeepSeek (deepseek-*)
#   - Qwen / LLaMA via zhizengzeng proxy
#   - Google Gemini (gemini-*)
#
# Faithfully replicates the original A2A-NT LanguageModel.py,
# with Claude support added and minor defensive improvements.
# ============================================================

import os
import json
import logging
import time
from typing import List, Dict, Optional

from openai import OpenAI, OpenAIError
from Config import OPENAI_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, GOOGLE_API_KEY, ZHI_API_KEY

try:
    import anthropic as anthropic_sdk
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

try:
    from google import genai
    _GOOGLE_AVAILABLE = True
except ImportError:
    _GOOGLE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LanguageModel:
    """
    Provider-agnostic LLM interface.

    Usage:
        lm = LanguageModel("gpt-4o")
        text = lm.get_response("Hello!")
        text = lm.get_chat_response([{"role": "user", "content": "Hello!"}])
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self._api_key_index = 0
        self._last_request_time = 0.0
        self._rate_limit_delay = 1.0  # seconds between requests
        self._setup_provider()

    # ------------------------------------------------------------------
    # Provider setup
    # ------------------------------------------------------------------

    def _setup_provider(self) -> None:
        name = self.model_name.lower()

        if "claude" in name:
            if not _ANTHROPIC_AVAILABLE:
                raise ImportError(
                    "anthropic package not installed. Run: pip install anthropic"
                )
            self.provider = "anthropic"
            self.api_keys = [ANTHROPIC_API_KEY]
            self.client = anthropic_sdk.Anthropic(api_key=ANTHROPIC_API_KEY)

        elif "gpt" in name or self._looks_like_openai_reasoning(name):
            self.provider = "openai"
            self.api_keys = [OPENAI_API_KEY]
            self.client = OpenAI(api_key=OPENAI_API_KEY)

        elif "deepseek" in name:
            self.provider = "deepseek"
            self.api_keys = DEEPSEEK_API_KEY if isinstance(DEEPSEEK_API_KEY, list) else [DEEPSEEK_API_KEY]
            self._setup_client_with_next_key()

        elif "qwen" in name or "llama" in name:
            self.provider = "zhizengzeng"
            self.api_keys = ZHI_API_KEY if isinstance(ZHI_API_KEY, list) else [ZHI_API_KEY]
            self._setup_client_with_next_key()

        elif "gemini" in name:
            if not _GOOGLE_AVAILABLE:
                raise ImportError(
                    "google-genai package not installed. Run: pip install google-genai"
                )
            self.provider = "google"
            self.api_keys = [GOOGLE_API_KEY]
            self.client = genai.Client(api_key=GOOGLE_API_KEY)

        else:
            raise ValueError(
                f"Unsupported model: '{self.model_name}'. "
                "Expected a model name containing one of: claude, gpt, o1, o3, deepseek, qwen, llama, gemini"
            )

    @staticmethod
    def _looks_like_openai_reasoning(name: str) -> bool:
        """Detect o1/o3/o4-mini style OpenAI reasoning models."""
        import re
        return bool(re.search(r'\bo[134]', name))

    def _setup_client_with_next_key(self) -> None:
        if not self.api_keys:
            raise ValueError("No API keys available for this provider.")
        self._api_key_index = (self._api_key_index + 1) % len(self.api_keys)
        key = self.api_keys[self._api_key_index]

        if self.provider == "deepseek":
            self.client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
        elif self.provider == "zhizengzeng":
            self.client = OpenAI(api_key=key, base_url="https://api.zhizengzeng.com/v1/chat/completions")

        logger.info(f"[{self.provider}] Using API key: {key[:8]}...")

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _enforce_rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < self._rate_limit_delay:
            time.sleep(self._rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    # ------------------------------------------------------------------
    # Core API call
    # ------------------------------------------------------------------

    def _make_api_call(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> Optional[str]:

        max_retries = 5
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                self._enforce_rate_limit()

                # ---- Anthropic ----
                if self.provider == "anthropic":
                    system_msg = next(
                        (m["content"] for m in messages if m["role"] == "system"), ""
                    )
                    chat_msgs = [m for m in messages if m["role"] != "system"]
                    # Anthropic requires alternating user/assistant turns
                    chat_msgs = self._ensure_alternating(chat_msgs)
                    response = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        system=system_msg,
                        messages=chat_msgs,
                    )
                    return response.content[0].text

                # ---- Google Gemini ----
                elif self.provider == "google":
                    validated = []
                    for m in messages:
                        if m["role"] == "system":
                            validated.append({"role": "user", "content": f"System: {m['content']}"})
                        else:
                            validated.append(m)
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=[m["content"] for m in validated],
                        generation_config={
                            "temperature": temperature,
                            "max_output_tokens": max_tokens,
                        },
                    )
                    return response.text

                # ---- OpenAI-compatible (openai / deepseek / zhizengzeng) ----
                else:
                    validated = self._validate_openai_messages(messages)
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=validated,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                    except Exception:
                        # Some OpenAI reasoning models use max_completion_tokens
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=validated,
                            max_completion_tokens=max_tokens,
                        )

                    if (
                        response is None
                        or not getattr(response, "choices", None)
                        or not getattr(response.choices[0], "message", None)
                    ):
                        raise ValueError("Malformed API response.")

                    return response.choices[0].message.content

            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if self.provider in ("deepseek", "zhizengzeng") and attempt < max_retries - 1:
                    self._setup_client_with_next_key()
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error("Max retries reached. Returning None.")
                    return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_openai_messages(messages: List[Dict]) -> List[Dict]:
        """Ensure every message has a valid role and non-None content."""
        valid_roles = {"system", "user", "assistant", "tool"}
        out = []
        for m in messages:
            role = m.get("role", "user")
            if role not in valid_roles:
                role = "user"
            content = m.get("content") or ""
            out.append({"role": role, "content": content})
        return out

    @staticmethod
    def _ensure_alternating(messages: List[Dict]) -> List[Dict]:
        """
        Anthropic requires strictly alternating user/assistant turns.
        Merge consecutive same-role messages by concatenating content.
        """
        if not messages:
            return messages
        out = [messages[0].copy()]
        for m in messages[1:]:
            if m["role"] == out[-1]["role"]:
                out[-1]["content"] += "\n" + m["content"]
            else:
                out.append(m.copy())
        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_response(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> Optional[str]:
        """Single-turn call — wraps prompt as a user message."""
        return self._make_api_call(
            [{"role": "user", "content": prompt}], temperature, max_tokens
        )

    def get_chat_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> Optional[str]:
        """Multi-turn call — pass the full message list."""
        return self._make_api_call(messages, temperature, max_tokens)
