import logging
import os
import time

from ai_content_service import AIProvider, AIProviderError, AIAuthenticationError, AITimeoutError, RateLimitError

logger = logging.getLogger(__name__)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

DEEPSEEK_PRICING = {
    "deepseek-v4-flash": {"input_cache_miss": 0.14, "input_cache_hit": 0.0028, "output": 0.28},
    "deepseek-v4-pro": {"input_cache_miss": 0.435, "input_cache_hit": 0.003625, "output": 0.87},
}


class DeepSeekProvider(AIProvider):
    """DeepSeek provider for AI content generation using the official DeepSeek API.

    Uses the OpenAI-compatible endpoint at https://api.deepseek.com.
    Model defaults to deepseek-v4-flash, configurable via DEEPSEEK_MODEL env var.
    API key read from DEEPSEEK_API_KEY env var.
    """

    def __init__(self) -> None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise AIAuthenticationError("DEEPSEEK_API_KEY is not set in the environment.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise AIProviderError("The openai package is not installed.") from exc

        self._model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self._client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL, timeout=120)

        try:
            import openai as _openai_module
            self._sdk_version = getattr(_openai_module, "__version__", None)
        except Exception:
            self._sdk_version = None

        logger.info(
            "DeepSeek AI provider initialised: model=%s base_url=%s sdk=%s",
            self._model,
            DEEPSEEK_BASE_URL,
            self._sdk_version,
        )

    def generate(self, prompt: str, system_message: str = "", max_tokens: int = 4096) -> dict:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        t_start = time.monotonic()

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.3,
            )
        except Exception as exc:
            elapsed = time.monotonic() - t_start
            logger.error(
                "DeepSeek generation failed after %.2fs: model=%s error=%s",
                elapsed,
                self._model,
                exc,
            )
            self._handle_api_error(exc)
            return {}  # unreachable

        elapsed = time.monotonic() - t_start

        content = response.choices[0].message.content or ""
        tokens_in = response.usage.prompt_tokens if response.usage else 0
        tokens_out = response.usage.completion_tokens if response.usage else 0

        logger.info(
            "DeepSeek generation complete: model=%s tokens_in=%d tokens_out=%d elapsed=%.2fs",
            self._model,
            tokens_in,
            tokens_out,
            elapsed,
        )

        return {
            "content": content.strip(),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "model": self._model,
        }

    def calculate_cost(self, tokens_in: int, tokens_out: int) -> float:
        pricing = DEEPSEEK_PRICING.get(
            self._model,
            DEEPSEEK_PRICING["deepseek-v4-flash"],
        )
        input_cost = (tokens_in / 1_000_000) * pricing["input_cache_miss"]
        output_cost = (tokens_out / 1_000_000) * pricing["output"]
        return round(input_cost + output_cost, 6)

    def is_available(self) -> bool:
        return bool(os.getenv("DEEPSEEK_API_KEY"))

    def get_models(self) -> list:
        return list(DEEPSEEK_PRICING.keys())

    def _handle_api_error(self, exc: Exception) -> None:
        try:
            import openai
        except ImportError:
            raise AIProviderError("Dependency missing.") from exc

        if isinstance(exc, openai.APITimeoutError):
            raise AITimeoutError() from exc
        if isinstance(exc, openai.APIConnectionError):
            raise AIProviderError("Could not connect to DeepSeek API.") from exc
        if isinstance(exc, openai.AuthenticationError):
            raise AIAuthenticationError("Invalid DeepSeek API key.") from exc
        if isinstance(exc, openai.RateLimitError):
            raise RateLimitError() from exc
        if isinstance(exc, openai.InternalServerError):
            raise AIProviderError("DeepSeek server error.") from exc
        if isinstance(exc, openai.APIStatusError):
            raise AIProviderError(
                f"DeepSeek API error (HTTP {getattr(exc, 'status_code', 'unknown')})."
            ) from exc

        raise AIProviderError("DeepSeek generation failed.") from exc
