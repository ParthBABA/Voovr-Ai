import json
import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class AIContentError(Exception):
    """Base exception for AI content generation failures."""
    def __init__(self, message: str, code: str = "AI_CONTENT_ERROR"):
        super().__init__(message)
        self.code = code


class AIProviderError(AIContentError):
    """Raised when the AI provider returns an error."""
    def __init__(self, message: str = "AI provider returned an error."):
        super().__init__(message, code="AI_PROVIDER_ERROR")


class AITimeoutError(AIContentError):
    """Raised when the AI request times out."""
    def __init__(self, message: str = "AI request timed out."):
        super().__init__(message, code="TIMEOUT")


class AIAuthenticationError(AIContentError):
    """Raised when the API key is invalid."""
    def __init__(self, message: str = "AI API key is invalid."):
        super().__init__(message, code="AUTH_ERROR")


class RateLimitError(AIContentError):
    """Raised when rate limited."""
    def __init__(self, message: str = "Rate limit exceeded. Please try again later."):
        super().__init__(message, code="RATE_LIMIT")


class AIProvider(ABC):
    """Abstract base for AI content generation providers."""
    
    @abstractmethod
    def generate(self, prompt: str, system_message: str = "", max_tokens: int = 4096) -> dict:
        """
        Generate content using the AI provider.
        
        Args:
            prompt: The user prompt.
            system_message: Optional system message to set context.
            max_tokens: Maximum tokens in the response.
            
        Returns:
            Dictionary with keys:
                - content: The generated text content
                - tokens_in: Input tokens used
                - tokens_out: Output tokens used
                - model: The model used
        """
        pass
    
    @abstractmethod
    def calculate_cost(self, tokens_in: int, tokens_out: int) -> float:
        """Calculate cost in USD for the given token counts."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and configured."""
        pass
    
    @abstractmethod
    def get_models(self) -> list:
        """Return list of available models."""
        pass


class OpenAIProvider(AIProvider):
    """OpenAI provider for AI content generation."""
    
    # Pricing per 1M tokens (USD) as of 2024
    PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    }
    
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise AIAuthenticationError("OPENAI_API_KEY is not set.")
        
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise AIProviderError("The openai package is not installed.") from exc
        
        self._model = os.getenv("OPENAI_AI_MODEL", "gpt-4o-mini")
        self._client = OpenAI(api_key=api_key, timeout=120)
        
        try:
            import openai as _openai_module
            self._sdk_version = getattr(_openai_module, "__version__", None)
        except Exception:
            self._sdk_version = None
        
        logger.info("OpenAI AI provider initialised: model=%s sdk=%s", self._model, self._sdk_version)
    
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
            logger.error("AI generation failed after %.2fs: model=%s error=%s", elapsed, self._model, exc)
            self._handle_api_error(exc)
            return {}  # unreachable
        
        elapsed = time.monotonic() - t_start
        
        content = response.choices[0].message.content or ""
        tokens_in = response.usage.prompt_tokens if response.usage else 0
        tokens_out = response.usage.completion_tokens if response.usage else 0
        
        logger.info(
            "AI generation complete: model=%s tokens_in=%d tokens_out=%d elapsed=%.2fs",
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
        pricing = self.PRICING.get(self._model, self.PRICING["gpt-4o-mini"])
        input_cost = (tokens_in / 1_000_000) * pricing["input"]
        output_cost = (tokens_out / 1_000_000) * pricing["output"]
        return round(input_cost + output_cost, 6)
    
    def is_available(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))
    
    def get_models(self) -> list:
        return list(self.PRICING.keys())
    
    def _handle_api_error(self, exc: Exception) -> None:
        try:
            import openai
        except ImportError:
            raise AIProviderError("Dependency missing.") from exc
        
        if isinstance(exc, openai.APITimeoutError):
            raise AITimeoutError() from exc
        if isinstance(exc, openai.APIConnectionError):
            raise AIProviderError("Could not connect to OpenAI API.") from exc
        if isinstance(exc, openai.AuthenticationError):
            raise AIAuthenticationError("Invalid API key.") from exc
        if isinstance(exc, openai.RateLimitError):
            raise RateLimitError() from exc
        if isinstance(exc, openai.InternalServerError):
            raise AIProviderError("OpenAI server error.") from exc
        if isinstance(exc, openai.APIStatusError):
            raise AIProviderError(f"API error (HTTP {getattr(exc, 'status_code', 'unknown')}).") from exc
        
        raise AIProviderError("AI generation failed.") from exc


# ---------------------------------------------------------------------------
# Prompt Registry
# ---------------------------------------------------------------------------

PROMPT_REGISTRY = {
    "NOTES": {
        "version": "2.0",
        "system_message": (
            "You are one of the world's best educators. "
            "Your job is NOT to summarize. "
            "Your job is to transform a lecture, meeting, classroom recording or educational discussion "
            "into the highest quality study material. "
            "The student should NEVER need to read the original transcript again. "
            "Do NOT copy the transcript. Rewrite everything. Remove repetition. Remove filler words. "
            "Remove unrelated conversation. Merge duplicate ideas. Use simple English. "
            "If the transcript contains technical terms, explain them naturally. "
            "Never hallucinate. Never invent information. Never add concepts not present in the transcript. "
            "If some information is missing, do not guess."
        ),
        "prompt_template": (
            "Convert the following transcript into an organized, exam-ready Smart Study Guide.\n\n"
            "The output MUST follow this exact structure with ALL sections:\n\n"
            "# Smart Notes\n\n"
            "---\n\n"
            "## 1. Topic Overview\n"
            "Explain the entire lecture in 5-8 concise bullet points. "
            "A student should understand the complete lecture in under one minute.\n\n"
            "---\n\n"
            "## 2. Core Concepts\n"
            "Break the lecture into logical sections. For every concept include:\n"
            "- Definition\n"
            "- Explanation\n"
            "- Why it matters\n"
            "- Real-world example\n"
            "- Common mistake (if applicable)\n\n"
            "---\n\n"
            "## 3. Key Points\n"
            "Create concise revision bullets. Maximum 15 bullets. Only exam-important information.\n\n"
            "---\n\n"
            "## 4. Visual Comparison Tables\n"
            "Whenever possible generate comparison tables using markdown. "
            "Examples: Differences between concepts, advantages vs disadvantages, features vs limitations.\n\n"
            "---\n\n"
            "## 5. Flowcharts\n"
            "Whenever a process is explained, represent it as ASCII flowcharts using arrows.\n\n"
            "---\n\n"
            "## 6. Mind Map\n"
            "Generate a small text-based mind map using tree structure with branches.\n\n"
            "---\n\n"
            "## 7. Important Numbers\n"
            "Place any dates, statistics, percentages, years, or key values in a separate section.\n\n"
            "---\n\n"
            "## 8. Important Formulas\n"
            "If formulas exist, create a dedicated section. Include meaning of every variable.\n\n"
            "---\n\n"
            "## 9. Memory Tricks\n"
            "Create easy tricks, mnemonics, acronyms, or memory shortcuts when possible. "
            "Never invent facts.\n\n"
            "---\n\n"
            "## 10. Real World Examples\n"
            "For every major concept provide at least one practical example.\n\n"
            "---\n\n"
            "## 11. Frequently Confused Concepts\n"
            "If two concepts are similar, explain what students usually confuse "
            "and how to remember the difference.\n\n"
            "---\n\n"
            "## 12. Exam Focus\n"
            "Generate prioritized topics with star ratings:\n"
            "- Most Important Topics (5 stars)\n"
            "- Important Topics (4 stars)\n"
            "- Revision Topics (3 stars)\n\n"
            "---\n\n"
            "## 13. One Minute Revision\n"
            "End with 10-15 ultra-short bullets. A student should revise the whole lecture in one minute.\n\n"
            "---\n\n"
            "FORMATTING RULES:\n"
            "- Use Markdown headings, bullet points, tables, flowcharts, bold keywords\n"
            "- Do NOT write large paragraphs. Keep paragraphs under 4 lines\n"
            "- Maximize readability and visual structure\n\n"
            "Transcript:\n{transcript}"
        ),
    },
    "FLASHCARDS": {
        "version": "2.0",
        "system_message": (
            "You are an expert at creating study materials. "
            "Given a transcript, create high-quality flashcards for effective learning. "
            "Each flashcard should test a distinct concept or fact. "
            "Make fronts clear and specific. Make backs concise but complete."
        ),
        "prompt_template": (
            "Create flashcards from the following transcript.\n\n"
            "Return a JSON array of objects, each with 'front' and 'back' keys.\n"
            "Create 10-20 flashcards covering the most important concepts.\n"
            "Example format: [{\"front\": \"What is X?\", \"back\": \"X is...\"}]\n\n"
            "Focus on:\n"
            "- Key definitions and terms\n"
            "- Cause-and-effect relationships\n"
            "- Important facts and statistics\n"
            "- Process steps\n"
            "- Common misconceptions\n\n"
            "Return ONLY the JSON array, no extra text.\n\n"
            "Transcript:\n{transcript}"
        ),
    },
    "KNOWLEDGE": {
        "version": "2.0",
        "system_message": (
            "You are an expert at extracting knowledge from educational content. "
            "Given a transcript, extract key concepts, terms, and relationships "
            "in a structured format for building a knowledge base."
        ),
        "prompt_template": (
            "Extract knowledge from the following transcript.\n\n"
            "Return a JSON object with:\n"
            "- concepts: array of key concepts, each with:\n"
            "  - name: the concept name\n"
            "  - definition: clear explanation\n"
            "  - importance: 1-5 scale\n"
            "  - category: grouping label\n"
            "- terms: array of technical terms with term and definition\n"
            "- relationships: array describing connections between concepts (e.g. 'A depends on B')\n"
            "- summary: a 2-3 sentence overview of the knowledge domain\n\n"
            "Return ONLY the JSON object, no extra text.\n\n"
            "Transcript:\n{transcript}"
        ),
    },
    "MCQS": {
        "version": "2.0",
        "system_message": (
            "You are an expert at creating assessment questions. "
            "Given a transcript, create multiple choice questions for testing understanding. "
            "Questions should be challenging but fair. "
            "Distractors should be plausible but clearly incorrect."
        ),
        "prompt_template": (
            "Create multiple choice questions from the following transcript.\n\n"
            "Return a JSON array of objects with:\n"
            "- question: the question text\n"
            "- options: array of 4 options\n"
            "- correctIndex: index of correct answer (0-3)\n"
            "- explanation: brief explanation of the correct answer\n\n"
            "Create 8-15 questions covering:\n"
            "- Core concepts and definitions\n"
            "- Application and reasoning questions\n"
            "- Common misconceptions\n"
            "- Important details and facts\n\n"
            "Return ONLY the JSON array, no extra text.\n\n"
            "Transcript:\n{transcript}"
        ),
    },
    "REVISION": {
        "version": "2.0",
        "system_message": (
            "You are an expert at creating revision summaries. "
            "Given a transcript, create a concise, high-impact revision guide "
            "that covers all essential information for exam preparation."
        ),
        "prompt_template": (
            "Create a revision summary from the following transcript.\n\n"
            "Include these sections:\n\n"
            "### Key Points\n"
            "- 10-15 essential bullet points\n\n"
            "### Important Formulas or Code\n"
            "- Any formulas, code snippets, or key equations\n\n"
            "### Common Pitfalls\n"
            "- Mistakes students often make\n"
            "- Misconceptions to avoid\n\n"
            "### Quick Reference\n"
            "- A concise summary table or list\n\n"
            "### Must-Know Facts\n"
            "- Critical numbers, dates, or statistics\n\n"
            "Keep it exam-focused and concise.\n\n"
            "Transcript:\n{transcript}"
        ),
    },
    "CHAT": {
        "version": "1.0",
        "system_message": (
            "You are a helpful assistant that answers questions based on a transcript. "
            "Only answer based on the information in the transcript. "
            "If the transcript doesn't contain the answer, say so."
        ),
        "prompt_template": (
            "Context transcript:\n{transcript}\n\n"
            "Question: {question}"
        ),
    },
}


# ---------------------------------------------------------------------------
# DB helpers for AI content
# ---------------------------------------------------------------------------

from backend.google_auth import get_db_connection


def save_ai_content(meeting_id: int, content_type: str, content_json: str, 
                    provider: str, model: str, prompt_version: str) -> int:
    """Save generated AI content to the database."""
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    conn = get_db_connection()
    cursor = conn.execute(
        """
        INSERT INTO meeting_ai_content 
        (meeting_id, content_type, version, content_json, provider, model, prompt_version, created_at)
        VALUES (?, ?, 1, ?, ?, ?, ?, ?)
        """,
        (meeting_id, content_type, content_json, provider, model, prompt_version, now),
    )
    content_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return content_id


def get_ai_content(meeting_id: int, content_type: str) -> dict | None:
    """Get the latest AI content of a given type for a meeting."""
    conn = get_db_connection()
    row = conn.execute(
        """
        SELECT * FROM meeting_ai_content 
        WHERE meeting_id = ? AND content_type = ?
        ORDER BY version DESC, id DESC
        LIMIT 1
        """,
        (meeting_id, content_type),
    ).fetchone()
    conn.close()
    
    if not row:
        return None
    
    return dict(row)


def log_generation(meeting_id: int, content_type: str, provider: str, model: str,
                   prompt_version: str, tokens_in: int, tokens_out: int,
                   latency_ms: int, cost_usd: float, status: str) -> None:
    """Log a generation event for cost tracking."""
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO generation_logs
        (meeting_id, content_type, provider, model, prompt_version, tokens_in, tokens_out, 
         latency_ms, cost_usd, status, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (meeting_id, content_type, provider, model, prompt_version, tokens_in, tokens_out,
         latency_ms, cost_usd, status, now),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_ai_provider() -> AIProvider:
    """Factory function that returns the configured AI provider."""
    provider_name = os.getenv("AI_PROVIDER", "deepseek").lower()
    if provider_name == "deepseek":
        from deepseek_service import DeepSeekProvider
        return DeepSeekProvider()
    if provider_name == "openai":
        return OpenAIProvider()
    raise AIProviderError(f"Unknown AI provider: '{provider_name}'")
