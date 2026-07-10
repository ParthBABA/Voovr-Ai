import logging
import mimetypes
import os
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """Base exception for transcription failures."""

    def __init__(self, message: str, code: str = "TRANSCRIPTION_ERROR"):
        super().__init__(message)
        self.code = code


class InvalidFileError(TranscriptionError):
    """Raised when the uploaded file is invalid or unsupported."""

    def __init__(self, message: str = "Invalid or unsupported audio file."):
        super().__init__(message, code="INVALID_FILE")


class APIKeyMissingError(TranscriptionError):
    """Raised when the provider API key is not configured or rejected."""

    def __init__(self, message: str = "Transcription API key is not configured."):
        super().__init__(message, code="API_KEY_MISSING")


class TranscriptionTimeoutError(TranscriptionError):
    """Raised when the transcription request times out."""

    def __init__(self, message: str = "Transcription request timed out."):
        super().__init__(message, code="TIMEOUT")


class APIFailureError(TranscriptionError):
    """Raised when the provider API returns an error."""

    def __init__(self, message: str = "Transcription API returned an error."):
        super().__init__(message, code="API_FAILURE")


class ModelAccessError(TranscriptionError):
    """Raised when the API key's project lacks access to the requested model."""

    def __init__(self, message: str = "The transcription model is not available."):
        super().__init__(message, code="MODEL_ACCESS_ERROR")


class TranscriptionProvider(ABC):
    """Abstract base for transcription providers.

    Implement ``transcribe()`` in a subclass to add a new provider.
    The rest of the application interacts only through this interface.
    """

    @abstractmethod
    def transcribe(self, file_path: str, filename: str) -> str:
        """Transcribe the audio file and return plain text.

        Args:
            file_path: Absolute path to the audio file on disk.
            filename: Original filename (used for validation / logging).

        Returns:
            The transcribed text.

        Raises:
            InvalidFileError: If the file is not supported.
            APIKeyMissingError: If the API key is missing.
            TranscriptionTimeoutError: If the request times out.
            ModelAccessError: If the project lacks model access.
            APIFailureError: If the API returns an error.
            TranscriptionError: For any other transcription failure.
        """
        ...


class OpenAITranscriptionProvider(TranscriptionProvider):
    """Transcription provider using the official OpenAI Speech-to-Text API.

    Uses the ``gpt-4o-mini-transcribe`` model by default.
    Configurable via ``OPENAI_TRANSCRIPTION_MODEL`` and
    ``OPENAI_TRANSCRIPTION_PROMPT`` environment variables.
    """

    DEFAULT_PROMPT = (
        "This audio is a meeting, lecture, interview, or discussion.\n"
        "Return an accurate transcript.\n"
        "Preserve:\n"
        "- punctuation\n"
        "- capitalization\n"
        "- technical terms\n"
        "- programming keywords\n"
        "- URLs\n"
        "- numbers\n"
        "- product names\n"
        "Do NOT summarize.\n"
        "Return only the transcript."
    )

    SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".webm", ".ogg", ".flac"}
    MIME_TYPES = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".webm": "audio/webm",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
    }
    MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB -- OpenAI limit
    TIMEOUT_SECONDS = 300  # 5 minutes

    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise APIKeyMissingError(
                "OPENAI_API_KEY is not set in the environment or .env file."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise TranscriptionError(
                "The openai package is not installed.", code="DEPENDENCY_MISSING"
            ) from exc

        self._model = os.getenv("OPENAI_TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe")
        self._prompt = os.getenv("OPENAI_TRANSCRIPTION_PROMPT") or self.DEFAULT_PROMPT
        self._client = OpenAI(api_key=api_key, timeout=self.TIMEOUT_SECONDS)

        try:
            import openai as _openai_module
            self._sdk_version = getattr(_openai_module, "__version__", None)
        except Exception:
            self._sdk_version = None

        logger.info(
            "OpenAI transcription provider initialised: model=%s response_format=text sdk=%s",
            self._model,
            self._sdk_version,
        )

    # ------------------------------------------------------------------
    # File validation
    # ------------------------------------------------------------------

    def _validate_file(self, file_path: str, filename: str) -> tuple[str, int]:
        """Validate the file and return (extension, size)."""
        ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""

        if ext not in self.SUPPORTED_EXTENSIONS:
            raise InvalidFileError(
                f"Unsupported file format '{ext}'. "
                f"Supported: {', '.join(sorted(e.lstrip('.') for e in self.SUPPORTED_EXTENSIONS))}."
            )

        if not os.path.isfile(file_path):
            raise InvalidFileError("The uploaded file does not exist on disk.")

        try:
            size = os.path.getsize(file_path)
        except OSError as exc:
            raise InvalidFileError("Cannot read the uploaded file.") from exc

        if size == 0:
            raise InvalidFileError("The uploaded file is empty.")

        if size > self.MAX_FILE_SIZE:
            raise InvalidFileError(
                f"File too large ({size // (1024 * 1024)} MB). "
                f"Maximum allowed is {self.MAX_FILE_SIZE // (1024 * 1024)} MB."
            )

        if not os.access(file_path, os.R_OK):
            raise InvalidFileError("The uploaded file is not readable.")

        mime, _ = mimetypes.guess_type(filename)
        expected_mime = self.MIME_TYPES.get(ext)

        if not expected_mime:
            raise InvalidFileError("The uploaded file type is not supported.")

        if mime is None:
            raise InvalidFileError("The uploaded file does not appear to be a valid audio file.")

        if mime != expected_mime:
            logger.warning(
                "Unexpected MIME type for upload: filename=%s detected=%s expected=%s",
                filename,
                mime,
                expected_mime,
            )

        return ext, size

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def transcribe(self, file_path: str, filename: str) -> str:
        """Send the audio file to the OpenAI transcription endpoint."""
        ext, file_size = self._validate_file(file_path, filename)

        logger.info(
            "Starting transcription: model=%s response_format=text file=%s size=%d",
            self._model,
            filename,
            file_size,
        )

        t_start = time.monotonic()

        try:
            with open(file_path, "rb") as audio_file:
                result = self._client.audio.transcriptions.create(
                    model=self._model,
                    file=audio_file,
                    response_format="text",
                    prompt=self._prompt,
                )
        except ImportError as exc:
            raise TranscriptionError(
                "The openai package is not installed.", code="DEPENDENCY_MISSING"
            ) from exc
        except Exception as exc:
            elapsed = time.monotonic() - t_start
            logger.error(
                "Transcription failed after %.2fs: model=%s file=%s error=%s",
                elapsed,
                self._model,
                filename,
                exc,
            )
            self._handle_api_error(exc)
            return ""  # unreachable, but satisfies type checker

        elapsed = time.monotonic() - t_start

        transcript: str = getattr(result, "text", "") or (result if isinstance(result, str) else "")

        if not transcript or not transcript.strip():
            raise APIFailureError("The transcription API returned an empty result.")

        logger.info(
            "Transcription complete: model=%s file=%s chars=%d elapsed=%.2fs",
            self._model,
            filename,
            len(transcript),
            elapsed,
        )

        return transcript.strip()

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def _handle_api_error(self, exc: Exception) -> None:
        """Translate OpenAI SDK exceptions into TranscriptionError subclasses."""
        logger.debug(
            "OpenAI API error detail: type=%s repr=%s str=%s status=%s code=%s",
            type(exc).__name__,
            repr(exc),
            str(exc),
            getattr(exc, "status_code", None),
            getattr(exc, "code", None),
        )

        try:
            import openai
        except ImportError:
            raise APIFailureError(
                "Transcription failed due to a missing dependency."
            ) from exc

        if isinstance(exc, openai.APITimeoutError):
            raise TranscriptionTimeoutError() from exc

        if isinstance(exc, openai.APIConnectionError):
            raise APIFailureError(
                "Could not connect to the OpenAI API."
            ) from exc

        if isinstance(exc, openai.AuthenticationError):
            raise APIKeyMissingError(
                "The OpenAI API key is invalid or expired."
            ) from exc

        if isinstance(exc, openai.RateLimitError):
            raise APIFailureError(
                "Rate limit exceeded. Please try again in a moment."
            ) from exc

        if isinstance(exc, openai.PermissionDeniedError):
            raise ModelAccessError(
                "The configured transcription model is unavailable for this OpenAI project.\n"
                "Verify:\n"
                "\u2022 API key\n"
                "\u2022 Project permissions\n"
                "\u2022 Model access"
            ) from exc

        if isinstance(exc, openai.BadRequestError):
            error_body = str(exc)
            if "model_not_found" in error_body or "does not have access to model" in error_body:
                raise ModelAccessError(
                    "The configured transcription model is unavailable for this OpenAI project.\n"
                    "Verify:\n"
                    "\u2022 API key\n"
                    "\u2022 Project permissions\n"
                    "\u2022 Model access"
                ) from exc
            raise APIFailureError(
                f"Invalid transcription request: {exc}"
            ) from exc

        if isinstance(exc, openai.UnprocessableEntityError):
            raise APIFailureError(
                "Could not process the audio data."
            ) from exc

        if isinstance(exc, openai.InternalServerError):
            raise APIFailureError(
                "OpenAI server error. Please try again later."
            ) from exc

        if isinstance(exc, openai.APIStatusError):
            raise APIFailureError(
                f"OpenAI API error (HTTP {getattr(exc, 'status_code', 'unknown')})."
            ) from exc

        raise APIFailureError("Transcription failed.") from exc


def get_transcription_provider() -> TranscriptionProvider:
    """Factory function that returns the configured transcription provider."""
    provider_name = os.getenv("TRANSCRIPTION_PROVIDER", "openai").lower()
    if provider_name == "openai":
        return OpenAITranscriptionProvider()
    raise TranscriptionError(
        f"Unknown transcription provider: '{provider_name}'",
        code="UNKNOWN_PROVIDER",
    )
