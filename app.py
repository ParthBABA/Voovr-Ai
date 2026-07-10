import json
import os
import tempfile
import time
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, session

from google_auth import get_db_connection, google_auth, init_db, init_oauth
from transcription_service import (
    APIFailureError,
    APIKeyMissingError,
    InvalidFileError,
    ModelAccessError,
    TranscriptionError,
    TranscriptionTimeoutError,
    get_transcription_provider,
)
from ai_content_service import (
    AIProviderError,
    AITimeoutError,
    AIAuthenticationError,
    RateLimitError,
    get_ai_provider,
    PROMPT_REGISTRY,
    save_ai_content,
    get_ai_content,
    log_generation,
)

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-secret")
app.config["SESSION_PERMANENT"] = True
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=30)

init_oauth(app)
init_db()
app.register_blueprint(google_auth)


# --------------------------
# Meeting DB helpers
# --------------------------

def create_meeting(user_id, file_name, file_size, file_type, file_duration, transcript):
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    conn = get_db_connection()
    cursor = conn.execute(
        """
        INSERT INTO meetings (user_id, title, file_name, file_size, file_type, file_duration, transcript, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'ready', ?, ?)
        """,
        (user_id, file_name, file_name, file_size, file_type, file_duration, transcript, now, now),
    )
    meeting_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return meeting_id


def get_meeting(meeting_id):
    conn = get_db_connection()
    meeting = conn.execute(
        "SELECT * FROM meetings WHERE id = ?", (meeting_id,)
    ).fetchone()
    conn.close()
    return meeting


# --------------------------
# Page Routes
# --------------------------

@app.route("/")
def home():
    if session.get("user"):
        return redirect("/dashboard")
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    user = session.get("user")
    if not user:
        return redirect("/")
    return render_template("dashboard.html", user=user)


@app.route("/workspace/audio")
def workspace_audio():
    user = session.get("user")
    if not user:
        return redirect("/")
    return render_template("audio_workspace.html", user=user)


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


# --------------------------
# API Routes
# --------------------------

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".webm", ".ogg", ".flac"}
MAX_UPLOAD_SIZE = 25 * 1024 * 1024  # 25 MB


@app.route("/api/transcribe", methods=["POST"])
def api_transcribe():
    user = session.get("user")
    if not user:
        return jsonify({"error": "Authentication required."}), 401

    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided."}), 400

    audio_file = request.files["audio"]
    if not audio_file.filename:
        return jsonify({"error": "No file selected."}), 400

    ext = "." + audio_file.filename.rsplit(".", 1)[-1].lower() if "." in audio_file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": f"Unsupported file format '{ext}'. Supported: MP3, WAV, M4A, AAC, WebM, OGG, FLAC."
        }), 400

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            audio_file.save(tmp)
            tmp_path = tmp.name

        file_size = os.path.getsize(tmp_path)
        if file_size > MAX_UPLOAD_SIZE:
            return jsonify({"error": "File too large. Maximum allowed is 25 MB."}), 400
        if file_size == 0:
            return jsonify({"error": "The uploaded file is empty."}), 400

        conn = get_db_connection()
        existing = conn.execute(
            "SELECT id, transcript FROM meetings WHERE user_id = ? AND file_name = ? AND file_size = ? AND deleted_at IS NULL",
            (user["id"], audio_file.filename, file_size),
        ).fetchone()
        conn.close()

        if existing:
            session["current_meeting_id"] = existing["id"]
            return jsonify({
                "meetingId": existing["id"],
                "transcript": existing["transcript"],
                "cached": True,
            })

        provider = get_transcription_provider()
        transcript = provider.transcribe(tmp_path, audio_file.filename)

        meeting_id = create_meeting(
            user_id=user["id"],
            file_name=audio_file.filename,
            file_size=file_size,
            file_type=ext.lstrip(".").upper(),
            file_duration=request.form.get("duration", ""),
            transcript=transcript,
        )

        session["current_meeting_id"] = meeting_id

        return jsonify({
            "meetingId": meeting_id,
            "transcript": transcript,
        })

    except InvalidFileError as exc:
        return jsonify({"error": str(exc)}), 400
    except ModelAccessError as exc:
        return jsonify({"error": str(exc)}), 500
    except APIKeyMissingError as exc:
        return jsonify({"error": str(exc)}), 500
    except TranscriptionTimeoutError as exc:
        return jsonify({"error": str(exc)}), 504
    except APIFailureError as exc:
        return jsonify({"error": str(exc)}), 502
    except TranscriptionError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:
        return jsonify({"error": "An unexpected error occurred. Please try again."}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@app.route("/api/meetings", methods=["GET"])
def api_list_meetings():
    user = session.get("user")
    if not user:
        return jsonify({"error": "Authentication required."}), 401

    conn = get_db_connection()
    rows = conn.execute(
        "SELECT id, title, file_name, file_size, file_type, file_duration, status, created_at "
        "FROM meetings WHERE user_id = ? AND deleted_at IS NULL AND archived_at IS NULL ORDER BY created_at DESC",
        (user["id"],),
    ).fetchall()
    conn.close()

    meetings = [
        {
            "meetingId": r["id"],
            "title": r["title"],
            "fileName": r["file_name"],
            "fileSize": r["file_size"],
            "fileType": r["file_type"],
            "fileDuration": r["file_duration"],
            "status": r["status"],
            "createdAt": r["created_at"],
        }
        for r in rows
    ]
    return jsonify(meetings)


@app.route("/api/meeting/<int:meeting_id>", methods=["GET"])
def api_get_meeting(meeting_id):
    user = session.get("user")
    if not user:
        return jsonify({"error": "Authentication required."}), 401

    meeting = get_meeting(meeting_id)
    if not meeting:
        return jsonify({"error": "Meeting not found."}), 404

    if meeting["user_id"] != user["id"]:
        return jsonify({"error": "Access denied."}), 403

    if meeting["deleted_at"] is not None:
        return jsonify({"error": "Meeting not found."}), 404

    return jsonify({
        "meetingId": meeting["id"],
        "title": meeting["title"],
        "fileName": meeting["file_name"],
        "fileSize": meeting["file_size"],
        "fileType": meeting["file_type"],
        "fileDuration": meeting["file_duration"],
        "transcript": meeting["transcript"],
        "status": meeting["status"],
        "createdAt": meeting["created_at"],
        "updatedAt": meeting["updated_at"],
        "archivedAt": meeting["archived_at"],
    })


# --------------------------
# AI Content API Routes
# --------------------------

ALLOWED_CONTENT_TYPES = {"NOTES", "FLASHCARDS", "KNOWLEDGE", "MCQS", "REVISION"}


@app.route("/api/meeting/<int:meeting_id>/generate/<content_type>", methods=["POST"])
def api_generate_content(meeting_id, content_type):
    user = session.get("user")
    if not user:
        return jsonify({"error": "Authentication required."}), 401

    content_type = content_type.upper()
    if content_type not in ALLOWED_CONTENT_TYPES:
        return jsonify({"error": f"Invalid content type. Allowed: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}"}), 400

    meeting = get_meeting(meeting_id)
    if not meeting:
        return jsonify({"error": "Meeting not found."}), 404
    if meeting["user_id"] != user["id"]:
        return jsonify({"error": "Access denied."}), 403
    if meeting["deleted_at"] is not None:
        return jsonify({"error": "Meeting not found."}), 404
    if not meeting["transcript"]:
        return jsonify({"error": "No transcript available for this meeting."}), 400

    prompt_config = PROMPT_REGISTRY.get(content_type)
    if not prompt_config:
        return jsonify({"error": "Content type not supported yet."}), 400

    existing = get_ai_content(meeting_id, content_type)
    if existing:
        try:
            return jsonify({
                "meetingId": meeting_id,
                "contentType": content_type,
                "content": json.loads(existing["content_json"]),
                "cached": True,
                "version": existing["version"],
                "createdAt": existing["created_at"],
            })
        except (json.JSONDecodeError, KeyError):
            pass

    try:
        provider = get_ai_provider()
    except Exception as exc:
        return jsonify({"error": f"AI provider not configured: {str(exc)}"}), 500

    provider_name = os.getenv("AI_PROVIDER", "deepseek").lower()

    user_question = ""
    if content_type == "CHAT":
        body = request.get_json(silent=True) or {}
        user_question = body.get("question", "").strip()
        if not user_question:
            return jsonify({"error": "A 'question' field is required for CHAT content type."}), 400

    try:
        prompt = prompt_config["prompt_template"].format(
            transcript=meeting["transcript"],
            question=user_question,
        )
    except KeyError:
        prompt = prompt_config["prompt_template"].format(transcript=meeting["transcript"])

    system_message = prompt_config.get("system_message", "")
    prompt_version = prompt_config.get("version", "1.0")

    t_start = time.monotonic()

    try:
        result = provider.generate(prompt, system_message=system_message)
    except AITimeoutError as exc:
        elapsed_ms = int((time.monotonic() - t_start) * 1000)
        log_generation(meeting_id, content_type, provider_name, provider._model,
                       prompt_version, 0, 0, elapsed_ms, 0.0, "TIMEOUT")
        return jsonify({"error": str(exc)}), 504
    except RateLimitError as exc:
        elapsed_ms = int((time.monotonic() - t_start) * 1000)
        log_generation(meeting_id, content_type, provider_name, provider._model,
                       prompt_version, 0, 0, elapsed_ms, 0.0, "RATE_LIMITED")
        return jsonify({"error": str(exc)}), 429
    except AIAuthenticationError as exc:
        return jsonify({"error": str(exc)}), 500
    except AIProviderError as exc:
        elapsed_ms = int((time.monotonic() - t_start) * 1000)
        log_generation(meeting_id, content_type, provider_name, provider._model,
                       prompt_version, 0, 0, elapsed_ms, 0.0, "ERROR")
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:
        elapsed_ms = int((time.monotonic() - t_start) * 1000)
        log_generation(meeting_id, content_type, provider_name, provider._model,
                       prompt_version, 0, 0, elapsed_ms, 0.0, "ERROR")
        return jsonify({"error": "An unexpected error occurred during content generation."}), 500

    elapsed_ms = int((time.monotonic() - t_start) * 1000)
    tokens_in = result.get("tokens_in", 0)
    tokens_out = result.get("tokens_out", 0)
    model = result.get("model", "unknown")
    raw_content = result.get("content", "")

    cost_usd = provider.calculate_cost(tokens_in, tokens_out)

    parsed_content = None
    if content_type in {"FLASHCARDS", "KNOWLEDGE", "MCQS"}:
        try:
            parsed_content = json.loads(raw_content)
        except json.JSONDecodeError:
            parsed_content = {"raw": raw_content}
    else:
        parsed_content = raw_content

    try:
        content_json = json.dumps(parsed_content) if not isinstance(parsed_content, str) else parsed_content
    except (TypeError, ValueError):
        content_json = json.dumps({"raw": str(parsed_content)})

    try:
        save_ai_content(meeting_id, content_type, content_json, provider_name, model, prompt_version)
        log_generation(meeting_id, content_type, provider_name, model, prompt_version,
                       tokens_in, tokens_out, elapsed_ms, cost_usd, "SUCCESS")
    except Exception as exc:
        logger.error("Failed to save AI content: %s", exc)

    return jsonify({
        "meetingId": meeting_id,
        "contentType": content_type,
        "content": parsed_content,
        "cached": False,
        "tokensIn": tokens_in,
        "tokensOut": tokens_out,
        "costUsd": cost_usd,
        "model": model,
    })


@app.route("/api/meeting/<int:meeting_id>/content/<content_type>", methods=["GET"])
def api_get_content(meeting_id, content_type):
    user = session.get("user")
    if not user:
        return jsonify({"error": "Authentication required."}), 401

    content_type = content_type.upper()

    meeting = get_meeting(meeting_id)
    if not meeting:
        return jsonify({"error": "Meeting not found."}), 404
    if meeting["user_id"] != user["id"]:
        return jsonify({"error": "Access denied."}), 403
    if meeting["deleted_at"] is not None:
        return jsonify({"error": "Meeting not found."}), 404

    content = get_ai_content(meeting_id, content_type)
    if not content:
        return jsonify({"error": "Content not found. Generate it first."}), 404

    try:
        parsed = json.loads(content["content_json"])
    except (json.JSONDecodeError, TypeError):
        parsed = content["content_json"]

    return jsonify({
        "meetingId": meeting_id,
        "contentType": content_type,
        "content": parsed,
        "cached": True,
        "version": content["version"],
        "createdAt": content["created_at"],
    })


import logging
logger = logging.getLogger(__name__)


# --------------------------
# Run App
# --------------------------

if __name__ == "__main__":
    app.run(debug=True)
