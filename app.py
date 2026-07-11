import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from functools import wraps

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, session

from google_auth import google_auth, get_db_connection, init_db, init_oauth

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-secret")
app.config["SESSION_PERMANENT"] = True
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=30)

init_oauth(app)
init_db()
app.register_blueprint(google_auth)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".webm", ".ogg", ".flac"}
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB

VALID_CONTENT_TYPES = {"NOTES", "FLASHCARDS", "KNOWLEDGE", "MCQS", "REVISION"}


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user"):
            return jsonify({"error": "Authentication required."}), 401
        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# API: List meetings
# ---------------------------------------------------------------------------

@app.route("/api/meetings")
@login_required
def api_list_meetings():
    user = session["user"]
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT id, title, file_name, file_size, file_type, file_duration,
               status, created_at
        FROM meetings
        WHERE user_id = ? AND deleted_at IS NULL
        ORDER BY created_at DESC
        """,
        (user["id"],),
    ).fetchall()
    conn.close()

    meetings = []
    for r in rows:
        meetings.append({
            "meetingId": r["id"],
            "title": r["title"],
            "fileName": r["file_name"],
            "fileSize": r["file_size"],
            "fileType": r["file_type"],
            "fileDuration": r["file_duration"],
            "status": r["status"],
            "createdAt": r["created_at"],
        })

    return jsonify(meetings)


# ---------------------------------------------------------------------------
# API: Get single meeting
# ---------------------------------------------------------------------------

@app.route("/api/meeting/<int:meeting_id>")
@login_required
def api_get_meeting(meeting_id):
    user = session["user"]
    conn = get_db_connection()
    row = conn.execute(
        """
        SELECT id, title, file_name, file_size, file_type, file_duration,
               transcript, status, created_at
        FROM meetings
        WHERE id = ? AND user_id = ? AND deleted_at IS NULL
        """,
        (meeting_id, user["id"]),
    ).fetchone()
    conn.close()

    if not row:
        return jsonify({"error": "Meeting not found."}), 404

    return jsonify({
        "meetingId": row["id"],
        "title": row["title"],
        "fileName": row["file_name"],
        "fileSize": row["file_size"],
        "fileType": row["file_type"],
        "fileDuration": row["file_duration"],
        "transcript": row["transcript"],
        "status": row["status"],
        "createdAt": row["created_at"],
    })


# ---------------------------------------------------------------------------
# API: Transcribe audio
# ---------------------------------------------------------------------------

@app.route("/api/transcribe", methods=["POST"])
@login_required
def api_transcribe():
    user = session["user"]

    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided."}), 400

    audio_file = request.files["audio"]
    if not audio_file.filename:
        return jsonify({"error": "No audio file selected."}), 400

    ext = "." + audio_file.filename.rsplit(".", 1)[-1].lower() if "." in audio_file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file format '{ext}'. Supported: mp3, wav, m4a, aac, webm, ogg, flac."}), 400

    file_data = audio_file.read()
    if len(file_data) > MAX_UPLOAD_SIZE:
        return jsonify({"error": "File too large. Maximum size is 100 MB."}), 400

    safe_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOAD_DIR, safe_name)
    with open(file_path, "wb") as f:
        f.write(file_data)

    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    duration = request.form.get("duration", "")

    conn = get_db_connection()
    cursor = conn.execute(
        """
        INSERT INTO meetings (user_id, title, file_name, file_size, file_type, file_duration, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, 'processing', ?, ?)
        """,
        (user["id"], audio_file.filename, audio_file.filename, len(file_data), ext.lstrip(".").upper(), duration, now, now),
    )
    meeting_id = cursor.lastrowid
    conn.commit()
    conn.close()

    try:
        from transcription_service import get_transcription_provider
        provider = get_transcription_provider()
        transcript = provider.transcribe(file_path, audio_file.filename)
    except Exception as exc:
        logger.error("Transcription failed for meeting %d: %s", meeting_id, exc)
        conn = get_db_connection()
        conn.execute(
            "UPDATE meetings SET status = 'failed', updated_at = ? WHERE id = ?",
            (now, meeting_id),
        )
        conn.commit()
        conn.close()
        return jsonify({"error": str(exc)}), 500
    finally:
        try:
            os.remove(file_path)
        except OSError:
            pass

    conn = get_db_connection()
    conn.execute(
        "UPDATE meetings SET transcript = ?, status = 'completed', updated_at = ? WHERE id = ?",
        (transcript, now, meeting_id),
    )
    conn.commit()
    conn.close()

    return jsonify({
        "meetingId": meeting_id,
        "transcript": transcript,
    })


# ---------------------------------------------------------------------------
# API: Generate AI content
# ---------------------------------------------------------------------------

@app.route("/api/meeting/<int:meeting_id>/generate/<content_type>", methods=["POST"])
@login_required
def api_generate_content(meeting_id, content_type):
    content_type = content_type.upper()

    if content_type not in VALID_CONTENT_TYPES:
        return jsonify({"error": f"Invalid content type '{content_type}'. Valid: {', '.join(sorted(VALID_CONTENT_TYPES))}"}), 400

    user = session["user"]
    conn = get_db_connection()
    meeting = conn.execute(
        """
        SELECT id, transcript, status
        FROM meetings
        WHERE id = ? AND user_id = ? AND deleted_at IS NULL
        """,
        (meeting_id, user["id"]),
    ).fetchone()

    if not meeting:
        conn.close()
        return jsonify({"error": "Meeting not found."}), 404

    transcript = meeting["transcript"]
    if not transcript:
        conn.close()
        return jsonify({"error": "No transcript available. Please transcribe the audio first."}), 400

    existing = conn.execute(
        """
        SELECT content_json, provider, model
        FROM meeting_ai_content
        WHERE meeting_id = ? AND content_type = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (meeting_id, content_type),
    ).fetchone()

    if existing:
        conn.close()
        try:
            content = json.loads(existing["content_json"])
        except (json.JSONDecodeError, TypeError):
            content = existing["content_json"]
        return jsonify({
            "content": content,
            "cached": True,
            "provider": existing["provider"],
            "model": existing["model"],
        })

    conn.close()

    try:
        from ai_content_service import PROMPT_REGISTRY, get_ai_provider, save_ai_content, log_generation
    except Exception as exc:
        logger.error("Failed to import AI content service: %s", exc)
        return jsonify({"error": "AI content service is not available."}), 500

    prompt_entry = PROMPT_REGISTRY.get(content_type)
    if not prompt_entry:
        return jsonify({"error": f"No prompt registered for '{content_type}'."}), 400

    prompt_text = prompt_entry["prompt_template"].replace("{transcript}", transcript)
    system_message = prompt_entry.get("system_message", "")
    prompt_version = prompt_entry.get("version", "1.0")

    try:
        provider = get_ai_provider()
    except Exception as exc:
        logger.error("Failed to initialise AI provider: %s", exc)
        return jsonify({"error": "AI provider is not configured."}), 500

    t_start = time.monotonic()

    try:
        result = provider.generate(prompt=prompt_text, system_message=system_message)
    except Exception as exc:
        elapsed_ms = int((time.monotonic() - t_start) * 1000)
        logger.error("AI generation failed: meeting=%d type=%s error=%s", meeting_id, content_type, exc)
        try:
            from ai_content_service import log_generation as _log
            _log(
                meeting_id=meeting_id,
                content_type=content_type,
                provider=type(provider).__name__,
                model=getattr(provider, "_model", "unknown"),
                prompt_version=prompt_version,
                tokens_in=0,
                tokens_out=0,
                latency_ms=elapsed_ms,
                cost_usd=0.0,
                status="error",
            )
        except Exception:
            pass
        return jsonify({"error": f"AI generation failed: {str(exc)}"}), 500

    elapsed_ms = int((time.monotonic() - t_start) * 1000)
    raw_content = result.get("content", "")
    tokens_in = result.get("tokens_in", 0)
    tokens_out = result.get("tokens_out", 0)
    model_name = result.get("model", "unknown")
    provider_name = type(provider).__name__

    if content_type in ("FLASHCARDS", "KNOWLEDGE", "MCQS"):
        try:
            parsed = json.loads(raw_content)
        except (json.JSONDecodeError, TypeError):
            parsed = raw_content
        content_to_store = json.dumps(parsed) if not isinstance(parsed, str) else parsed
    else:
        content_to_store = raw_content
        parsed = raw_content

    try:
        cost = provider.calculate_cost(tokens_in, tokens_out)
    except Exception:
        cost = 0.0

    try:
        save_ai_content(
            meeting_id=meeting_id,
            content_type=content_type,
            content_json=content_to_store,
            provider=provider_name,
            model=model_name,
            prompt_version=prompt_version,
        )
        log_generation(
            meeting_id=meeting_id,
            content_type=content_type,
            provider=provider_name,
            model=model_name,
            prompt_version=prompt_version,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=elapsed_ms,
            cost_usd=cost,
            status="success",
        )
    except Exception as exc:
        logger.warning("Failed to save AI content or log: %s", exc)

    return jsonify({
        "content": parsed,
        "cached": False,
        "provider": provider_name,
        "model": model_name,
    })


# ---------------------------------------------------------------------------
# Run App
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
