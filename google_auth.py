import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from authlib.integrations.flask_client import OAuth
from flask import Blueprint, redirect, session, url_for


google_auth = Blueprint("google_auth", __name__)

oauth = OAuth()

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "users.db"


def init_oauth(app):
    oauth.init_app(app)

    oauth.register(
        name="google",
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )


def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            google_id TEXT UNIQUE,
            email TEXT UNIQUE,
            name TEXT,
            picture TEXT,
            provider TEXT DEFAULT 'google',
            created_at TEXT NOT NULL,
            last_login TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS meetings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL DEFAULT '',
            file_name TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            file_type TEXT NOT NULL,
            file_duration TEXT,
            transcript TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            archived_at TEXT,
            deleted_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS meeting_ai_content (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            meeting_id INTEGER NOT NULL,
            content_type TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            content_json TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (meeting_id) REFERENCES meetings(id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS generation_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            meeting_id INTEGER NOT NULL,
            content_type TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            tokens_in INTEGER NOT NULL,
            tokens_out INTEGER NOT NULL,
            latency_ms INTEGER NOT NULL,
            cost_usd REAL NOT NULL,
            status TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (meeting_id) REFERENCES meetings(id)
        )
        """
    )
    conn.commit()
    conn.close()


def upsert_user(google_id, email, name, picture):
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    conn = get_db_connection()

    existing_user = conn.execute(
        "SELECT id FROM users WHERE google_id = ? OR email = ?",
        (google_id, email),
    ).fetchone()

    if existing_user:
        user_id = existing_user["id"]
        conn.execute(
            """
            UPDATE users
            SET email = ?, name = ?, picture = ?, provider = 'google', last_login = ?
            WHERE id = ?
            """,
            (email, name, picture, now, user_id),
        )
    else:
        cursor = conn.execute(
            """
            INSERT INTO users (google_id, email, name, picture, provider, created_at, last_login)
            VALUES (?, ?, ?, ?, 'google', ?, ?)
            """,
            (google_id, email, name, picture, now, now),
        )
        user_id = cursor.lastrowid

    conn.commit()
    conn.close()
    return user_id


@google_auth.route("/login/google")
def login():
    redirect_uri = url_for("google_auth.callback", _external=True)
    return oauth.google.authorize_redirect(redirect_uri)


@google_auth.route("/auth/google/callback")
def callback():
    token = oauth.google.authorize_access_token()
    userinfo_response = oauth.google.get(
        "https://www.googleapis.com/oauth2/v3/userinfo",
        token=token,
    )
    userinfo_response.raise_for_status()
    userinfo = userinfo_response.json()

    google_id = userinfo.get("sub")
    email = userinfo.get("email")
    name = userinfo.get("name")
    picture = userinfo.get("picture", "")

    if not google_id or not email:
        return redirect("/")

    user_id = upsert_user(google_id, email, name, picture)

    session.clear()
    session["user"] = {
        "id": user_id,
        "name": name,
        "email": email,
        "picture": picture,
    }
    session.permanent = True

    return redirect("/dashboard")