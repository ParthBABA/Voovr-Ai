from datetime import timedelta
import os

from dotenv import load_dotenv
from flask import Flask, redirect, render_template, session

from google_auth import google_auth, init_db, init_oauth

load_dotenv()


def transcribe_audio(file_path):
    import os
    from groq import Groq

    client = Groq()
    filename = os.path.dirname(__file__) + "/audio.m4a"

    with open(filename, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(filename, file.read()),
            model="whisper-large-v3-turbo",
            temperature=0,
            response_format="verbose_json",
        )
        print(transcription.text)

    return transcription.text


app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-secret")
app.config["SESSION_PERMANENT"] = True
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=30)

init_oauth(app)
init_db()
app.register_blueprint(google_auth)


# --------------------------
# Routes
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
# Run App
# --------------------------

if __name__ == "__main__":
    app.run(debug=True)