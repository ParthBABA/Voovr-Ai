from flask import Flask, render_template, session, redirect
from dotenv import load_dotenv
from auth.google_auth import google_auth, init_oauth
import os

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY")

init_oauth(app)

app.register_blueprint(google_auth)

# --------------------------
# Routes
# --------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")

    user = session["user"]

    return f"""
    <h1>Welcome {user['name']}</h1>
    <h3>{user['email']}</h3>
    <img src="{user['picture']}" width="120">
    <br><br>
    <a href="/logout">Logout</a>
    """


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


# --------------------------
# Run App
# --------------------------

if __name__ == "__main__":
    app.run(debug=True)