from flask import Blueprint, redirect, url_for, session
from authlib.integrations.flask_client import OAuth
import os

google_auth = Blueprint("google_auth", __name__)

oauth = OAuth()


def init_oauth(app):
    oauth.init_app(app)

    oauth.register(
        name="google",
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={
            "scope": "openid email profile"
        }
    )


@google_auth.route("/login/google")
def login():
    redirect_uri = url_for(
        "google_auth.callback",
        _external=True
    )

    return oauth.google.authorize_redirect(redirect_uri)


@google_auth.route("/auth/google/callback")
def callback():
    token = oauth.google.authorize_access_token()

    user = token["userinfo"]

    session["user"] = {
        "id": user["sub"],
        "name": user["name"],
        "email": user["email"],
        "picture": user["picture"]
    }

    return redirect("/dashboard")