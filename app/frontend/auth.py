"""
Google OAuth authentication gate for the Quality Copilot Streamlit app.

Flow:
  1. Unauthenticated  → login page with "Sign in with Google" button
  2. Authenticated but not on allowlist → Access Denied + logout
  3. Authenticated and on allowlist → user card in sidebar, returns True

Required environment variables (set in .env):
  GOOGLE_CLIENT_ID
  GOOGLE_CLIENT_SECRET
  AUTH_COOKIE_KEY       (any random secret string, keep it stable across restarts)
  REDIRECT_URI          (defaults to http://localhost:8501)
  ALLOWED_USERS_PATH    (defaults to config/allowed_users.txt)
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Set

import streamlit as st

_ROOT = Path(__file__).resolve().parents[3]
_ALLOWED_USERS_PATH = Path(
    os.getenv("ALLOWED_USERS_PATH", str(_ROOT / "config" / "allowed_users.txt"))
)
_REDIRECT_URI = os.getenv("REDIRECT_URI", "http://localhost:8501")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_allowed_emails() -> Set[str]:
    if not _ALLOWED_USERS_PATH.exists():
        return set()
    emails: Set[str] = set()
    for line in _ALLOWED_USERS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            emails.add(line.lower())
    return emails


def _credentials_json() -> dict:
    """Build the Google OAuth client-secrets structure from env vars."""
    return {
        "web": {
            "client_id":     os.getenv("GOOGLE_CLIENT_ID", ""),
            "client_secret": os.getenv("GOOGLE_CLIENT_SECRET", ""),
            "redirect_uris": [_REDIRECT_URI],
            "auth_uri":      "https://accounts.google.com/o/oauth2/auth",
            "token_uri":     "https://oauth2.googleapis.com/token",
        }
    }


def _make_authenticator():
    """Instantiate the streamlit-google-auth Authenticate object."""
    from streamlit_google_auth import Authenticate  # pip install streamlit-google-auth

    creds = _credentials_json()
    # streamlit-google-auth requires a file path, so we write to a named temp file.
    # The file is only on disk for the duration of this call.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir=tempfile.gettempdir()
    ) as fh:
        json.dump(creds, fh)
        creds_path = fh.name

    cookie_key = os.getenv("AUTH_COOKIE_KEY", "quality_copilot_change_me_in_production")

    return Authenticate(
        secret_credentials_path=creds_path,
        cookie_name="quality_copilot_auth",
        cookie_key=cookie_key,
        redirect_uri=_REDIRECT_URI,
    )


# ── UI renderers ──────────────────────────────────────────────────────────────

def _render_login_page(authenticator) -> None:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] { display: none !important; }
        .main .block-container { max-width: 480px; margin: 0 auto; padding-top: 6rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="text-align:center;margin-bottom:2rem;">
          <h2 style="color:#E6EDF3;margin-bottom:0.25rem;">🔍 Quality Copilot</h2>
          <p style="color:#8B949E;font-size:0.9rem;">
            Sign in with your authorised Google account to continue.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    authorization_url = authenticator.get_authorization_url()
    st.link_button(
        "Sign in with Google",
        authorization_url,
        use_container_width=True,
    )


def _render_access_denied(email: str) -> None:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] { display: none !important; }
        .main .block-container { max-width: 480px; margin: 0 auto; padding-top: 6rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div style="text-align:center;margin-bottom:2rem;">
          <h2 style="color:#F85149;">⛔ Access Denied</h2>
          <p style="color:#8B949E;font-size:0.9rem;">
            <strong style="color:#E6EDF3;">{email}</strong> is not on the approved access list.<br>
            Contact your administrator to request access.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("← Logout and try a different account", use_container_width=True):
        for key in ("connected", "user_info", "oauth_state"):
            st.session_state.pop(key, None)
        st.rerun()


def _render_user_sidebar(user_info: dict) -> None:
    name    = user_info.get("name", "User")
    email   = user_info.get("email", "")
    picture = user_info.get("picture", "")

    avatar_html = (
        f"<img src='{picture}' style='width:32px;height:32px;border-radius:50%;"
        f"object-fit:cover;flex-shrink:0;'>"
        if picture else
        "<div style='width:32px;height:32px;border-radius:50%;background:#388BFD;"
        "display:flex;align-items:center;justify-content:center;font-size:1rem;"
        "flex-shrink:0;'>👤</div>"
    )

    with st.sidebar:
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:0.6rem;
                        padding:0.55rem 0.65rem;background:#21262D;
                        border:1px solid #30363D;border-radius:8px;
                        margin-bottom:0.75rem;">
              {avatar_html}
              <div style="overflow:hidden;">
                <div style="font-size:0.8rem;font-weight:600;color:#E6EDF3;
                            white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                  {name}
                </div>
                <div style="font-size:0.68rem;color:#8B949E;
                            white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                  {email}
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Logout", key="auth_logout_btn", use_container_width=True):
            for key in ("connected", "user_info", "oauth_state"):
                st.session_state.pop(key, None)
            st.rerun()


# ── Public entry point ────────────────────────────────────────────────────────

def check_auth() -> bool:
    """
    Run the full auth gate. Returns True only when the user is both
    authenticated with Google and present in the allowlist.

    Renders its own UI (login page, access-denied page, or sidebar user card)
    as a side-effect. Call st.stop() after a False return to halt rendering.
    """
    client_id     = os.getenv("GOOGLE_CLIENT_ID", "")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "")

    if not client_id or not client_secret:
        st.error(
            "Google OAuth is not configured. "
            "Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in your .env file.",
            icon="🔑",
        )
        st.stop()

    authenticator = _make_authenticator()
    authenticator.check_authentification()

    if not st.session_state.get("connected"):
        _render_login_page(authenticator)
        return False

    user_info = st.session_state.get("user_info", {})
    email     = (user_info.get("email") or "").lower()
    allowed   = _load_allowed_emails()

    if email not in allowed:
        _render_access_denied(email)
        return False

    _render_user_sidebar(user_info)
    return True
