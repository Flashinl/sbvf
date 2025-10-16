from __future__ import annotations
import time
import hmac
import secrets
from typing import Optional

import jwt
from passlib.context import CryptContext
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

from .config import Settings

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    try:
        return pwd_context.verify(password, hashed)
    except Exception:
        return False


def _settings():
    return Settings.load()


def create_jwt(subject: str, expires_in: int = 3600) -> str:
    st = _settings()
    payload = {
        "sub": subject,
        "iat": int(time.time()),
        "exp": int(time.time()) + int(expires_in or 3600),
    }
    return jwt.encode(payload, st.jwt_secret or st.secret_key, algorithm="HS256")


def verify_jwt(token: str) -> Optional[dict]:
    st = _settings()
    try:
        return jwt.decode(token, st.jwt_secret or st.secret_key, algorithms=["HS256"])
    except Exception:
        return None


def signer() -> URLSafeTimedSerializer:
    st = _settings()
    return URLSafeTimedSerializer(st.secret_key)


def sign_token(data: str) -> str:
    return signer().dumps(data)


def unsign_token(token: str, max_age_seconds: int) -> Optional[str]:
    try:
        return signer().loads(token, max_age=max_age_seconds)
    except (BadSignature, SignatureExpired):
        return None


# CSRF utilities (double submit cookie pattern)

def generate_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def constant_time_compare(a: str, b: str) -> bool:
    return hmac.compare_digest(a or "", b or "")

