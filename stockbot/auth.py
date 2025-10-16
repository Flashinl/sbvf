from __future__ import annotations
from typing import Optional
from fastapi import APIRouter, Depends, Form, Request, HTTPException, status
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr, ValidationError
from starlette.middleware.sessions import SessionMiddleware

from .db import get_db, Base, engine
from .models import User
from .security import hash_password, verify_password, create_jwt, verify_jwt, generate_csrf_token, constant_time_compare, sign_token, unsign_token
from .config import Settings

# Ensure tables exist (simple bootstrap; migrations can be added later)
Base.metadata.create_all(bind=engine)

router = APIRouter(prefix="/auth", tags=["auth"])
templates = Jinja2Templates(directory="templates")


class RegisterForm(BaseModel):
    email: EmailStr
    password: str


class LoginForm(BaseModel):
    email: EmailStr
    password: str


def _get_client_ip(request: Request) -> str:
    return request.client.host if request and request.client else "unknown"

# Simple in-memory rate limiter (per-IP)
_RATE_LIMIT = {}
_RATE_LIMIT_MAX = 10
_RATE_LIMIT_WINDOW = 60


def rate_limit(request: Request):
    import time
    ip = _get_client_ip(request)
    now = time.time()
    window = _RATE_LIMIT.setdefault(ip, [])
    # drop old
    _RATE_LIMIT[ip] = [t for t in window if now - t < _RATE_LIMIT_WINDOW]
    if len(_RATE_LIMIT[ip]) >= _RATE_LIMIT_MAX:
        raise HTTPException(status_code=429, detail="Too many requests")
    _RATE_LIMIT[ip].append(now)


# Dependencies for session/JWT

def get_current_user(request: Request, db: Session = Depends(get_db)) -> Optional[User]:
    uid = request.session.get("uid") if hasattr(request, "session") else None
    if uid:
        return db.query(User).filter(User.id == uid, User.is_active == True).first()
    # Try Authorization: Bearer
    auth = request.headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
        payload = verify_jwt(token)
        if payload and payload.get("sub"):
            return db.query(User).filter(User.email == payload["sub"], User.is_active == True).first()
    return None


def require_user(user: Optional[User] = Depends(get_current_user)) -> User:
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


@router.get("/login")
async def login_get(request: Request):
    csrf = generate_csrf_token()
    request.session["csrf"] = csrf
    return templates.TemplateResponse("login.html", {"request": request, "csrf": csrf})


@router.post("/login")
async def login_post(request: Request, email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    rate_limit(request)
    try:
        data = LoginForm(email=email, password=password)
    except ValidationError:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid input"}, status_code=400)

    # CSRF
    form_csrf = request.form()._dict.get("csrf") if hasattr(request, "form") else None
    if not constant_time_compare(request.session.get("csrf"), (await request.form()).get("csrf")):
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid CSRF"}, status_code=400)

    user = db.query(User).filter(User.email == data.email).first()
    if not user or not verify_password(data.password, user.password_hash):
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"}, status_code=400)
    if not user.is_active:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Account disabled"}, status_code=403)

    request.session["uid"] = user.id
    return RedirectResponse(url="/", status_code=303)


@router.get("/register")
async def register_get(request: Request):
    csrf = generate_csrf_token()
    request.session["csrf"] = csrf
    return templates.TemplateResponse("register.html", {"request": request, "csrf": csrf})


@router.post("/register")
async def register_post(request: Request, email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    rate_limit(request)
    try:
        data = RegisterForm(email=email, password=password)
    except ValidationError:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Invalid input"}, status_code=400)

    form = await request.form()
    if not constant_time_compare(request.session.get("csrf"), form.get("csrf")):
        return templates.TemplateResponse("register.html", {"request": request, "error": "Invalid CSRF"}, status_code=400)

    existing = db.query(User).filter(User.email == data.email).first()
    if existing:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Email already registered"}, status_code=400)

    user = User(email=data.email, password_hash=hash_password(data.password), is_active=True, is_verified=False)
    db.add(user)
    db.commit()
    db.refresh(user)

    # Email verify link (console)
    token = sign_token(user.email)
    print(f"[DEV] Email verification link: http://localhost:8000/auth/verify?token={token}")

    request.session["uid"] = user.id
    return RedirectResponse(url="/", status_code=303)


@router.get("/verify")
async def verify_email(token: str, db: Session = Depends(get_db)):
    email = unsign_token(token, max_age_seconds=60 * 60 * 24)
    if not email:
        return JSONResponse(status_code=400, content={"error": "Invalid/expired token"})
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return JSONResponse(status_code=404, content={"error": "User not found"})
    user.is_verified = True
    db.add(user)
    db.commit()
    return RedirectResponse(url="/", status_code=303)


@router.post("/logout")
async def logout(request: Request):
    request.session.pop("uid", None)
    return RedirectResponse(url="/", status_code=303)


# Password reset flow
@router.get("/reset-request")
async def reset_request_get(request: Request):
    csrf = generate_csrf_token()
    request.session["csrf"] = csrf
    return templates.TemplateResponse("reset_request.html", {"request": request, "csrf": csrf})


@router.post("/reset-request")
async def reset_request_post(request: Request, email: str = Form(...), db: Session = Depends(get_db)):
    rate_limit(request)
    form = await request.form()
    if not constant_time_compare(request.session.get("csrf"), form.get("csrf")):
        return templates.TemplateResponse("reset_request.html", {"request": request, "error": "Invalid CSRF"}, status_code=400)

    user = db.query(User).filter(User.email == email).first()
    if user:
        token = sign_token(user.email)
        print(f"[DEV] Password reset link: http://localhost:8000/auth/reset?token={token}")
    return templates.TemplateResponse("reset_request.html", {"request": request, "info": "If that email exists, a reset link was generated (see server logs)."})


@router.get("/reset")
async def reset_get(request: Request, token: str):
    csrf = generate_csrf_token()
    request.session["csrf"] = csrf
    return templates.TemplateResponse("reset_confirm.html", {"request": request, "csrf": csrf, "token": token})


@router.post("/reset")
async def reset_post(request: Request, token: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    rate_limit(request)
    form = await request.form()
    if not constant_time_compare(request.session.get("csrf"), form.get("csrf")):
        return templates.TemplateResponse("reset_confirm.html", {"request": request, "error": "Invalid CSRF", "token": token}, status_code=400)

    email = unsign_token(token, max_age_seconds=60 * 60 * 24)
    if not email:
        return templates.TemplateResponse("reset_confirm.html", {"request": request, "error": "Invalid/expired token", "token": token}, status_code=400)

    user = db.query(User).filter(User.email == email).first()
    if not user:
        return templates.TemplateResponse("reset_confirm.html", {"request": request, "error": "User not found", "token": token}, status_code=404)

    user.password_hash = hash_password(password)
    db.add(user)
    db.commit()
    return RedirectResponse(url="/auth/login", status_code=303)


# JWT token for API clients
@router.post("/token")
async def token_api(email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    token = create_jwt(subject=user.email)
    return {"access_token": token, "token_type": "bearer"}

