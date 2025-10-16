from __future__ import annotations
import json
import time
from typing import Optional
from urllib import request, error

from .config import Settings

HF_API_BASE = "https://api-inference.huggingface.co/models"


def _post_json(url: str, payload: dict, headers: dict, timeout: int) -> Optional[dict]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
            try:
                return json.loads(raw)
            except Exception:
                return None
    except error.HTTPError as e:
        # Some HF models return 503 while spinning up; brief backoff
        if e.code in (503, 524):
            time.sleep(1.2)
            try:
                with request.urlopen(req, timeout=timeout) as resp:
                    raw = resp.read().decode("utf-8", errors="ignore")
                    return json.loads(raw)
            except Exception:
                return None
        return None
    except Exception:
        return None


def _clean(text: str) -> str:
    t = (text or "").strip()
    # Keep it concise and avoid numeric clutter from models that echo numbers
    # Remove trailing repeated whitespace and collapse spaces
    t = " ".join(t.split())
    return t


def generate_narrative(
    *,
    about: str,
    why_moving: str,
    drivers_line: str,
    timing_msgs: list[str],
    why_continue: str,
    watch_list: list[str],
    bull_summary: str,
    bear_summary: str,
    cond_up_txt: str,
    invalid_txt: str,
    sources: str,
    label: str,
    settings: Settings,
) -> Optional[str]:
    """
    Calls a free Hugging Face Inference API model (default: google/flan-t5-base)
    to produce a smart, qualitative narrative. No extra packages required.

    Returns a single-paragraph narrative or None on failure.
    """
    if not settings or not getattr(settings, "llm_enabled", False):
        return None
    token = getattr(settings, "hf_api_token", None)
    model = getattr(settings, "llm_model", "google/flan-t5-base")
    if not token or not model:
        return None

    # Build instruction and context; avoid pushing lots of numbers into the model
    ctx = []
    if about:
        ctx.append(f"About: {about}")
    if why_moving:
        ctx.append(f"Why it's moving: {why_moving}")
    if drivers_line:
        ctx.append(f"Specific drivers (from multiple sources): {drivers_line}")
    if timing_msgs:
        ctx.append(f"Timing: {'; '.join(timing_msgs)}")
    if why_continue:
        ctx.append(f"Why it can continue: {why_continue}")
    if watch_list:
        ctx.append(f"What the stock needs: {', '.join(watch_list[:4])}")
    if bull_summary:
        ctx.append(f"Positives from headlines: {bull_summary}")
    if bear_summary:
        ctx.append(f"Risks from headlines: {bear_summary}")
    if cond_up_txt:
        ctx.append(f"To go higher: {cond_up_txt}")
    if invalid_txt:
        ctx.append(f"What could go wrong: {invalid_txt}")
    if sources:
        ctx.append(f"Sources: {sources}")

    instruction = (
        "Write a concise, human, qualitative analysis for a stock. Avoid numeric precision, percentages, and price figures in the narrative (prices are shown elsewhere). "
        "Use plain language. Capture business context and multi-source drivers. Keep sections in this order and tone: About the company; Why it's moving; Specific drivers; Timing; Why it can continue; What the stock needs; Positives; Risks; To go higher; What could go wrong; Bottom line. "
        "For Bottom line, output only the label (Buy/Hold/Sell) with no confidence number."
    )

    prompt = instruction + "\n\n" + "\n".join(ctx)

    # Trim prompt to a safe size
    max_chars = int(getattr(settings, "llm_max_input_chars", 4000) or 4000)
    if len(prompt) > max_chars:
        prompt = prompt[-max_chars:]

    url = f"{HF_API_BASE}/{model}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    # Try text2text and text-generation formats gracefully.
    payload_t2t = {"inputs": prompt, "parameters": {"max_new_tokens": 220, "temperature": 0.4}}
    res = _post_json(url, payload_t2t, headers, int(getattr(settings, "llm_timeout_seconds", 18) or 18))
    text = None
    if isinstance(res, list) and res and isinstance(res[0], dict) and "generated_text" in res[0]:
        text = res[0]["generated_text"]
    elif isinstance(res, dict) and "generated_text" in res:
        text = res["generated_text"]
    elif isinstance(res, list) and res and isinstance(res[0], dict) and "summary_text" in res[0]:
        text = res[0]["summary_text"]

    if not text:
        # Fallback to text-generation
        payload_tg = {"inputs": prompt, "parameters": {"max_new_tokens": 220, "do_sample": False, "temperature": 0.4}}
        res2 = _post_json(url, payload_tg, headers, int(getattr(settings, "llm_timeout_seconds", 18) or 18))
        if isinstance(res2, list) and res2 and isinstance(res2[0], dict) and "generated_text" in res2[0]:
            text = res2[0]["generated_text"]
        elif isinstance(res2, dict) and "generated_text" in res2:
            text = res2["generated_text"]

    return _clean(text) if text else None

