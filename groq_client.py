"""
Shared Groq client for all LLM text generation.
Gemini is kept separately for embeddings only (groq has no embedding API).

The model is configurable with the GROQ_MODEL environment variable. Groq retires models
from time to time (llama-3.3-70b-versatile was retired, which silently broke search
parsing, the onboarding quiz analysis, the improvement bot and chat analysis), so if the
configured model is rejected we automatically try the fallbacks below and remember the
first one that works.
"""
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]

# Kept for backwards compatibility (some scripts read/override it)
GROQ_MODEL = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)

_client = None
_working_model = None


def get_groq_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return _client


def _is_model_error(exc) -> bool:
    msg = str(exc).lower()
    return ('model' in msg and ('not found' in msg or 'does not exist' in msg
                                or 'decommissioned' in msg or 'model_not_found' in msg))


def _candidate_models():
    seen, out = set(), []
    for m in [_working_model, GROQ_MODEL] + FALLBACK_MODELS:
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _is_rate_limit(exc) -> bool:
    msg = str(exc).lower()
    return (getattr(exc, 'status_code', None) == 429 or 'rate limit' in msg
            or 'rate_limit' in msg or 'too many requests' in msg or 'error code: 429' in msg)


def _create_with_retry(client, **kwargs):
    """Groq's free tier rate-limits bursts. Wait briefly and retry instead of failing
    (a failure made the onboarding quiz silently save a canned profile)."""
    import time
    delay = 2.0
    for attempt in range(4):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            if not _is_rate_limit(e) or attempt == 3:
                raise
            retry_after = None
            try:
                retry_after = float(e.response.headers.get('retry-after'))
            except Exception:
                pass
            time.sleep(min(retry_after or delay, 10.0))
            delay *= 2


def _complete(messages, temperature=0.7) -> str:
    global _working_model
    client = get_groq_client()
    last_exc = None
    for model in _candidate_models():
        try:
            response = _create_with_retry(
                client,
                model=model,
                messages=messages,
                temperature=temperature,
            )
            text = (response.choices[0].message.content or '').strip()
            if not text and 'gpt-oss' in model:
                # Reasoning models sometimes spend the whole output on hidden reasoning and
                # return empty text (seen in production for the onboarding quiz). Retry once
                # with low reasoning effort so the answer itself gets the tokens.
                try:
                    response = _create_with_retry(client, model=model, messages=messages,
                                                  temperature=temperature, reasoning_effort='low')
                    text = (response.choices[0].message.content or '').strip()
                except Exception as e:
                    print(f"groq_client: low-effort retry on '{model}' failed: {str(e)[:120]}")
            if not text:
                print(f"groq_client: empty response from '{model}'; trying next model")
                last_exc = RuntimeError(f"empty response from {model}")
                continue
            if _working_model != model:
                if _working_model is not None or model != GROQ_MODEL:
                    print(f"groq_client: using model '{model}'")
                _working_model = model
            return text
        except Exception as e:
            last_exc = e
            if _is_model_error(e):
                print(f"groq_client: model '{model}' unavailable ({str(e)[:120]}); trying next")
                continue
            raise  # not a model problem (rate limit, network, auth) — let the caller decide
    raise last_exc if last_exc else RuntimeError("No Groq model available")


def groq_generate(prompt: str, system: str = None, temperature: float = 0.7) -> str:
    """
    Send a single-turn prompt to Groq and return the response text.
    Raises on error so callers can handle fallbacks themselves.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return _complete(messages, temperature=temperature)


def groq_chat(messages: list) -> str:
    """
    Send a multi-turn chat messages list to Groq and return the response text.
    Each message must be {"role": "user"|"assistant"|"system", "content": "..."}
    """
    return _complete(messages)
