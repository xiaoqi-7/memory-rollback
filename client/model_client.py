import os
import math
import argparse
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import openai

_ENV_LOADED = False


@dataclass(frozen=True)
class LLMTokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class LLMResult:
    text: str
    usage: Optional[LLMTokenUsage]


@dataclass(frozen=True)
class LLMClientSettings:
    profile: str
    api_type: str
    api_key: str
    model: str
    base_url: Optional[str]


_DEFAULT_ANTHROPIC_VERSION = "2023-06-01"


_LLM_PROFILES: Dict[str, Dict[str, object]] = {
    "gpt-4o": {
        "aliases": ("gpt4o", "openai", "openai:gpt-4o"),
        "api_type": "openai_compatible",
        "api_key_envs": ("OPENAI_API_KEY",),
        "model_env": "OPENAI_MODEL",
        "model": "gpt-4o",
        "base_url": None,
    },
    "gemini-3.6-flash": {
        "aliases": ("gemini", "gemini-flash", "gemini3.6-flash", "gemini-3.6"),
        "api_type": "openai_compatible",
        "api_key_envs": ("GEMINI_API_KEY",),
        "model_env": "GEMINI_MODEL",
        "model": "gemini-3.6-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
    "claude-opus-4.8": {
        "aliases": ("claude", "claude-opus", "opus-4.8", "claude4.8-opus"),
        "api_type": "anthropic_messages",
        "api_key_envs": ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"),
        "model_env": "ANTHROPIC_MODEL",
        "model": "claude-opus-4.8",
        "base_url": "https://api.anthropic.com/v1/messages",
    },
    "qwen3.6-27b": {
        "aliases": ("qwen", "qwen3.6", "qwen-3.6-27b", "qwen3.6-27B"),
        "api_type": "openai_compatible",
        "api_key_envs": ("QWEN_API_KEY", "DASHSCOPE_API_KEY"),
        "model_env": "QWEN_MODEL",
        "model": "Qwen3.6-27B",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
}

_LLM_PROFILE_ALIASES = {
    alias.lower(): name
    for name, profile in _LLM_PROFILES.items()
    for alias in (name, *profile.get("aliases", ()))
}


def _normalize_profile_key(value: str) -> str:
    return "-".join(value.strip().lower().replace("_", "-").split())

def _usage_field(usage: object, primary: str, fallback: Optional[str] = None) -> object:
    """Read one usage field from an SDK object or a plain mapping."""
    if isinstance(usage, dict):
        value = usage.get(primary)
        if value is None and fallback is not None:
            value = usage.get(fallback)
        return value
    value = getattr(usage, primary, None)
    if value is None and fallback is not None:
        value = getattr(usage, fallback, None)
    return value


def _normalize_llm_usage(usage: object) -> Optional[LLMTokenUsage]:
    input_tokens = _usage_field(usage, "prompt_tokens", "input_tokens")
    output_tokens = _usage_field(usage, "completion_tokens", "output_tokens")
    total_tokens = _usage_field(usage, "total_tokens")
    values = (input_tokens, output_tokens)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        return None
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens
    if isinstance(total_tokens, bool) or not isinstance(total_tokens, int) or total_tokens < 0:
        return None
    if input_tokens + output_tokens != total_tokens:
        return None
    return LLMTokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def load_env(env_path: Optional[str] = None) -> None:
    """Load environment variables from a .env file located at the repo root."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    env_file = Path(env_path) if env_path else Path(__file__).resolve().parent.parent / ".env"
    if not env_file.exists():
        _ENV_LOADED = True
        return

    with env_file.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            raw_line = line.strip()
            if not raw_line or raw_line.startswith("#"):
                continue

            if raw_line.startswith("export "):
                raw_line = raw_line[len("export "):]

            if "=" not in raw_line:
                continue

            key, value = raw_line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            if key not in os.environ:
                os.environ[key] = value

    _ENV_LOADED = True


def get_api_key(env_var: str = "OPENAI_API_KEY", default: Optional[str] = None) -> str:
    """Return a required API key from the environment."""
    load_env()
    value = os.getenv(env_var, default)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable '{env_var}'. "
            "Set it in your shell or in a .env file at the repository root."
        )
    return value


def get_openai_api_key() -> str:
    return get_api_key("OPENAI_API_KEY")


def _first_env(*names: str) -> Optional[str]:
    load_env()
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def get_qwen_api_key() -> str:
    """Return the API key used for Qwen/DashScope embedding calls."""
    load_env()
    return os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or get_api_key("QWEN_API_KEY")


def _normalize_embedding(vector: Sequence[float]) -> List[float]:
    """Normalize an embedding so dot product can be used as cosine similarity."""
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return [float(value) for value in vector]
    return [float(value) / norm for value in vector]


def _get_llm_config() -> dict:
    """Read LLM hyperparameters from environment variables.

    Supported env vars:
        LLM_TEMPERATURE  - float, controls randomness (e.g. 0.0 - 2.0)
        LLM_SEED         - int, for mostly-deterministic outputs
    """
    load_env()
    config = {}

    temperature = os.getenv("LLM_TEMPERATURE")
    if temperature is not None:
        config["temperature"] = float(temperature)

    seed = os.getenv("LLM_SEED")
    if seed is not None:
        config["seed"] = int(seed)
    

    return config


def _get_openai_client_kwargs(include_base_url: bool = True) -> dict:
    """Read OpenAI SDK client options from environment variables."""
    load_env()
    kwargs = {}

    base_url = _first_env("LLM_BASE_URL", "OPENAI_BASE_URL") if include_base_url else None
    if base_url:
        kwargs["base_url"] = base_url

    timeout = _first_env("LLM_TIMEOUT", "OPENAI_TIMEOUT")
    if timeout is not None:
        kwargs["timeout"] = float(timeout)

    max_retries = _first_env("LLM_MAX_RETRIES", "OPENAI_MAX_RETRIES")
    if max_retries is not None:
        kwargs["max_retries"] = int(max_retries)

    return kwargs


def _get_llm_client_settings(model: Optional[str] = None) -> LLMClientSettings:
    """Resolve credentials for one of the supported chat model profiles.

    Supported profile keys:
        gpt-4o, gemini-3.6-flash, claude-opus-4.8, qwen3.6-27b

    Select the profile with ``model=...`` or ``LLM_MODEL=...``. Provider-specific
    env vars store the actual remote model id:
        OPENAI_MODEL, GEMINI_MODEL, ANTHROPIC_MODEL, QWEN_MODEL
    """
    load_env()
    requested_model = model or os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o"
    profile_name = _LLM_PROFILE_ALIASES.get(
        requested_model.strip().lower(),
        _LLM_PROFILE_ALIASES.get(_normalize_profile_key(requested_model)),
    )
    if profile_name is None:
        supported = ", ".join(sorted(_LLM_PROFILES))
        raise ValueError(
            f"Unsupported LLM model/profile '{requested_model}'. "
            f"Supported profiles: {supported}."
        )
    profile = _LLM_PROFILES[profile_name]

    api_key_envs = ["LLM_API_KEY", *profile.get("api_key_envs", ())]
    api_key = _first_env(*api_key_envs)
    if not api_key:
        raise RuntimeError(
            "Missing required LLM API key. Set one of: "
            f"{', '.join(api_key_envs)}."
        )

    model_env = profile.get("model_env")
    provider_model_override = os.getenv(model_env) if isinstance(model_env, str) else None
    resolved_model = provider_model_override or str(profile["model"])
    base_url = (
        os.getenv("LLM_BASE_URL")
        or (
            os.getenv("OPENAI_BASE_URL")
            if profile_name == "gpt-4o"
            else None
        )
        or (os.getenv("GEMINI_BASE_URL") if profile_name == "gemini-3.6-flash" else None)
        or (os.getenv("ANTHROPIC_BASE_URL") if profile_name == "claude-opus-4.8" else None)
        or (os.getenv("QWEN_BASE_URL") if profile_name == "qwen3.6-27b" else None)
        or profile.get("base_url")
    )

    return LLMClientSettings(
        profile=profile_name,
        api_type=str(profile["api_type"]),
        api_key=api_key,
        model=resolved_model,
        base_url=str(base_url) if base_url else None,
    )


def _mapping_get(value: object, key: str, default: object = None) -> object:
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _extract_chat_text(response: object) -> str:
    """Extract assistant text from OpenAI-compatible chat responses.

    Some proxy providers return the message content as a plain string instead of
    the OpenAI SDK's ChatCompletion object.  Keep both paths supported.
    """
    if isinstance(response, str):
        text = response.strip()
        if text:
            return text
        raise ValueError("chat completion response string is empty")

    direct_content = _mapping_get(response, "content")
    if isinstance(direct_content, str) and direct_content.strip():
        return direct_content.strip()

    direct_text = _mapping_get(response, "text")
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text.strip()

    choices = _mapping_get(response, "choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(
            "chat completion response does not contain choices or direct text"
        )

    choice = choices[0]
    message = _mapping_get(choice, "message")
    if message is not None:
        content = _mapping_get(message, "content")
        if isinstance(content, str) and content.strip():
            return content.strip()

    content = _mapping_get(choice, "content")
    if isinstance(content, str) and content.strip():
        return content.strip()

    text = _mapping_get(choice, "text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    raise ValueError("chat completion response choice has no text content")


def _response_usage(response: object) -> object:
    return response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)


def _get_llm_max_tokens(default: int = 4096) -> int:
    value = os.getenv("LLM_MAX_TOKENS")
    if value is None:
        return default
    return int(value)


def _call_openai_compatible_chat(
    settings: LLMClientSettings,
    messages: List[dict],
    llm_config: dict,
) -> object:
    request_config = dict(llm_config)
    if settings.profile == "gemini-3.6-flash":
        request_config.pop("seed", None)

    client_kwargs = _get_openai_client_kwargs(include_base_url=False)
    if settings.base_url:
        client_kwargs["base_url"] = settings.base_url
    client = openai.OpenAI(api_key=settings.api_key, **client_kwargs)
    return client.chat.completions.create(
        model=settings.model,
        messages=messages,
        **request_config,
    )


def _call_anthropic_messages(
    settings: LLMClientSettings,
    system_prompt: str,
    user_prompt: str,
    llm_config: dict,
    response_format: Optional[dict],
    tools: Optional[List],
) -> object:
    if tools is not None:
        raise ValueError("tools are not supported for the claude-opus-4.8 profile")

    content = user_prompt
    if response_format == {"type": "json_object"}:
        content = f"{user_prompt}\n\nReturn only a valid JSON object."

    payload: Dict[str, object] = {
        "model": settings.model,
        "max_tokens": _get_llm_max_tokens(),
        "system": system_prompt,
        "messages": [{"role": "user", "content": content}],
    }
    for key in ("temperature",):
        if key in llm_config:
            payload[key] = llm_config[key]

    url = settings.base_url or "https://api.anthropic.com/v1/messages"
    timeout = float(_first_env("LLM_TIMEOUT", "ANTHROPIC_TIMEOUT", "OPENAI_TIMEOUT") or 60)
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-api-key": settings.api_key,
            "anthropic-version": os.getenv(
                "ANTHROPIC_VERSION",
                _DEFAULT_ANTHROPIC_VERSION,
            ),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Anthropic API request failed: HTTP {exc.code}: {body}") from exc


def _extract_anthropic_text(response: object) -> str:
    content = _mapping_get(response, "content")
    if not isinstance(content, list):
        raise ValueError("Anthropic response does not contain a content list")
    text_parts = []
    for item in content:
        if _mapping_get(item, "type") == "text":
            text = _mapping_get(item, "text")
            if isinstance(text, str):
                text_parts.append(text)
    text = "".join(text_parts).strip()
    if not text:
        raise ValueError("Anthropic response content has no text")
    return text


def run_embedding(
    text: Union[str, Sequence[str]],
    model: Optional[str] = None,
    normalize: bool = True,
    **kwargs,
) -> Union[List[float], List[List[float]]]:
    """Create embeddings with Qwen3-Embedding-8B through an OpenAI-compatible API.

    Set ``QWEN_API_KEY`` or ``DASHSCOPE_API_KEY`` in ``.env``. Optional knobs:
    ``QWEN_EMBEDDING_MODEL`` and ``QWEN_EMBEDDING_BASE_URL``.
    """
    load_env()
    if openai is None:
        raise ImportError(
            "The openai package is required to use embedding APIs. "
            "Install it with `pip install openai`."
        )

    single_input = isinstance(text, str)
    inputs = [text] if single_input else list(text)
    if not inputs:
        return []

    api_key = get_qwen_api_key()
    resolved_model = model or os.getenv("QWEN_EMBEDDING_MODEL", "Qwen3-Embedding-8B")
    base_url = os.getenv(
        "QWEN_EMBEDDING_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
        **_get_openai_client_kwargs(include_base_url=False),
    )
    response = client.embeddings.create(model=resolved_model, input=inputs, **kwargs)
    embeddings = [item.embedding for item in response.data]
    if normalize:
        embeddings = [_normalize_embedding(embedding) for embedding in embeddings]

    return embeddings[0] if single_input else embeddings


def run_llm(system_prompt: str, user_prompt: str, model: Optional[str] = None,
            response_format: Optional[dict] = None, tools: Optional[List] = None,
            return_usage: bool = False) -> Union[str, LLMResult]:
    """Send a prompt to OpenAI-compatible API and return the text response.

    response_format and tools are optional
    """
    load_env()
    if openai is None:
        raise ImportError(
            "The openai package is required to use LLM APIs. "
            "Install it with `pip install openai`."
        )

    settings = _get_llm_client_settings(model)

    # Merge env-based config
    llm_config = _get_llm_config()
    if response_format is not None:
        llm_config["response_format"] = response_format
    if tools is not None:
        llm_config["tools"] = tools

    
    
    prompt=[{"role": "system", "content": system_prompt},{"role": "user", "content": user_prompt}]

    if settings.api_type == "openai_compatible":
        response = _call_openai_compatible_chat(settings, prompt, llm_config)
        text = _extract_chat_text(response)
    elif settings.api_type == "anthropic_messages":
        response = _call_anthropic_messages(
            settings,
            system_prompt,
            user_prompt,
            llm_config,
            response_format,
            tools,
        )
        text = _extract_anthropic_text(response)
    else:
        raise ValueError(f"Unsupported LLM API type '{settings.api_type}'")

    if not return_usage:
        return text
    return LLMResult(
        text=text,
        usage=_normalize_llm_usage(_response_usage(response)),
    )


if __name__ == "__main__":
    load_env()
    parser = argparse.ArgumentParser(description="Smoke-test one configured LLM profile.")
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Profile to use: gpt-4o, gemini, claude, or qwen. "
            "The actual provider model id comes from OPENAI_MODEL, "
            "GEMINI_MODEL, ANTHROPIC_MODEL, or QWEN_MODEL."
        ),
    )
    parser.add_argument(
        "--system",
        default="You are a helpful assistant.",
        help="System prompt for the smoke test.",
    )
    parser.add_argument(
        "--prompt",
        default="Reccommend a good book to read.",
        help="User prompt for the smoke test.",
    )
    args = parser.parse_args()

    settings = _get_llm_client_settings(args.model)
    system_prompt = args.system
    user_prompt = args.prompt
    print(f"Testing LLM profile/model: {settings.profile}/{settings.model}")
    print(f"Config: {_get_llm_config()}")
    response = run_llm(system_prompt, user_prompt, model=args.model)
    print(f"Prompt: {system_prompt} {user_prompt}")
    print(f"Response: {response}")
