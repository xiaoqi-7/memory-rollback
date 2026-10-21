import json
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Tuple


# Standalone default; AgentRunner overrides this from domain config data_source.
_DEFAULT_DB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "adapted_lmev2_subset",
    "lme_derived",
    "catalog",
)
_CONFIGURED_DB_DIR = _DEFAULT_DB_DIR

_MAX_TOP_K = 3
_MAX_BEST_STATES = 2
_MAX_SNIPPET_CHARS = 300
_MAX_LOOKUP_TEXT_CHARS = 3000
_MAX_EXCERPT_CHARS = 260

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "the",
    "this",
    "to",
    "what",
    "when",
    "which",
    "with",
    "you",
    "your",
}


def _db_dir() -> str:
    return _CONFIGURED_DB_DIR


def configure_db_dir(path: str) -> None:
    """Select the catalog derived from the active domain data_source."""
    global _CONFIGURED_DB_DIR
    _CONFIGURED_DB_DIR = os.path.abspath(path)
    _load_jsonl.cache_clear()
    _trajectories_by_id.cache_clear()
    _states_by_key.cache_clear()


@lru_cache(maxsize=None)
def _load_jsonl(filename: str) -> List[Dict[str, Any]]:
    """Load and cache a JSONL catalog file."""
    path = os.path.join(_db_dir(), filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"LME-derived catalog file not found: {path}. "
            "Expected <data_source>/catalog for lme_derived."
        )

    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_no} of {path}: {exc}"
                ) from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _trajectories() -> List[Dict[str, Any]]:
    return _load_jsonl("trajectory_index.jsonl")


def _states() -> List[Dict[str, Any]]:
    return _load_jsonl("state_chunks.jsonl")


@lru_cache(maxsize=None)
def _trajectories_by_id() -> Dict[str, Dict[str, Any]]:
    return {
        str(row.get("trajectory_id")): row
        for row in _trajectories()
        if row.get("trajectory_id") is not None
    }


def _state_rank(row: Dict[str, Any]) -> Tuple[int, int]:
    text_len = len(str(row.get("search_text") or ""))
    try:
        score = int(row.get("score") or 0)
    except (TypeError, ValueError):
        score = 0
    return score, text_len


@lru_cache(maxsize=None)
def _states_by_key() -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Index states by (trajectory_id, state_index), deduplicating question refs."""
    indexed: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in _states():
        trajectory_id = row.get("trajectory_id")
        state_index = row.get("state_index")
        if trajectory_id is None or state_index is None:
            continue
        try:
            key = (str(trajectory_id), int(state_index))
        except (TypeError, ValueError):
            continue
        previous = indexed.get(key)
        if previous is None or _state_rank(row) > _state_rank(previous):
            indexed[key] = row
    return indexed


def _tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if token in _STOPWORDS:
            continue
        if len(token) <= 1 and not token.isdigit():
            continue
        tokens.append(token)
    return list(dict.fromkeys(tokens))


def _query_phrases(tokens: List[str]) -> List[str]:
    phrases: List[str] = []
    for width in range(min(5, len(tokens)), 1, -1):
        for start in range(0, len(tokens) - width + 1):
            phrases.append(" ".join(tokens[start : start + width]))
    return list(dict.fromkeys(phrases))


def _normalize_for_phrase(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _searchable_trajectory_text(row: Dict[str, Any]) -> str:
    fields = ("search_text", "goal", "outcome", "start_url", "environment")
    return "\n".join(str(row.get(field) or "") for field in fields)


def _searchable_state_text(row: Dict[str, Any]) -> str:
    fields = ("search_text", "thought", "action", "url")
    return "\n".join(str(row.get(field) or "") for field in fields)


def _term_hits(text: str, query_tokens: List[str]) -> List[str]:
    normalized = text.lower()
    return [token for token in query_tokens if token in normalized]


def _phrase_hits(text: str, query_phrases: List[str]) -> List[str]:
    normalized = _normalize_for_phrase(text)
    return [phrase for phrase in query_phrases if phrase in normalized]


def _snippet(text: str, matched_terms: List[str]) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= _MAX_SNIPPET_CHARS:
        return collapsed

    first_hit = min(
        (collapsed.lower().find(term) for term in matched_terms if term),
        default=-1,
    )
    if first_hit < 0:
        return collapsed[:_MAX_SNIPPET_CHARS] + "..."

    start = max(0, first_hit - (_MAX_SNIPPET_CHARS // 3))
    end = min(len(collapsed), start + _MAX_SNIPPET_CHARS)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(collapsed) else ""
    return prefix + collapsed[start:end] + suffix


_NOISY_PHRASES = {
    "additions text",
    "all",
    "assertive",
    "both",
    "false",
    "listbox",
    "polite",
    "true",
}


def _clean_visible_phrase(phrase: str) -> str | None:
    phrase = re.sub(r"\s+", " ", phrase).strip()
    if not phrase:
        return None
    lowered = phrase.lower()
    if lowered in _NOISY_PHRASES:
        return None
    if "=" in phrase or phrase.startswith(","):
        return None
    if len(phrase) > 140:
        return None
    return phrase


def _quoted_phrases(text: str) -> List[str]:
    phrases: List[str] = []

    for label, value in re.findall(r"combobox '([^']{2,80})' value='([^']{2,80})'", text):
        clean_label = _clean_visible_phrase(label)
        clean_value = _clean_visible_phrase(value)
        if clean_label and clean_value:
            phrases.append(f"{clean_label}: {clean_value}")
            phrases.append(clean_value)

    ui_pattern = (
        r"(?:StaticText|button|link|option|textbox|heading|tab|menuitem|alert|region) "
        r"'([^']{2,140})'"
    )
    for phrase in re.findall(ui_pattern, text):
        clean = _clean_visible_phrase(phrase)
        if clean:
            phrases.append(clean)

    for match in re.findall(r"'([^']{2,140})'|\"([^\"]{2,140})\"", text):
        phrase = _clean_visible_phrase(match[0] or match[1])
        if phrase:
            phrases.append(phrase)

    return list(dict.fromkeys(phrases))


def _phrase_score(phrase: str) -> tuple[int, int, int]:
    lowered = phrase.lower()
    score = 0
    if lowered.startswith("approval:"):
        score += 6
    if lowered.startswith("stage:"):
        score -= 4
    if ":" in phrase:
        score += 3
    if any(token in lowered for token in ("approval", "requested", "created", "device", "email", "popup", "message", "value")):
        score += 4
    if "email" in lowered and "device" in lowered and "created" in lowered:
        score += 6
    if lowered.startswith("primary "):
        score += 2
    if len(phrase.split()) >= 2:
        score += 2
    if len(phrase) <= 40:
        score += 1
    return score, -len(phrase), -len(phrase.split())


def _evidence_phrases(text: str, limit: int = 3) -> List[str]:
    phrases = _quoted_phrases(text)
    if not phrases:
        return []
    return sorted(phrases, key=_phrase_score, reverse=True)[:limit]


def _evidence_excerpt(text: str, phrases: List[str]) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if not collapsed:
        return ""
    candidates = phrases or _evidence_phrases(collapsed, limit=1)
    if not candidates:
        return collapsed[:_MAX_EXCERPT_CHARS]
    phrase = candidates[0]
    idx = collapsed.lower().find(phrase.lower())
    if idx < 0 and ":" in phrase:
        phrase = phrase.split(":", 1)[1].strip()
        idx = collapsed.lower().find(phrase.lower())
    if idx < 0:
        return collapsed[:_MAX_EXCERPT_CHARS]
    start = max(0, idx - 100)
    end = min(len(collapsed), idx + len(phrase) + 120)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(collapsed) else ""
    return prefix + collapsed[start:end] + suffix


def _clamp_top_k(top_k: int) -> int:
    try:
        value = int(top_k)
    except (TypeError, ValueError):
        value = 5
    return max(1, min(_MAX_TOP_K, value))


def _public_trajectory(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "trajectory_id": row.get("trajectory_id"),
        "domain": row.get("domain"),
        "environment": row.get("environment"),
        "goal": row.get("goal"),
        "outcome": row.get("outcome"),
        "start_url": row.get("start_url"),
        "state_count": row.get("state_count"),
    }


def search_trajectory_history(query: str, top_k: int = 5) -> dict:
    """Read-only search over recorded LME-derived trajectory and state evidence."""
    query_text = str(query or "")
    query_tokens = _tokenize(query_text)
    query_phrases = _query_phrases(query_tokens)
    limit = _clamp_top_k(top_k)

    if not query_tokens:
        return {"query": query, "results": [], "count": 0}

    states_by_trajectory: Dict[str, List[Dict[str, Any]]] = {}
    for state in _states_by_key().values():
        trajectory_id = str(state.get("trajectory_id"))
        state_text = _searchable_state_text(state)
        matched = _term_hits(state_text, query_tokens)
        matched_phrases = _phrase_hits(state_text, query_phrases)
        if not matched and not matched_phrases:
            continue
        states_by_trajectory.setdefault(trajectory_id, []).append(
            {
                "state": state,
                "matched_terms": list(dict.fromkeys(matched + matched_phrases)),
                "score": len(matched) + (len(matched_phrases) * 3),
            }
        )

    scored: List[Dict[str, Any]] = []
    for trajectory in _trajectories():
        trajectory_id = str(trajectory.get("trajectory_id"))
        trajectory_text = _searchable_trajectory_text(trajectory)
        trajectory_matches = _term_hits(trajectory_text, query_tokens)
        trajectory_phrase_matches = _phrase_hits(trajectory_text, query_phrases)
        state_hits = states_by_trajectory.get(trajectory_id, [])
        if not trajectory_matches and not trajectory_phrase_matches and not state_hits:
            continue

        best_states = sorted(
            state_hits,
            key=lambda item: (
                item["score"],
                _state_rank(item["state"])[0],
                -int(item["state"].get("state_index") or 0),
            ),
            reverse=True,
        )[:_MAX_BEST_STATES]
        state_score = sum(item["score"] for item in best_states)
        score = (
            (len(trajectory_matches) * 2)
            + (len(trajectory_phrase_matches) * 6)
            + state_score
        )
        matched_terms = list(
            dict.fromkeys(
                trajectory_matches
                + trajectory_phrase_matches
                + [term for item in best_states for term in item["matched_terms"]]
            )
        )
        public = _public_trajectory(trajectory)
        public.update(
            {
                "score": score,
                "matched_terms": matched_terms,
                "best_states": [
                    {
                        "trajectory_id": item["state"].get("trajectory_id"),
                        "state_index": item["state"].get("state_index"),
                        "step": item["state"].get("step"),
                        "url": item["state"].get("url"),
                        "score": item["score"],
                        "matched_terms": item["matched_terms"],
                        "snippet": _snippet(
                            str(item["state"].get("search_text") or ""),
                            item["matched_terms"],
                        ),
                    }
                    for item in best_states
                ],
            }
        )
        scored.append(public)

    scored.sort(
        key=lambda row: (
            int(row.get("score") or 0),
            len(row.get("best_states") or []),
            str(row.get("trajectory_id") or ""),
        ),
        reverse=True,
    )
    results = scored[:limit]
    return {"query": query, "results": results, "count": len(results)}


def lookup_state(trajectory_id: str, state_index: int) -> dict:
    """Read-only lookup for one recorded LME-derived state evidence chunk."""
    try:
        key = (str(trajectory_id), int(state_index))
    except (TypeError, ValueError):
        return {
            "trajectory_id": trajectory_id,
            "state_index": state_index,
            "error": "state not found",
        }

    state = _states_by_key().get(key)
    if state is None:
        return {
            "trajectory_id": trajectory_id,
            "state_index": state_index,
            "error": "state not found",
        }

    search_text = str(state.get("search_text") or "")
    key_phrases = _evidence_phrases(search_text)
    evidence_excerpt = _evidence_excerpt(search_text, key_phrases)
    truncated = len(search_text) > _MAX_LOOKUP_TEXT_CHARS
    if truncated:
        search_text = search_text[:_MAX_LOOKUP_TEXT_CHARS] + "\n...[truncated]"

    return {
        "trajectory_id": state.get("trajectory_id"),
        "state_index": state.get("state_index"),
        "step": state.get("step"),
        "url": state.get("url"),
        "thought": state.get("thought"),
        "action": state.get("action"),
        "evidence_excerpt": evidence_excerpt,
        "key_phrases": key_phrases,
        "search_text": search_text,
        "truncated": truncated,
    }


def lookup_trajectory(trajectory_id: str) -> dict:
    """Read-only lookup for one recorded LME-derived trajectory summary."""
    trajectory = _trajectories_by_id().get(str(trajectory_id))
    if trajectory is None:
        return {"trajectory_id": trajectory_id, "error": "trajectory not found"}

    return {
        "trajectory_id": trajectory.get("trajectory_id"),
        "domain": trajectory.get("domain"),
        "environment": trajectory.get("environment"),
        "goal": trajectory.get("goal"),
        "outcome": trajectory.get("outcome"),
        "start_url": trajectory.get("start_url"),
        "state_count": trajectory.get("state_count"),
        "state_outline": trajectory.get("state_outline"),
    }
