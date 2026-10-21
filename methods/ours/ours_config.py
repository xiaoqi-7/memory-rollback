# methods/ours/ours_config.py
"""Typed config for the "ours" repair pipeline.

This module is the single loader for ``configs/ours_base.yaml``. It turns that
YAML into a typed :class:`OursConfig` (a dataclass-per-section tree) so callers
get attribute access and typo-catching — ``config.support_check.trust_threshold``
rather than ``config["support_check"]["trust_threshold"]`` with a silent
``KeyError`` on a misspelled key.

Two design choices mirror ``graph_schema.py``'s "catch drift early" philosophy:

* :func:`load_config` fails loud on an unrecognized YAML key (top-level OR
  nested) instead of silently ignoring it, so a typo in the base file — or a
  field that drifted out of the dataclasses — is a hard error, not a value that
  silently reverts to its default.
* :func:`apply_overrides` coerces each CLI string to the TARGET field's real
  type (read off ``dataclasses.fields``), and raises naming the bad dotted path
  if it does not resolve to a real field — a misspelled override never
  no-ops silently.
"""

import dataclasses
from dataclasses import dataclass, fields
from typing import Any, Dict

import yaml

# NOTE: deliberately NO ``from __future__ import annotations`` here. Both
# _SECTION_TYPES and _coerce rely on ``dataclasses.fields(...).type`` being the
# real type OBJECT (so ``section_cls(**raw)`` and ``target_type is bool`` work);
# the future import would stringize every annotation ("bool", "float") and break
# that. No forward references are needed — each section dataclass is defined
# before OursConfig references it.


# --------------------------------------------------------------------------- #
# Section dataclasses — one per top-level YAML block. Defaults here mirror
# configs/ours_base.yaml so an OursConfig is constructible without the file,
# but load_config is the normal entry point.
# --------------------------------------------------------------------------- #

@dataclass
class SupportCheckConfig:
    use_trust_score: bool = False
    trust_threshold: float = 0.8


@dataclass
class RollbackObjectiveConfig:
    lambda_m: float = 1.0
    lambda_c: float = 1.0
    lambda_a: float = 1.0
    lambda_t: float = 0.01
    lambda_u: float = 1.0


@dataclass
class AblationConfig:
    skip_support_check: bool = False
    # skip_derived_cleanup is intentionally not exposed as a public config
    # entry. The rollback optimizer still keeps the implementation branch for
    # internal/design-detail experiments, but configs/ours_base.yaml and
    # ordinary --set overrides should not surface it as a main ablation.
    # skip_derived_cleanup: bool = False
    skip_rollback_optimizer: bool = False
    skip_selective_replay: bool = False


@dataclass
class OursConfig:
    support_check: SupportCheckConfig
    rollback_objective: RollbackObjectiveConfig
    ablation: AblationConfig


# Top-level field name -> section dataclass. Pulled from dataclasses.fields so
# it can't drift if OursConfig gains/loses a section.
_SECTION_TYPES: Dict[str, type] = {
    f.name: f.type for f in fields(OursConfig)
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def _build_section(section_name: str, section_cls: type, raw: Any) -> Any:
    """Instantiate one section dataclass from its YAML sub-dict, failing loud.

    Raises ValueError naming the offending ``section.key`` if ``raw`` carries a
    key not declared on ``section_cls`` (a typo or a drifted field), rather than
    dropping it silently.
    """
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"config section '{section_name}' must be a mapping, "
            f"got {type(raw).__name__}"
        )
    known = {f.name for f in fields(section_cls)}
    unknown = set(raw) - known
    if unknown:
        bad = sorted(unknown)[0]
        raise ValueError(
            f"unrecognized config key '{section_name}.{bad}' "
            f"(known keys: {sorted(known)})"
        )
    return section_cls(**raw)


def load_config(path: str = "configs/ours_base.yaml") -> OursConfig:
    """Load YAML at ``path`` into a typed :class:`OursConfig`.

    Uses ``yaml.safe_load`` (never ``yaml.load``). Raises ``ValueError`` naming
    the offending key if the YAML contains a top-level OR nested key not
    recognized by the dataclasses above — failing loud on a typo or drifted
    field rather than silently ignoring it (mirrors graph_schema.py's self-check
    philosophy). A missing section falls back to that section's dataclass
    defaults.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(
            f"config root must be a mapping, got {type(data).__name__}"
        )

    unknown_sections = set(data) - set(_SECTION_TYPES)
    if unknown_sections:
        bad = sorted(unknown_sections)[0]
        raise ValueError(
            f"unrecognized top-level config section '{bad}' "
            f"(known sections: {sorted(_SECTION_TYPES)})"
        )

    return OursConfig(
        **{
            name: _build_section(name, cls, data.get(name))
            for name, cls in _SECTION_TYPES.items()
        }
    )


# --------------------------------------------------------------------------- #
# Overrides
# --------------------------------------------------------------------------- #

def _coerce(value: str, target_type: type, dotted_path: str) -> Any:
    """Coerce a CLI string to ``target_type`` (bool/float/int/str).

    Booleans accept only "true"/"false" (case-insensitive) and reject anything
    else with a clear ValueError; floats/ints parse via their constructors. The
    target type is read off the existing field, so overrides are never blindly
    eval()'d or assumed to be one type.
    """
    if target_type is bool:
        low = value.strip().lower()
        if low == "true":
            return True
        if low == "false":
            return False
        raise ValueError(
            f"override '{dotted_path}' expects a boolean "
            f"('true'/'false', case-insensitive), got {value!r}"
        )
    if target_type is float:
        try:
            return float(value)
        except ValueError:
            raise ValueError(
                f"override '{dotted_path}' expects a float, got {value!r}"
            )
    if target_type is int:
        try:
            return int(value)
        except ValueError:
            raise ValueError(
                f"override '{dotted_path}' expects an int, got {value!r}"
            )
    # str (and any unforeseen type) passes through verbatim.
    return value


def apply_overrides(config: OursConfig, overrides: Dict[str, str]) -> OursConfig:
    """Return a NEW :class:`OursConfig` with dotted-path fields overridden.

    Does NOT mutate ``config``: it deep-copies first, then writes each override
    onto the copy. Keys are dotted paths matching the YAML nesting, e.g.
    ``{"support_check.use_trust_score": "false"}``. Values arrive as strings
    (from CLI parsing) and are coerced to the target field's actual type by
    inspecting that field via ``dataclasses.fields`` — bool/float/int/str are
    handled distinctly, nothing is eval()'d.

    Raises ``KeyError`` naming the bad dotted path if it does not resolve to a
    real ``section.field`` (e.g. ``support_check.trust_threshhold`` typo, or a
    path without exactly one dot), so a misspelled override fails loud instead
    of silently no-op-ing. Coercion failures raise ``ValueError`` (see
    :func:`_coerce`).
    """
    new_config = dataclasses.replace(
        config,
        support_check=dataclasses.replace(config.support_check),
        rollback_objective=dataclasses.replace(config.rollback_objective),
        ablation=dataclasses.replace(config.ablation),
    )

    for dotted_path, raw_value in overrides.items():
        parts = dotted_path.split(".")
        if len(parts) != 2:
            raise KeyError(
                f"override path '{dotted_path}' must be 'section.field' "
                f"(exactly one dot)"
            )
        section_name, field_name = parts

        if section_name not in _SECTION_TYPES:
            raise KeyError(
                f"override path '{dotted_path}' names unknown section "
                f"'{section_name}' (known: {sorted(_SECTION_TYPES)})"
            )
        section = getattr(new_config, section_name)
        field_types = {f.name: f.type for f in fields(section)}
        if field_name not in field_types:
            raise KeyError(
                f"override path '{dotted_path}' names unknown field "
                f"'{field_name}' on section '{section_name}' "
                f"(known: {sorted(field_types)})"
            )

        coerced = _coerce(raw_value, field_types[field_name], dotted_path)
        setattr(section, field_name, coerced)

    return new_config
