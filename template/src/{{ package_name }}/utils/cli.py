"""Hydra-style CLI on top of tyro: typed configs, familiar syntax.

Keeps the two ergonomics Hydra got right, on fully typed pydantic configs:

1. Free-order ``key=value`` tokens with whole-group swaps::

       python train.py experiment=example loss=contrastive loss.temperature=0.1 model.lr=1e-3

   ``experiment=`` picks a preset config, ``loss=`` swaps a typed config
   variant (a "group"), and every other ``a.b=v`` is a dot-path override —
   in any order. Booleans take true/false/1/0/yes/no; optional fields take
   ``null``. Typos die at parse time with a suggestion, not at runtime.

2. ``${...}`` references inside YAML override files (see ``parse``'s
   ``local_yaml``): ``random_state: ${seed}`` resolves against the composed
   config before CLI overrides apply. (In code, prefer the derived-field
   pattern: a ``None`` default resolved explicitly from a sibling config.)

Plain tyro flags (``--model.lr 1e-3``, ``--no-deterministic``, ``--help``)
work too — ``key=value`` tokens are rewritten to flags before tyro parses.
"""

import re
import sys
import types
import typing
from collections.abc import Mapping
from pathlib import Path
from typing import TypeVar, cast

import pydantic
import tyro
import yaml

T = TypeVar("T", bound=pydantic.BaseModel)
M = TypeVar("M", bound=pydantic.BaseModel)

_REF = re.compile(r"\$\{([\w.]+)\}")
_TRUE = {"true", "1", "yes", "on"}
_FALSE = {"false", "0", "no", "off"}


def deep_merge(model: M, data: dict) -> M:
    """Nested model_copy(update=...): dicts merge into sub-models, scalars replace.

    Sub-models not named in ``data`` are deep-copied, so the result never
    aliases the input's nested configs (presets stay pristine across calls).
    """
    if not isinstance(data, Mapping):
        raise TypeError(f"expected a mapping to merge into {type(model).__name__}, got {data!r}")
    update = {}
    for key in type(model).model_fields:
        current = getattr(model, key)
        if key in data:
            value = data[key]
            if isinstance(current, pydantic.BaseModel) and isinstance(value, Mapping):
                update[key] = deep_merge(current, dict(value))
            else:
                update[key] = value
        elif isinstance(current, pydantic.BaseModel):
            update[key] = current.model_copy(deep=True)  # never alias nested configs
    unknown = set(data) - set(type(model).model_fields)
    if unknown:
        raise KeyError(
            f"unknown config key(s) {sorted(unknown)} (valid: {sorted(type(model).model_fields)})"
        )
    return cast(M, model.model_copy(update=update))


def _lookup(root: pydantic.BaseModel, dotted: str):
    obj = root
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def _resolve_refs(value, root: pydantic.BaseModel):
    """Replace ``${a.b}`` strings in a YAML tree with values from the config.

    A whole-string reference keeps the referenced value's type; references
    embedded in a longer string substitute as text (Hydra-style).
    """
    if isinstance(value, str):
        match = _REF.fullmatch(value.strip())
        if match:
            return _lookup(root, match.group(1))
        return _REF.sub(lambda m: str(_lookup(root, m.group(1))), value)
    if isinstance(value, dict):
        return {k: _resolve_refs(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_refs(v, root) for v in value]
    return value


def _skeleton(config_type: type[T]) -> T:
    """A default instance with required fields set to tyro.MISSING.

    Lets local.yaml merge into entry points whose config has required fields
    (eval.py's ckpt_path) — tyro still demands the MISSING ones from the CLI.
    """
    required = {
        name: tyro.MISSING
        for name, field in config_type.model_fields.items()
        if field.is_required()
    }
    return cast(T, config_type.model_construct(**required))


def _annotation(root: type[pydantic.BaseModel] | None, dotted: str):
    """The leaf annotation for a dotted path, or None if it can't be resolved."""
    current: object = root
    for part in dotted.split("."):
        if not (isinstance(current, type) and issubclass(current, pydantic.BaseModel)):
            return None
        field = current.model_fields.get(part)
        if field is None:
            return None
        current = field.annotation
    return current


def _accepts_none(annotation) -> bool:
    if annotation is None:
        return False
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        return type(None) in typing.get_args(annotation)
    return False


def parse(
    config_type: type[T],
    presets: dict[str, T] | None = None,
    groups: Mapping[str, Mapping[str, type[pydantic.BaseModel]]] | None = None,
    argv: list[str] | None = None,
    local_yaml: str | Path = "configs/local.yaml",
    description: str = "",
) -> T:
    """Parse a typed config from Hydra-style CLI args.

    Composition order (later wins): preset -> configs/local.yaml -> group
    swaps -> key=value / flag overrides.

    Args:
        config_type: Root pydantic config class.
        presets: Named full configs; ``experiment=<name>`` selects one
            ("base" is the default and must exist if presets are given).
        groups: Swappable variants per field, e.g.
            ``{"loss": {"contrastive": ContrastiveLossConfig}}`` enables
            ``loss=contrastive``.
        argv: Defaults to sys.argv[1:].
        local_yaml: Machine-local override file (gitignored); silently
            skipped when absent. ``${a.b}`` strings resolve against the
            composed config. Applies to every entry point, presets or not.
        description: Shown in --help, alongside available presets/groups.
    """
    groups = groups or {}
    argv = list(sys.argv[1:]) if argv is None else list(argv)

    # pass 1: pull out experiment= (preset choice must precede everything)
    rest: list[str] = []
    default: T | None = None
    for token in argv:
        if token.startswith("experiment=") and not token.startswith("-"):
            if not presets:
                sys.exit("experiment= is not supported by this entry point")
            name = token.split("=", 1)[1]
            if name not in presets:
                sys.exit(f"unknown experiment {name!r} — available: {', '.join(sorted(presets))}")
            default = presets[name]
        else:
            rest.append(token)
    if default is None and presets:
        default = presets["base"]

    # pass 2: machine-local overrides, with ${...} resolution. Without
    # presets, merge into a skeleton so local.yaml applies everywhere
    # (required fields stay tyro.MISSING and must come from the CLI).
    local_path = Path(local_yaml)
    if local_path.exists():
        data = yaml.safe_load(local_path.read_text()) or {}
        if not isinstance(data, dict):
            sys.exit(f"{local_path} must be a mapping of config keys, got {type(data).__name__}")
        if data:
            base = default if default is not None else _skeleton(config_type)
            default = deep_merge(base, _resolve_refs(data, base))

    # pass 3a: collect group swaps and key=value overrides; pass plain
    # flags (--help, --model.lr ...) straight through to tyro
    flags: list[str] = []
    overrides: list[tuple[str, str]] = []
    swaps: dict[str, type[pydantic.BaseModel]] = {}
    for token in rest:
        if "=" in token and not token.startswith("-"):
            key, value = token.split("=", 1)
            if key in groups:
                if value not in groups[key]:
                    sys.exit(
                        f"unknown {key} variant {value!r} — available: "
                        f"{', '.join(sorted(groups[key]))}"
                    )
                swaps[key] = groups[key][value]
            else:
                overrides.append((key, value))
        else:
            flags.append(token)
    if swaps:
        default = cast(
            T,
            (default or config_type()).model_copy(
                update={key: variant() for key, variant in swaps.items()}
            ),
        )

    # pass 3b: rewrite overrides as tyro flags, translating booleans
    # (tyro uses --x/--no-x switches) and null for optional fields
    for key, value in overrides:
        first, _, remainder = key.partition(".")
        if first in swaps and remainder:
            leaf_annotation = _annotation(swaps[first], remainder)
        else:
            leaf_annotation = _annotation(config_type, key)
        if leaf_annotation is bool:
            lowered = value.strip().lower()
            if lowered in _TRUE:
                flags.append(f"--{key}")
            elif lowered in _FALSE:
                prefix, _, leaf = key.rpartition(".")
                flags.append(f"--{prefix}.no-{leaf}" if prefix else f"--no-{leaf}")
            else:
                sys.exit(f"invalid boolean for {key}: {value!r} (use true/false)")
        else:
            if value in ("null", "~") and _accepts_none(leaf_annotation):
                value = "None"
            flags.extend([f"--{key}", value])

    help_lines = [description] if description else []
    if presets:
        help_lines.append(f"experiments: {', '.join(sorted(presets))} (experiment=<name>)")
    for group, members in groups.items():
        help_lines.append(f"{group} variants: {', '.join(sorted(members))} ({group}=<name>)")

    shape = tyro.conf.AvoidSubcommands[config_type]  # type: ignore[valid-type]
    help_text = "\n".join(help_lines)
    if default is None:
        result = tyro.cli(shape, args=flags, description=help_text)  # type: ignore[call-overload]
    else:
        result = tyro.cli(shape, default=default, args=flags, description=help_text)  # type: ignore[call-overload]
    return cast(T, result)
