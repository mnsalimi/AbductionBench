"""Runtime resolution of child adapters.

Adapters are named in configuration as ``"module.path:ClassName"`` and imported
on demand.  Nothing in :mod:`abductionbench.core` imports adapter modules, so
the engine remains loadable (and testable) with no adapters installed at all.

A decorator-based registry is also provided for adapters that prefer to
self-register under a short id; ``resolve_adapter`` accepts either form.
"""

from __future__ import annotations

import importlib
import logging
from typing import Callable, TypeVar

from .adapter import DatasetAdapter
from .errors import ConfigError

logger = logging.getLogger(__name__)

__all__ = ["register_adapter", "resolve_adapter", "registered_adapters"]

_REGISTRY: dict[str, type[DatasetAdapter]] = {}

T = TypeVar("T", bound=type[DatasetAdapter])


def register_adapter(dataset_id: str) -> Callable[[T], T]:
    """Class decorator registering an adapter under a short id."""

    def decorator(cls: T) -> T:
        if not issubclass(cls, DatasetAdapter):
            raise ConfigError(f"{cls!r} is not a DatasetAdapter subclass")
        existing = _REGISTRY.get(dataset_id)
        if existing is not None and existing is not cls:
            raise ConfigError(
                f"adapter id {dataset_id!r} already registered by {existing.__name__}"
            )
        _REGISTRY[dataset_id] = cls
        if not cls.dataset_id:
            cls.dataset_id = dataset_id
        return cls

    return decorator


def registered_adapters() -> dict[str, type[DatasetAdapter]]:
    return dict(_REGISTRY)


def resolve_adapter(spec: str) -> type[DatasetAdapter]:
    """Resolve ``"module:Class"``, ``"module"`` or a registered short id.

    ``"module"`` (no colon) imports the module and expects exactly one
    ``DatasetAdapter`` subclass defined in it -- convenient for the common
    one-adapter-per-module layout.
    """
    if spec in _REGISTRY:
        return _REGISTRY[spec]

    module_name, _, class_name = spec.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ConfigError(
            f"cannot import adapter module {module_name!r} for spec {spec!r}: {exc}"
        ) from exc

    if class_name:
        try:
            cls = getattr(module, class_name)
        except AttributeError as exc:
            raise ConfigError(f"module {module_name!r} has no attribute {class_name!r}") from exc
    else:
        candidates = [
            obj
            for obj in vars(module).values()
            if isinstance(obj, type)
            and issubclass(obj, DatasetAdapter)
            and obj is not DatasetAdapter
            and obj.__module__ == module.__name__
        ]
        if len(candidates) != 1:
            raise ConfigError(
                f"adapter spec {spec!r} is ambiguous: module defines "
                f"{[c.__name__ for c in candidates]}; use 'module:ClassName'"
            )
        cls = candidates[0]

    if not (isinstance(cls, type) and issubclass(cls, DatasetAdapter)):
        raise ConfigError(f"{spec!r} does not name a DatasetAdapter subclass")
    return cls
