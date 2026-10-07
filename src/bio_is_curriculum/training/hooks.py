"""Training hooks for dynamic curricula."""

from __future__ import annotations

from typing import Callable

HookFn = Callable[[], None]
