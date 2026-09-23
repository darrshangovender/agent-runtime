"""Input and output guards.

* :func:`sanitize_input` strips control characters and wraps untrusted text in
  ``<user_input>`` delimiters so a prompt can distinguish instructions from data.
* :func:`tenant_filter` refuses any tool result that does not carry the caller's
  ``tenant_id``, so a mis-scoped query cannot leak another tenant's rows.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

OPEN_TAG = "<user_input>"
CLOSE_TAG = "</user_input>"

# C0 controls except tab/newline/carriage return, DEL, C1 controls, and the
# zero-width / bidi-override code points routinely used to hide instructions
# inside pasted text. Built from code points so no literal control character
# has to live in this source file.
_STRIP_CODEPOINTS: list[int] = (
    [c for c in range(0x20) if c not in (0x09, 0x0A, 0x0D)]
    + list(range(0x7F, 0xA0))
    + [0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x2028, 0x2029]
    + list(range(0x202A, 0x202F))
    + list(range(0x2060, 0x2065))
    + [0xFEFF]
)
_CONTROL_RE = re.compile("[" + "".join(re.escape(chr(c)) for c in _STRIP_CODEPOINTS) + "]")
_TAG_RE = re.compile(r"<\s*/?\s*user_input\s*>", re.IGNORECASE)


def strip_control_chars(text: str) -> str:
    """Remove control characters (keeps tab, newline, carriage return)."""
    text = unicodedata.normalize("NFC", text)
    return _CONTROL_RE.sub("", text)


def sanitize_input(text: Any, *, max_chars: int | None = None) -> str:
    """Return ``text`` cleaned and wrapped in ``<user_input>`` delimiters.

    Any literal ``<user_input>``/``</user_input>`` tags inside the untrusted
    text are neutralised so the text cannot close the delimiter early.
    """
    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text)
    cleaned = strip_control_chars(text)
    cleaned = _TAG_RE.sub(lambda m: m.group(0).replace("<", "&lt;").replace(">", "&gt;"), cleaned)
    cleaned = cleaned.strip()
    if max_chars is not None and max_chars >= 0 and len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
    return f"{OPEN_TAG}\n{cleaned}\n{CLOSE_TAG}"


def unwrap_input(wrapped: str) -> str:
    """Inverse of :func:`sanitize_input` for logging/tests (does not restore tags)."""
    return wrapped.removeprefix(OPEN_TAG).removesuffix(CLOSE_TAG).strip("\n")


class TenantMismatchError(PermissionError):
    """A tool result was missing the caller's tenant id or carried a different one."""

    def __init__(self, expected: str, found: Any, where: str = "result") -> None:
        self.expected = expected
        self.found = found
        super().__init__(
            f"tenant filter: {where} has tenant_id={found!r}, caller is {expected!r}"
        )


def _get_field(obj: Any, key: str) -> tuple[bool, Any]:
    if isinstance(obj, BaseModel):
        if key in type(obj).model_fields:
            return True, getattr(obj, key)
        extra = getattr(obj, "model_extra", None) or {}
        return (key in extra), extra.get(key)
    if isinstance(obj, Mapping):
        return (key in obj), obj.get(key)
    if hasattr(obj, key):
        return True, getattr(obj, key)
    return False, None


def tenant_filter(result: Any, tenant_id: str, *, key: str = "tenant_id") -> Any:
    """Assert every record in ``result`` belongs to ``tenant_id`` and return it.

    ``result`` may be a mapping, a Pydantic model, an object with the attribute,
    or a list/tuple of those. A record that lacks the key, or whose value differs
    from ``tenant_id``, raises :class:`TenantMismatchError`. Scalars (str, bytes,
    numbers) cannot carry a tenant and are rejected too.
    """
    if not tenant_id:
        raise ValueError("tenant_filter: caller tenant_id must be non-empty")
    if isinstance(result, (str, bytes, int, float, bool)) or result is None:
        raise TenantMismatchError(tenant_id, None, where=f"{type(result).__name__} result")
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        for i, item in enumerate(result):
            present, found = _get_field(item, key)
            if not present or found != tenant_id:
                raise TenantMismatchError(tenant_id, found if present else None, where=f"item {i}")
        return result
    present, found = _get_field(result, key)
    if not present or found != tenant_id:
        raise TenantMismatchError(tenant_id, found if present else None)
    return result


__all__ = [
    "CLOSE_TAG",
    "OPEN_TAG",
    "TenantMismatchError",
    "sanitize_input",
    "strip_control_chars",
    "tenant_filter",
    "unwrap_input",
]
