"""
ui/glyphs.py
=============

Terminal-safe symbol selection.

Problem this solves:
    Windows' legacy console (cmd.exe / conhost without UTF-8 mode) often
    defaults to a non-UTF-8 codepage (e.g. CP1252). Any Unicode box-drawing
    or symbol character PyClaw prints -- borders, checkmarks, arrows,
    warning signs -- comes back from that terminal as "?" or a broken box
    glyph, because the terminal's output encoding can't represent the
    character at all. This is a terminal/encoding limitation, not
    something Rich or Textual can paper over by themselves if the
    underlying stream is misconfigured.

Fix:
    Detect whether stdout can actually encode UTF-8 (and, on Windows,
    whether the console codepage is UTF-8) at startup, once. If not, every
    symbol PyClaw prints anywhere falls back to a plain ASCII equivalent
    instead of a Unicode character. Modules that print symbols call
    `glyph("check")` etc. instead of hardcoding "\u2713" directly, so the
    fallback is applied consistently everywhere.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from typing import Dict

# Each symbol has a preferred Unicode form and a plain-ASCII fallback.
# Keys are stable names used throughout the codebase; values are
# (unicode_form, ascii_fallback) pairs.
_SYMBOLS: Dict[str, tuple] = {
    "check": ("\u2713", "[x]"),
    "box_empty": ("\u25a1", "[ ]"),
    "arrow": ("\u2192", "->"),
    "bullet": ("\u25cf", "*"),
    "warning": ("\u26a0", "!"),
    "border_h": ("\u2500", "-"),
    "border_v": ("\u2502", "|"),
    "border_tl": ("\u256d", "+"),
    "border_tr": ("\u256e", "+"),
    "border_bl": ("\u2570", "+"),
    "border_br": ("\u256f", "+"),
}


@lru_cache(maxsize=1)
def supports_unicode() -> bool:
    """Return True if the current stdout stream can safely encode common
    Unicode symbols, False if it should fall back to plain ASCII.

    Cached after first call since terminal capability does not change
    mid-process. The check is intentionally conservative: any uncertainty
    (no encoding reported, encode failure, exception) resolves to False so
    a misdetection degrades to extra "?" characters at worst -- not the
    other way around, which would risk a crash on write.
    """
    encoding = getattr(sys.stdout, "encoding", None)
    if not encoding:
        return False

    normalized = encoding.lower().replace("-", "").replace("_", "")
    if "utf8" not in normalized and "utf" not in normalized:
        return False

    # Even when Python reports a UTF-8 stream encoding, legacy Windows
    # consoles can still be backed by a non-UTF-8 codepage that silently
    # mangles output. Confirm by actually trying to encode one of our
    # symbols the way the stream would.
    try:
        sample = "\u2713\u2500\u2192"
        sample.encode(encoding, errors="strict")
    except (UnicodeEncodeError, LookupError):
        return False

    return True


def safe_box():
    """Return the Rich `box` style appropriate for the current terminal.

    Rich's default Panel/Table border (`box.ROUNDED`) uses Unicode
    box-drawing characters. On a terminal that can't render Unicode (see
    supports_unicode()), those come back as "?" -- which is exactly the
    "random bar that looks like garbled boxes" artifact this module exists
    to prevent. `box.ASCII` draws panels using only `-`, `|`, `+`, which
    is always safe.

    Usage: `Panel(body, box=safe_box())` instead of leaving Panel's
    default box style implicit.
    """
    from rich import box as rich_box

    return rich_box.ROUNDED if supports_unicode() else rich_box.ASCII


def glyph(name: str) -> str:
    """Return the appropriate form of a named symbol for the current
    terminal: the Unicode glyph if supported, otherwise an ASCII fallback.

    Unknown names return an empty string rather than raising, so a typo'd
    glyph name degrades silently instead of crashing UI rendering.
    """
    pair = _SYMBOLS.get(name)
    if pair is None:
        return ""
    unicode_form, ascii_form = pair
    return unicode_form if supports_unicode() else ascii_form
