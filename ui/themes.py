"""
ui/themes.py
=============

Named color themes for the PyClaw Textual UI.

Each theme is a plain dict of named colors (not tied to Textual's Theme
class directly) so that:
    * `/theme list` can print every color swatch as readable text/hex
      values, even outside Textual (e.g. for documentation or the simple
      REPL), satisfying "I can see the color of the theme."
    * Building a `textual.theme.Theme` from a dict is a one-line
      conversion, so the actual live re-theming stays simple.

Colors follow a small, consistent set of roles used throughout ui/tui.py's
CSS (`$primary`, `$secondary`, `$success`, `$warning`, `$error`,
`$background`, `$surface`, `$foreground`, `$accent`), matching Textual's
own design-token naming so they drop straight into the existing CSS
without rewriting every selector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ThemeDef:
    """A named PyClaw theme: a human label plus a role->hex color mapping."""

    name: str
    label: str
    dark: bool
    colors: Dict[str, str]

    def swatch_lines(self) -> List[str]:
        """Render each color role as a readable 'role: #hexcode' line, so
        the actual colors are visible as text -- useful in the REPL, in
        `/theme list`, or anywhere true color swatches can't be drawn."""
        return [f"  {role:<12} {hexcode}" for role, hexcode in self.colors.items()]


# ----------------------------------------------------------------------
# Theme registry
# ----------------------------------------------------------------------
THEMES: Dict[str, ThemeDef] = {
    "default-dark": ThemeDef(
        name="default-dark",
        label="Default Dark",
        dark=True,
        colors={
            "primary": "#5e9cf5",
            "secondary": "#7c5cf5",
            "accent": "#f5c542",
            "success": "#3ddc84",
            "warning": "#f5a623",
            "error": "#f55c5c",
            "background": "#101418",
            "surface": "#181c22",
            "foreground": "#e6e6e6",
        },
    ),
    "default-light": ThemeDef(
        name="default-light",
        label="Default Light",
        dark=False,
        colors={
            "primary": "#2563eb",
            "secondary": "#7c3aed",
            "accent": "#d97706",
            "success": "#16a34a",
            "warning": "#d97706",
            "error": "#dc2626",
            "background": "#fafafa",
            "surface": "#ffffff",
            "foreground": "#1a1a1a",
        },
    ),
    "midnight": ThemeDef(
        name="midnight",
        label="Midnight (deep blue/black, easy on the eyes at night)",
        dark=True,
        colors={
            "primary": "#3b82f6",
            "secondary": "#1e3a8a",
            "accent": "#22d3ee",
            "success": "#34d399",
            "warning": "#fbbf24",
            "error": "#f87171",
            "background": "#05070d",
            "surface": "#0d1117",
            "foreground": "#c9d1d9",
        },
    ),
    "forest": ThemeDef(
        name="forest",
        label="Forest (green/brown, calm and grounded)",
        dark=True,
        colors={
            "primary": "#4ade80",
            "secondary": "#65a30d",
            "accent": "#facc15",
            "success": "#22c55e",
            "warning": "#eab308",
            "error": "#ef4444",
            "background": "#0f1411",
            "surface": "#161d18",
            "foreground": "#e3e8e0",
        },
    ),
    "pink": ThemeDef(
        name="pink",
        label="Pink (cute light pink theme)",
        dark=False,
        colors={
            "primary": "#ff6fa5",
            "secondary": "#ff9ecb",
            "accent": "#ffb6d9",
            "success": "#4caf82",
            "warning": "#ffa6c9",
            "error": "#e8537a",
            "background": "#fff0f6",
            "surface": "#ffe3ef",
            "foreground": "#5c2a3d",
        },
    ),
    "pink-dark": ThemeDef(
        name="pink-dark",
        label="Pink Dark (cute pink, but easier on the eyes at night)",
        dark=True,
        colors={
            "primary": "#ff8fb8",
            "secondary": "#d6669a",
            "accent": "#ffc1dd",
            "success": "#5fd49a",
            "warning": "#ffb3cf",
            "error": "#ff6b8e",
            "background": "#1a1117",
            "surface": "#241620",
            "foreground": "#f7d9e6",
        },
    ),
    "high-contrast": ThemeDef(
        name="high-contrast",
        label="High Contrast (max readability, small/low-quality screens)",
        dark=True,
        colors={
            "primary": "#ffffff",
            "secondary": "#00ffff",
            "accent": "#ffff00",
            "success": "#00ff00",
            "warning": "#ffaa00",
            "error": "#ff0000",
            "background": "#000000",
            "surface": "#000000",
            "foreground": "#ffffff",
        },
    ),
}

DEFAULT_THEME_NAME = "default-dark"


def get_theme(name: str) -> ThemeDef:
    """Look up a theme by name, raising KeyError with a helpful message if
    it doesn't exist (callers should catch this and list THEMES.keys())."""
    key = name.strip().lower()
    if key not in THEMES:
        raise KeyError(f"Unknown theme '{name}'. Available: {', '.join(THEMES.keys())}")
    return THEMES[key]


def list_themes_text() -> str:
    """Render every theme as a readable block of names + color swatches,
    used by `/theme list` in both the TUI and the simple REPL."""
    lines = []
    for theme in THEMES.values():
        lines.append(f"{theme.name} -- {theme.label}")
        lines.extend(theme.swatch_lines())
        lines.append("")
    lines.append("Switch with: /theme set <name>")
    return "\n".join(lines).rstrip()
