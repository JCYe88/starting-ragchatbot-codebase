# Frontend Changes: Dark/Light Theme Toggle

## Files Modified

- `frontend/index.html`
- `frontend/style.css`
- `frontend/script.js`

---

## index.html

- Bumped cache-busting version from `v=9` to `v=10` on `style.css` and `script.js` imports.
- Added a `<button id="themeToggle">` element directly inside `<body>` (before the scripts). Contains two inline SVGs: `.icon-sun` (shown in dark mode) and `.icon-moon` (shown in light mode). Marked with `aria-hidden` since the button itself carries the accessible label.

---

## style.css

### New / updated CSS variables in `:root` (dark theme)
Added variables for previously hardcoded values:
- `--code-bg` — background for `<code>` and `<pre>` blocks
- `--source-link-border`, `--source-link-color`, `--source-link-bg`, `--source-link-hover-color`, `--source-link-hover-bg` — source chip link colours
- `--toggle-bg`, `--toggle-hover-bg`, `--toggle-color` — theme toggle button appearance

### New `[data-theme="light"]` block
Overrides all colour variables for the light theme:
- Background: `#f8fafc` (near-white), surface: `#ffffff`
- Text: `#0f172a` (near-black) primary, `#64748b` secondary
- Border: `#e2e8f0` (light grey)
- Source chip link colour adjusted to `#1e40af` (dark blue) for contrast on white
- Toggle button uses lighter greys

### Hardcoded colours replaced with CSS variables
- `rgba(0,0,0,0.2)` in `.message-content code` and `pre` → `var(--code-bg)`
- Hex/rgba colour literals in `a.source-chip` and `a.source-chip:hover` → `var(--source-link-*)` variables

### Theme transition helper class
```css
.theme-transitioning,
.theme-transitioning * {
    transition: background-color 0.3s ease, color 0.25s ease,
                border-color 0.3s ease, box-shadow 0.3s ease !important;
}
```
Applied only during the toggle click (removed after 350 ms) to avoid interfering with existing animations like the loading-dots bounce.

### Theme toggle button styles (`.theme-toggle`)
- `position: fixed; top: 1rem; right: 1rem; z-index: 100`
- 40 × 40 px circle, uses `--toggle-bg` / `--toggle-color` variables
- Focus ring via `box-shadow: 0 0 0 3px var(--focus-ring)` (keyboard-accessible)
- `.icon-moon` hidden by default; swapped via `[data-theme="light"]` selector (no JS class toggling needed)

---

## script.js

- Bumped version reference in `<script src>` (handled in HTML).
- Added `setupThemeToggle()` called on `DOMContentLoaded` before other setup.
  - Reads `localStorage.getItem('theme')` (defaults to `'dark'`).
  - Sets `data-theme` attribute on `<html>` element.
  - On click: adds `.theme-transitioning` to `<body>`, toggles `data-theme`, saves to `localStorage`, removes the transition class after 350 ms.
- Added `updateToggleLabel(theme)` — updates `aria-label` on the button to reflect what clicking will do ("Switch to light/dark theme").
