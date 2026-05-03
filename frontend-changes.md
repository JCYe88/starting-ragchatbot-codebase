# Frontend Changes

## Code Quality Tooling

### New files

| File | Purpose |
|---|---|
| `frontend/package.json` | npm project manifest; declares Prettier and ESLint as dev dependencies and defines npm scripts |
| `frontend/.prettierrc` | Prettier configuration (2-space indent, single quotes, trailing commas, LF line endings) |
| `frontend/eslint.config.js` | ESLint flat config for `script.js`; enforces `no-var`, `prefer-const`, `eqeqeq`, and no undefined globals |
| `frontend/.gitignore` | Excludes `node_modules/` from version control |
| `scripts/frontend-quality.sh` | Root-level shell script to run all quality checks (or auto-fix) from any working directory |

### npm scripts (run from `frontend/`)

| Script | Command | What it does |
|---|---|---|
| `format` | `prettier --write` | Auto-format HTML, CSS, JS in place |
| `format:check` | `prettier --check` | Check formatting without modifying files (CI-safe) |
| `lint` | `eslint script.js` | Lint JS for errors and style issues |
| `lint:fix` | `eslint --fix script.js` | Auto-fix fixable lint issues |
| `quality` | `format:check && lint` | Full read-only check (use in CI) |
| `quality:fix` | `format && lint:fix` | Auto-fix everything |

### Root-level script

```bash
./scripts/frontend-quality.sh        # check-only (exits non-zero on failure)
./scripts/frontend-quality.sh --fix  # auto-fix formatting + linting
```

The script installs `node_modules` automatically if missing, so it works in a fresh checkout.

### Formatting applied to existing files

Prettier was run on all three existing frontend files. Changes were stylistic only — no logic was altered:

- **`index.html`** — indentation changed from 4-space to 2-space, void elements gained self-closing slashes (`<meta ... />`), long attribute lines were wrapped
- **`style.css`** — minor whitespace normalisation (trailing spaces, consistent blank lines)
- **`script.js`** — double-quoted strings converted to single quotes, trailing commas added to multi-line structures

## Backend Testing Infrastructure

No frontend files were modified in this feature.

This feature added backend testing infrastructure only:

- `backend/tests/conftest.py` — expanded with shared fixtures (`mock_rag_system`, `test_app`, `client`) used by the new API endpoint tests
- `backend/tests/test_api_endpoints.py` — 22 new tests covering POST /api/query, GET /api/courses, and DELETE /api/session/{id}
- `pyproject.toml` — added `[tool.pytest.ini_options]` with `testpaths`, `addopts`, and `markers` configuration

## Dark/Light Theme Toggle

### Files Modified

- `frontend/index.html`
- `frontend/style.css`
- `frontend/script.js`

### index.html

- Bumped cache-busting version from `v=9` to `v=10` on `style.css` and `script.js` imports.
- Added a `<button id="themeToggle">` element directly inside `<body>` (before the scripts). Contains two inline SVGs: `.icon-sun` (shown in dark mode) and `.icon-moon` (shown in light mode). Marked with `aria-hidden` since the button itself carries the accessible label.

### style.css

#### New / updated CSS variables in `:root` (dark theme)
Added variables for previously hardcoded values:
- `--code-bg` — background for `<code>` and `<pre>` blocks
- `--source-link-border`, `--source-link-color`, `--source-link-bg`, `--source-link-hover-color`, `--source-link-hover-bg` — source chip link colours
- `--toggle-bg`, `--toggle-hover-bg`, `--toggle-color` — theme toggle button appearance

#### New `[data-theme="light"]` block
Overrides all colour variables for the light theme:
- Background: `#f8fafc` (near-white), surface: `#ffffff`
- Text: `#0f172a` (near-black) primary, `#64748b` secondary
- Border: `#e2e8f0` (light grey)
- Source chip link colour adjusted to `#1e40af` (dark blue) for contrast on white
- Toggle button uses lighter greys

#### Hardcoded colours replaced with CSS variables
- `rgba(0,0,0,0.2)` in `.message-content code` and `pre` → `var(--code-bg)`
- Hex/rgba colour literals in `a.source-chip` and `a.source-chip:hover` → `var(--source-link-*)` variables

#### Theme transition helper class
Applied only during the toggle click (removed after 350 ms) to avoid interfering with existing animations like the loading-dots bounce.

#### Theme toggle button styles (`.theme-toggle`)
- `position: fixed; top: 1rem; right: 1rem; z-index: 100`
- 40 × 40 px circle, uses `--toggle-bg` / `--toggle-color` variables
- Focus ring via `box-shadow: 0 0 0 3px var(--focus-ring)` (keyboard-accessible)
- `.icon-moon` hidden by default; swapped via `[data-theme="light"]` selector (no JS class toggling needed)

### script.js

- Added `setupThemeToggle()` called on `DOMContentLoaded` before other setup.
  - Reads `localStorage.getItem('theme')` (defaults to `'dark'`).
  - Sets `data-theme` attribute on `<html>` element.
  - On click: adds `.theme-transitioning` to `<body>`, toggles `data-theme`, saves to `localStorage`, removes the transition class after 350 ms.
- Added `updateToggleLabel(theme)` — updates `aria-label` on the button to reflect what clicking will do ("Switch to light/dark theme").
