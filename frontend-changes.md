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
