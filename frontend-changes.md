# Frontend Changes

No frontend files were modified in this feature.

This feature added backend testing infrastructure only:

- `backend/tests/conftest.py` — expanded with shared fixtures (`mock_rag_system`, `test_app`, `client`) used by the new API endpoint tests
- `backend/tests/test_api_endpoints.py` — 22 new tests covering POST /api/query, GET /api/courses, and DELETE /api/session/{id}
- `pyproject.toml` — added `[tool.pytest.ini_options]` with `testpaths`, `addopts`, and `markers` configuration
