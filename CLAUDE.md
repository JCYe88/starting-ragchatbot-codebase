# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Management

Always use `uv` — never `pip` directly.

## Running the Application

```bash
# Install dependencies
uv sync

# Set up environment
cp .env.example .env  # then add ANTHROPIC_API_KEY

# Start server (from repo root)
./run.sh

# Or manually (must run from backend/ for relative paths to resolve)
cd backend && uv run uvicorn app:app --reload --port 8000
```

App runs at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

`main.py` at the repo root is unused — the real entrypoint is `backend/app.py`.

## Architecture

Full-stack RAG chatbot. FastAPI serves both the JSON API and the static frontend from a single process. On startup, course documents from `docs/` are ingested into ChromaDB.

**Query flow:**
1. Frontend POSTs to `/api/query` with `{query, session_id}`
2. `RAGSystem` prepends conversation history and calls Claude with the `search_course_content` tool available
3. Claude optionally calls the tool → `CourseSearchTool` runs semantic search in ChromaDB → results returned to Claude
4. Claude synthesizes a final answer; sources are pulled from `ToolManager` and conversation history is saved to `SessionManager`

**Two ChromaDB collections** (stored at `backend/chroma_db/`):
- `course_catalog` — one doc per course (title, instructor, link, lessons serialized as JSON string)
- `course_content` — chunked lesson text with `course_title` and `lesson_number` metadata for filtering

**Session state is in-memory only** — all conversation history is lost on server restart.

**Course document format** (files in `docs/`):
```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>
Lesson 0: <title>
Lesson Link: <url>
<content...>
Lesson 1: <title>
...
```

## Key Configuration (`backend/config.py`)

| Setting | Default | Notes |
|---|---|---|
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Change here to swap models |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Shared by both ChromaDB collections |
| `CHUNK_SIZE` | 800 | Characters per chunk |
| `CHUNK_OVERLAP` | 100 | Overlap between chunks |
| `MAX_RESULTS` | 5 | Search results returned to Claude |
| `MAX_HISTORY` | 2 | Conversation exchanges kept per session |

## Extending the System

**Add a new tool:** Subclass `Tool` in `backend/search_tools.py`, implement `get_tool_definition()` and `execute()`, then register with `tool_manager.register_tool(your_tool)` in `RAGSystem.__init__`.

**Add course documents:** Drop `.txt`, `.pdf`, or `.docx` files into `docs/`. They are ingested on next startup; existing courses (matched by title) are skipped.

**Rebuild the vector DB:** Delete `backend/chroma_db/` and restart, or call `add_course_folder(path, clear_existing=True)`.
