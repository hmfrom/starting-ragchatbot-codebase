# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Always use `uv` for package management and running Python files. Do not use pip or python directly.**

```bash
# Run any Python file
uv run python <file.py>

# Run a module
uv run python -m <module>
```

```bash
# Install dependencies
uv sync

# Run the application (from project root)
./run.sh
# Or manually:
cd backend && uv run uvicorn app:app --reload --port 8000

# Access points
# Web UI: http://localhost:8000
# API docs: http://localhost:8000/docs
```

## Environment Setup

Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_key_here
```

## Architecture Overview

This is a RAG (Retrieval-Augmented Generation) chatbot that answers questions about course materials using semantic search and OpenAI AI.

### Request Flow

1. **Frontend** (`frontend/`) - Static HTML/JS/CSS served by FastAPI
   - `script.js:sendMessage()` sends queries to `/api/query`

2. **API Layer** (`backend/app.py`) - FastAPI endpoints
   - `POST /api/query` - Main query endpoint
   - `GET /api/courses` - Course statistics

3. **Orchestration** (`backend/rag_system.py`) - `RAGSystem` coordinates all components:
   - Manages document processing, vector storage, AI generation, and sessions
   - `query()` method orchestrates the full RAG pipeline

4. **AI Generation** (`backend/ai_generator.py`) - `AIGenerator` handles OpenAI API calls:
   - Uses tool calling (`tool_choice: auto`) for search decisions
   - `_handle_tool_execution()` processes tool use responses

5. **Search Tools** (`backend/search_tools.py`) - Tool definitions for OpenAI:
   - `CourseSearchTool` - semantic search over course content
   - `ToolManager` - registers and executes tools

6. **Vector Storage** (`backend/vector_store.py`) - ChromaDB integration:
   - Two collections: `course_catalog` (metadata) and `course_content` (chunks)
   - Uses SentenceTransformer (`all-MiniLM-L6-v2`) for embeddings

7. **Document Processing** (`backend/document_processor.py`) - Parses course files:
   - Extracts course title, instructor, lessons from text files
   - Chunks content (800 chars, 100 overlap)

8. **Session Management** (`backend/session_manager.py`) - Conversation context:
   - Maintains history per session (max 2 exchanges by default)

### Key Configuration (`backend/config.py`)

- `OPENAI_MODEL`: gpt-5-nano
- `EMBEDDING_MODEL`: all-MiniLM-L6-v2
- `CHUNK_SIZE`: 800 characters
- `MAX_RESULTS`: 5 search results
- `MAX_HISTORY`: 2 conversation turns

### Data Storage

- ChromaDB persists to `backend/chroma_db/`
- Course documents loaded from `docs/` folder on startup
