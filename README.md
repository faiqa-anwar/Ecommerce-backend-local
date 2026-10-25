# E-commerce AI Support Assistant (local LLM)

FastAPI backend + browser UI. Customers chat with an assistant that answers
from your product catalog and policy documents (RAG), remembers the
conversation, and can look up orders / check stock / start returns through
tool calls. The LLM runs **locally with Ollama**, no cloud AI and no API keys.

Features: JWT login (one store per account) - product, order and knowledge-base
management - local embeddings (fastembed) - chat memory - tool calling with
security checks (an order is only shown if the email matches; stores can't see
each other's data) - rate limiting - 71 automated tests.

## What you need
- Python 3.10+  - Ollama (https://ollama.com/download)
- Docker Desktop (for Postgres), or a Supabase/Postgres URL
- Internet **once**: to download the LLM (~2 GB) and the embedding model (~90 MB)

## Run it (Windows PowerShell shown)
1. **Get the model**
   ```
   ollama pull llama3.2
   ```
   (Ollama runs in the background after install. 8 GB+ RAM? `ollama pull llama3.1`
   follows tool instructions better; then set `LLM_MODEL=llama3.1` in .env.)
2. **Start the database**
   ```
   docker compose up -d db
   ```
3. **Install**
   ```
   python -m venv venv
   venv\Scripts\Activate.ps1        # if blocked: Set-ExecutionPolicy -Scope Process Bypass
   pip install -r requirements.txt
   copy .env.example .env           # then set SECRET_KEY inside .env
   ```
4. **Start the app**
   ```
   python -m uvicorn app.main:app
   ```
   The first start downloads the embedding model, so give it a minute.
5. **Open http://localhost:8000** -> *Create account* -> **Load demo data** -> chat.
   API docs for every endpoint: http://localhost:8000/docs

## Things to try in the chat
- "Do you have wireless headphones?"  (catalog search)
- "What's your return policy?"  (knowledge base)
- "Where is order #1?"  with email `customer@example.com`  (order tool)
- "I want to return order #2, it doesn't fit" (return tool; changes the order status)
- Ask a follow-up like "how much are they?"  (conversation memory)
- Orders tab: change a status / stock, then ask the chat again.
- "Copy customer chat link" gives a login-free, chat-only page for customers.

## Tests
```
docker compose exec db createdb -U postgres ecommerce_test_db
pytest -q
```
The tests mock the LLM, so they don't need Ollama running.

## Troubleshooting
- *"Can't reach the local LLM"* - Ollama isn't running: open the Ollama app or run `ollama serve`.
- *"Model 'x' isn't available locally"* - run `ollama pull x`.
- Slow replies - normal on CPU, especially the first one (model loading). Use a smaller model or raise `LLM_REQUEST_TIMEOUT_SECONDS`.
- Model ignores order/stock questions - it may not support tool calling well; try `llama3.1` or `qwen2.5`. (Models with no tool support still work as a knowledge-base chatbot.)
- Search quality poor / log says "using hashing fallback" - the embedding model couldn't download; restart with internet.
- Other OpenAI-compatible local servers (LM Studio, GPT4All's API server) can work by changing `LLM_BASE_URL` and `LLM_MODEL`; Ollama is the one tested here.

## Design notes
- Vector search is brute-force cosine similarity in Python (no pgvector); fine up to a few thousand chunks.
- Tables are created on startup (`create_all`); add Alembic before changing the schema on live data.
- `/chat` rate limit is in-memory (per process); use Redis if you run several instances.
