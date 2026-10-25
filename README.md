# E-commerce AI Support Assistant (local LLM)

FastAPI backend + browser UI. Customers chat with an assistant that answers
from your product catalog and policy documents (RAG), remembers the
conversation, and can look up orders / check stock / start returns through
tool calls. The LLM runs **locally with Ollama** (llama3.2)

Features: JWT login (one store per account) - product, order and knowledge-base
management - local embeddings (fastembed) - chat memory - tool calling with
security checks (an order is only shown if the email matches; stores can't see
each other's data) 

## What you need
- Python 3.10+  - Ollama (https://ollama.com/download)
- Docker Desktop (for Postgres), or a Supabase/Postgres URL
- Internet **once**: to download the LLM and the embedding model

## Run it (Windows PowerShell shown)
1. **Get the model**
   ```
   ollama pull llama3.2
   ```
   (Ollama runs in the background after install.  `ollama pull llama3.1`
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

