# nextbos-ai

LangGraph + FastAPI agent skeleton. Redis stores short-term LangGraph checkpoints; PostgreSQL stores agent runtime state (long-term memories and, later, runtime configuration such as LLM provider settings). Business data is intentionally not modeled here and must be accessed through Tools or external integrations.

```bash
cp .env.example .env
uv sync --group dev
docker compose up -d
PYTHONPATH=src uv run uvicorn app.main:app --reload
```

Open <http://localhost:8000/docs>. PostgreSQL schema initialization creates only
`agent_memories`; Redis requires RedisJSON and RediSearch and is provided by the
Redis Stack image in `docker-compose.yml`.

The current graph is a deterministic starter node (`已收到：...`) so it runs
without an LLM key. Add model and business integrations behind `app/llm` and
`app/tools/business` when requirements are defined.
