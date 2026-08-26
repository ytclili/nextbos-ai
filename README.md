# nextbos-ai

LangGraph + FastAPI agent skeleton. Redis stores short-term LangGraph checkpoints; PostgreSQL stores agent runtime state (long-term memories and, later, runtime configuration such as LLM provider settings). Business data is intentionally not modeled here and must be accessed through Tools or external integrations.

```bash
cp .env.example .env
docker compose up -d --build
```

Open <http://localhost:8010/docs>. PostgreSQL schema initialization creates the
agent-owned runtime tables: long-term memories plus LLM model profiles,
provider credentials, and effective model config snapshots. Redis requires
RedisJSON and RediSearch and is provided by the Redis Stack image in
`docker-compose.yml`.

For server deployment:

```bash
git clone <repo-url> nextbos-ai
cd nextbos-ai
cp .env.example .env
```

Edit `.env` and fill at least the LLM and business tool settings:

```env
APP_ENV=prod
LOG_LEVEL=INFO

LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://work.tokenok.net/v1
LLM_MODEL=gpt-5.6-sol
LLM_API_KEY=your-llm-key

CORE_INTERNAL_BASE_URL=http://your-core-service
CORE_INTERNAL_TOKEN=your-mock-token

OTEL_ENABLED=false
OTEL_LOGS_ENABLED=false
```

Then start all services:

```bash
docker compose up -d --build
docker compose logs -f app
```

Check health:

```bash
curl http://127.0.0.1:8010/health
```

Inside Docker Compose, the app connects to Redis/PostgreSQL through service
names (`redis`, `postgres`), so `.env` may keep the local defaults for
`REDIS_URL` and `POSTGRES_DSN`; `docker-compose.yml` overrides them for the app
container.

The current graph calls the configured LangChain chat model and business tools,
so production deployment should provide a valid OpenAI-compatible LLM
configuration in `.env`.

LLM model selection is designed as a three-layer configuration flow:

1. Database model profiles define the provider, request base URL, model name,
   credential reference, default params, status, and default flag.
2. `.env` values remain the fallback when no active database default exists.
3. Each resolved run writes an immutable effective-config snapshot containing
   the selected model metadata and credential reference, but never the API key.

Runtime model calls go through LangChain chat model integrations, keeping
provider protocol details out of the agent graph.
