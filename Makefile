.PHONY: install dev test lint infra-up infra-down
install:
	uv sync --group dev
dev:
	PYTHONPATH=src uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
test:
	uv run pytest
lint:
	uv run ruff check .
infra-up:
	docker compose up -d
infra-down:
	docker compose down
