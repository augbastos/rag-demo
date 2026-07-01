.PHONY: up schema ingest test down

up:
	docker compose up -d

schema:
	psql "$$DATABASE_URL" -f schema.sql

ingest:
	python ingest.py

test:
	pytest -q

down:
	docker compose down
