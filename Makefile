.PHONY: install lint test train api clean

install:        ## instala dependências (instala do zero)
	poetry install

lint:           ## ruff sem erros
	poetry run ruff check src tests

test:           ## roda os testes
	poetry run pytest -q

train:          ## treina a MLP e registra no MLflow
	poetry run python -m churn.train

api:            ## sobe a API de inferência
	poetry run uvicorn churn.api.main:app --reload

clean:          ## remove caches e artefatos do MLflow local
	rm -rf .pytest_cache .ruff_cache mlruns mlartifacts
