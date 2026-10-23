.PHONY: install test lint benchmark dashboard api
install:
	pip install -e ".[dev]"
test:
	pytest
lint:
	ruff check .
benchmark:
	drugbench run --config configs/default.yaml
dashboard:
	streamlit run dashboard/app.py
api:
	uvicorn api.main:app --reload
