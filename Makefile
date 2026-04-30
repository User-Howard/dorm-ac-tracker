.PHONY: install uv-install venv sync

install: uv-install venv sync

uv-install:
	@command -v uv >/dev/null 2>&1 || { \
		echo "uv not found, installing..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	}

venv:
	uv venv --seed --system-site-packages

sync:
	uv sync
