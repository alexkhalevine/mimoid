.PHONY: dev install sidecar-venv sidecar-dev build clean eval

SIDECAR_VENV := sidecar/.venv

# Full app dev loop: Tauri spawns and supervises the sidecar itself
# (see src-tauri/src/sidecar.rs), so this just needs the venv to exist.
dev: $(SIDECAR_VENV)
	npm run tauri dev

install: $(SIDECAR_VENV)
	npm install

$(SIDECAR_VENV):
	python3 -m venv $(SIDECAR_VENV)
	$(SIDECAR_VENV)/bin/pip install -q -r sidecar/requirements.txt

# Run the sidecar standalone (with hot reload) when iterating on it in isolation.
sidecar-dev: $(SIDECAR_VENV)
	cd sidecar && ../$(SIDECAR_VENV)/bin/uvicorn app.main:app --port 8756 --reload

# Runs the fixed prompt bank through the persona pipeline and scores replies
# with deterministic gates + a local LLM-as-judge. Requires Ollama running.
eval: $(SIDECAR_VENV)
	cd sidecar && ../$(SIDECAR_VENV)/bin/python -m app.eval_runner

build:
	npm run tauri build

clean:
	rm -rf $(SIDECAR_VENV) dist src-tauri/target node_modules
