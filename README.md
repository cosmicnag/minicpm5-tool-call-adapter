# MiniCPM5 tool-call adapter

A small proxy for running **MiniCPM5-1B** with **llama.cpp** while converting the model's native XML tool calls into OpenAI-compatible `tool_calls`.

## What it does

- accepts OpenAI-style chat/completions requests
- forwards them to `llama-server`
- rewrites MiniCPM5 XML tool blocks like `<function ...>` / `<param ...>` into JSON `tool_calls`
- works as a standalone proxy or as a backend launched by **llama-swap**

## Prerequisites

- Python 3.12+
- `llama-server` from llama.cpp
- a MiniCPM5-1B GGUF model
- the matching MiniCPM5 Jinja chat template
- CUDA / NVIDIA GPU if you want local GPU inference

## Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick start

Set the model and template paths, then start the proxy:

```bash
export MINICPM5_MODEL_PATH=/path/to/minicpm5-1b.gguf
export MINICPM5_TEMPLATE_PATH=/path/to/chat_template.jinja
./minicpm5_adapter.sh --port 10001
```

The service listens on `http://127.0.0.1:10001/v1`.

## Usage scenarios

### 1) Standalone proxy + llama.cpp

Use this when you want the adapter to manage its own `llama-server` process.

```bash
./minicpm5_adapter.sh --port 10001
```

### 2) Existing llama-server upstream

Use this when `llama-server` is already running somewhere else.

```bash
export MINICPM5_UPSTREAM_URL=http://127.0.0.1:8001
./minicpm5_adapter.sh --port 10001
```

### 3) llama-swap backend

Use this when you want llama-swap to own startup, scheduling, and model lifecycle.

```yaml
command: /path/to/minicpm5_adapter.sh --port 10001
```

Point the llama-swap model entry at the launcher script, and keep the adapter as the OpenAI-compatible backend.

## Launcher script

`minicpm5_adapter.sh` is a barebones launcher that:

- prefers `.venv/bin/python`
- falls back to `python3`
- sets `HF_HOME` and `LLAMA_CACHE` to the repo by default
- sets `PYTHONPATH` to `src/`
- leaves the model/template paths to env vars or CLI flags

You can override the GPU with `CUDA_VISIBLE_DEVICES`.

## Configuration

### Required

- `MINICPM5_MODEL_PATH` / `--model-path`
- `MINICPM5_TEMPLATE_PATH` / `--template-path`
- `--port`

### Optional

- `MINICPM5_UPSTREAM_URL` / `--upstream-url`
- `MINICPM5_LLAMABIN` / `--llama-bin`
- `MINICPM5_MODEL_NAME` / `--served-model-name`
- `MINICPM5_REASONING=1` / `--reasoning`
- `MINICPM5_LOG_PATH` / `--log-path`
- `MINICPM5_CHAT_TEMPLATE_KWARGS` / `--chat-template-kwargs`
- `MINICPM5_LLAMA_EXTRA_ARGS` / `--llama-extra-args`

## Development

Run the tests:

```bash
python -m unittest discover -s tests -v
```

## License

MIT
