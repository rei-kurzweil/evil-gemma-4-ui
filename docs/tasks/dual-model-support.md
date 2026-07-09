# Dual Model Support: Gemma 4 + Llama 3.1

## Scope

Add support for loading and serving **two** GGUF models side-by-side:

| Model ID | File | Vision | Stop Tokens |
|---|---|---|---|
| `gemma-4-local` | `Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q6_K_P.gguf` | Yes (mmproj) | `DEFAULT_STOP` list |
| `llama-3.1-local` | `Llama-3.1-8B-Lexi-Uncensored-Q6_K_L.gguf` | No | None (GGUF template) |

## File changes

### 1. Rename `vision_inference.py` → `inference.py`

- `git mv vision_inference.py inference.py`
- Update `import` in `app.py` from `vision_inference` to `inference`

### 2. `inference.py` — refactor model loading

- Remove module-level `MODEL_PATH`, `MMPROJ_PATH`, `model_instance` singleton
- Add constants for both models:

```python
MODELS = {
    "gemma-4-local": {
        "model_path": "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q6_K_P.gguf",
        "mmproj_path": "mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf",
    },
    "llama-3.1-local": {
        "model_path": "Llama-3.1-8B-Lexi-Uncensored-Q6_K_L.gguf",
        "mmproj_path": None,
    },
}
```

- Replace `get_model()` with `get_models()` — loads **all** models at once, returns `dict[str, GemmaVisionModel]`.

### 3. `GemmaVisionModel.__init__` — optional vision

- `mmproj_path` defaults to `None`.
- If `mmproj_path is None`: skip creating `DebugLlava15ChatHandler`, set internal flag `self._supports_vision = False`.
- `capabilities()` — return `supports_vision` based on `self._supports_vision`.

### 4. `create_chat_completion` — vision validation

- After `_prepare_messages` and before `_select_chat_runtime`, check if any message has image content.
- If model has no vision and images are present: raise `ValueError("Model does not support vision.")`.

### 5. `_select_chat_runtime` — handle no-vision model

- If `not self._supports_vision`: always set `chat_handler = None`, restore `chat_format`. Skip image scanning.

### 6. Stop tokens

- Gemma: keeps existing `DEFAULT_STOP` list.
- Llama: passes `stop=None` from app.py (no stop tokens — GGUF template handles stopping via `<|eot_id|>` / `<|end_of_text|>`).

### 7. `app.py` — serve both models

- `get_models()` called at startup; result stored as `models: dict`.
- `/v1/models` iterates `models.items()`, returns one `data` entry per model with its `capabilities()`.
- `chat_completions()` looks up `normalized["model"]` in `models` dict. If not found → `400` "Unknown model".
- Pass the selected model wrapper to `create_chat_completion` and `stream_completion`.
- `/chat` legacy endpoint hardcodes `"gemma-4-local"`.

## Key design decisions

- Vision messages sent to Llama → 400 error with descriptive message.
- Each model keeps its own GGUF embedded chat template (Jinja2). No manual prompt formatting needed.
- `/chat` stays on Gemma 4 for backward compatibility.
- Llama uses no custom stop tokens — the GGUF template's built-in `<|eot_id|>` stops generation naturally.

## Testing

1. Start server — verify both models load without error.
2. `GET /v1/models` — returns 2 entries with correct capabilities.
3. `POST /v1/chat/completions` with `model: "gemma-4-local"` — works as before.
4. `POST /v1/chat/completions` with `model: "llama-3.1-local"` — generates with Llama format.
5. `POST /v1/chat/completions` with image + `"llama-3.1-local"` — 400 error.
6. `POST /v1/chat/completions` with `model: "nonexistent"` — 400 error.
7. `POST /chat` — still routes to Gemma 4.