# Draft: Streaming Responses Implementation

**Author:** Gemini CLI
**Date:** May 20, 2026

## Objective
Implement real-time streaming of model responses to the Web UI to improve perceived performance and user experience.

## Proposed Solution

### 1. Backend (Flask)
Update the `/chat` endpoint to use a generator function and Flask's `Response` object with `mimetype='text/event-stream'`.

- **Module:** `vision_inference.py`
  - Modify `generate_response` to accept a `stream=True` parameter.
  - When `stream=True`, return the generator provided by `self.llm.create_chat_completion`.
- **Module:** `app.py`
  - Create a generator function that yields data in the Server-Sent Events (SSE) format (e.g., `data: <content>\n\n`).

### 2. Frontend (JavaScript)
Update `static/script.js` to handle the `ReadableStream` from the `fetch` API.

- **Changes:**
  - Use `fetch('/chat', { ... })` and then iterate over `response.body.getReader()`.
  - Append chunks to the AI message bubble in real-time.
  - Automatically scroll the chat window as new text arrives.

## Incremental Implementation Steps

1.  **Refactor `vision_inference.py`**: Add support for streaming in the `generate_response` method.
2.  **Update `app.py`**: Create the streaming route and generator.
3.  **Update `static/script.js`**: Implement the `ReadableStream` reader logic.
4.  **Polish UI**: Add a "typing" indicator or subtle animation for the streaming text.

## Alternatives Considered
- **WebSockets**: Overkill for a simple chatbot and requires more complex state management.
- **Short Polling**: Poor performance and higher latency.

## Decision
Server-Sent Events (SSE) via Flask Generators is the most idiomatic and efficient way to handle LLM streaming in this architecture.
