from flask import Flask, Response, jsonify, render_template, request
from analyzer import consume_complete_sentences
from demultiplexer import Demultiplexer
from vision_inference import get_model
import json
import time
import uuid

app = Flask(__name__)

DEFAULT_MODEL_NAME = "gemma-4-local"

print("Pre-loading Gemma-4 model...")
try:
    llm_wrapper = get_model()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    llm_wrapper = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    status = 200 if llm_wrapper is not None else 503
    return jsonify({"ok": llm_wrapper is not None}), status


@app.route("/v1/models")
def list_models():
    return jsonify(
        {
            "object": "list",
            "data": [
                {
                    "id": DEFAULT_MODEL_NAME,
                    "object": "model",
                    "owned_by": "local",
                }
            ],
        }
    )


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    if llm_wrapper is None:
        return openai_error("Model not loaded on server.", 503)

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return openai_error("Expected a JSON object request body.", 400)

    try:
        normalized = normalize_chat_request(payload)
    except ValueError as exc:
        return openai_error(str(exc), 400)

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    model_name = normalized["model"]

    if normalized["stream"]:
        return Response(
            stream_completion(normalized, completion_id, created, model_name),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        response = llm_wrapper.create_chat_completion(
            messages=normalized["messages"],
            stream=False,
            max_tokens=normalized["max_tokens"],
            temperature=normalized["temperature"],
            stop=normalized["stop"],
            model=model_name,
        )
    except Exception as exc:
        return openai_error(f"Completion failed: {exc}", 500)

    return jsonify(
        normalize_non_stream_response(response, completion_id, created, model_name)
    )


@app.route("/chat", methods=["POST"])
def chat():
    if llm_wrapper is None:
        return jsonify({"response": "Error: Model not loaded on server."}), 500

    data = request.get_json(silent=True) or {}
    try:
        payload = legacy_chat_payload_to_openai(data)
    except ValueError as exc:
        return jsonify({"response": f"Error: {exc}"}), 400

    with app.test_request_context(
        "/v1/chat/completions",
        method="POST",
        json=payload,
    ):
        return chat_completions()


def openai_error(message, status_code):
    return (
        jsonify(
            {
                "error": {
                    "message": message,
                    "type": "invalid_request_error"
                    if status_code < 500
                    else "server_error",
                }
            }
        ),
        status_code,
    )


def normalize_chat_request(payload):
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("`messages` must be a non-empty array.")

    normalized_messages = [normalize_message(message) for message in messages]
    model_name = payload.get("model") or DEFAULT_MODEL_NAME
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("`model` must be a non-empty string when provided.")

    stream = payload.get("stream", False)
    if not isinstance(stream, bool):
        raise ValueError("`stream` must be a boolean.")

    max_tokens = payload.get("max_tokens", 1024)
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("`max_tokens` must be a positive integer.")

    temperature = payload.get("temperature", 0.1)
    if not isinstance(temperature, (int, float)):
        raise ValueError("`temperature` must be numeric.")

    stop = payload.get("stop")
    if stop is not None and not isinstance(stop, (str, list)):
        raise ValueError("`stop` must be a string or an array of strings.")
    if isinstance(stop, list) and not all(isinstance(item, str) for item in stop):
        raise ValueError("`stop` array entries must all be strings.")

    return {
        "model": model_name.strip(),
        "messages": normalized_messages,
        "stream": stream,
        "max_tokens": max_tokens,
        "temperature": float(temperature),
        "stop": stop,
        "tools": normalize_tools(payload.get("tools")),
        "tool_choice": normalize_tool_choice(payload.get("tool_choice")),
    }


def normalize_message(message):
    if not isinstance(message, dict):
        raise ValueError("Each message must be an object.")

    role = message.get("role")
    if role not in {"system", "user", "assistant", "tool"}:
        raise ValueError("Each message `role` must be system, user, assistant, or tool.")

    content = message.get("content")
    normalized = {"role": role}
    if isinstance(content, str):
        normalized["content"] = content
    elif content is None:
        normalized["content"] = None
    elif isinstance(content, list):
        normalized["content"] = [normalize_content_part(part) for part in content]
    else:
        raise ValueError("Each message `content` must be a string, null, or an array.")

    if role == "assistant" and "tool_calls" in message:
        normalized["tool_calls"] = normalize_tool_calls(message.get("tool_calls"))

    if role == "tool":
        tool_call_id = message.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise ValueError("Tool messages must include a non-empty `tool_call_id`.")
        normalized["tool_call_id"] = tool_call_id

    return normalized


def normalize_tools(tools):
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise ValueError("`tools` must be an array.")

    normalized = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise ValueError("Each tool definition must be an object.")
        if tool.get("type") != "function":
            raise ValueError("Only `function` tools are supported.")
        function = tool.get("function")
        if not isinstance(function, dict):
            raise ValueError("Each function tool must include a `function` object.")
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each function tool must include a non-empty `name`.")
        normalized_tool = {
            "type": "function",
            "function": {
                "name": name.strip(),
                "description": function.get("description"),
                "parameters": function.get("parameters"),
            },
        }
        normalized.append(normalized_tool)
    return normalized


def normalize_tool_choice(tool_choice):
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice not in {"none", "auto", "required"}:
            raise ValueError("String `tool_choice` must be none, auto, or required.")
        return tool_choice
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") != "function":
            raise ValueError("Object `tool_choice` must have type `function`.")
        function = tool_choice.get("function")
        if not isinstance(function, dict):
            raise ValueError("Object `tool_choice` must include a `function` object.")
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Object `tool_choice.function.name` must be non-empty.")
        return {"type": "function", "function": {"name": name.strip()}}
    raise ValueError("`tool_choice` must be a string or object.")


def normalize_tool_calls(tool_calls):
    if not isinstance(tool_calls, list):
        raise ValueError("`tool_calls` must be an array.")

    normalized = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            raise ValueError("Each tool call must be an object.")
        if tool_call.get("type") != "function":
            raise ValueError("Only `function` tool calls are supported.")
        function = tool_call.get("function")
        if not isinstance(function, dict):
            raise ValueError("Each tool call must include a `function` object.")
        call_id = tool_call.get("id")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("Each tool call must include a non-empty `id`.")
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each tool call function name must be non-empty.")
        if not isinstance(arguments, str):
            raise ValueError("Each tool call function arguments field must be a string.")
        normalized.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name.strip(),
                    "arguments": arguments,
                },
            }
        )
    return normalized


def normalize_content_part(part):
    if not isinstance(part, dict):
        raise ValueError("Message content parts must be objects.")

    part_type = part.get("type")
    if part_type == "text":
        text = part.get("text")
        if not isinstance(text, str):
            raise ValueError("Text content parts must include a string `text` field.")
        return {"type": "text", "text": text}

    if part_type == "image_url":
        image_url = part.get("image_url")
        if not isinstance(image_url, dict):
            raise ValueError("Image content parts must include an `image_url` object.")
        url = image_url.get("url")
        if not isinstance(url, str) or not url:
            raise ValueError("`image_url.url` must be a non-empty string.")
        return {"type": "image_url", "image_url": {"url": url}}

    raise ValueError("Supported content part types are `text` and `image_url`.")


def normalize_non_stream_response(response, completion_id, created, model_name):
    choices = response.get("choices") or []
    first_choice = choices[0] if choices else {}
    message = first_choice.get("message") or {"role": "assistant", "content": ""}
    finish_reason = first_choice.get("finish_reason", "stop")

    normalized_message = {
        "role": message.get("role", "assistant"),
        "content": message.get("content", ""),
    }
    if "tool_calls" in message:
        normalized_message["tool_calls"] = message.get("tool_calls")

    output = {
        "id": response.get("id", completion_id),
        "object": "chat.completion",
        "created": response.get("created", created),
        "model": response.get("model", model_name),
        "choices": [
            {
                "index": 0,
                "message": normalized_message,
                "finish_reason": finish_reason,
            }
        ],
    }

    if "usage" in response and isinstance(response["usage"], dict):
        output["usage"] = response["usage"]

    return output


def stream_completion(normalized, completion_id, created, model_name):
    pending_text = ""
    demultiplexer = Demultiplexer()

    try:
        stream = llm_wrapper.create_chat_completion(
            messages=normalized["messages"],
            stream=True,
            max_tokens=normalized["max_tokens"],
            temperature=normalized["temperature"],
            stop=normalized["stop"],
            model=model_name,
        )

        yield sse_chunk(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            }
        )

        for chunk in stream:
            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if content:
                pending_text += content
                yield sse_chunk(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": content},
                                "finish_reason": None,
                            }
                        ],
                    }
                )

                complete_sentences, pending_text = consume_complete_sentences(pending_text)
                for sentence in complete_sentences:
                    demultiplexer.route_sentence(sentence)

        if pending_text.strip():
            demultiplexer.route_sentence(pending_text.strip())

        print("\n\n\n")
        print(demultiplexer.pretty_print())
        print("\n\n\n")

        yield sse_chunk(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
        yield "data: [DONE]\n\n"
    except Exception as exc:
        print(f"Streaming error: {exc}")
        yield sse_chunk(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
                "error": {"message": str(exc), "type": "server_error"},
            }
        )
        yield "data: [DONE]\n\n"


def sse_chunk(payload):
    return f"data: {json.dumps(payload)}\n\n"


def legacy_chat_payload_to_openai(data):
    user_message = (data.get("message") or "").strip()
    image_b64 = data.get("image")
    image_mime_type = data.get("image_mime_type") or "image/jpeg"

    if not user_message and not image_b64:
        raise ValueError("message or image is required.")
    if image_b64 is not None and not isinstance(image_b64, str):
        raise ValueError("image must be a base64 string.")
    if not isinstance(image_mime_type, str) or not image_mime_type.startswith("image/"):
        raise ValueError("image_mime_type must be an image MIME type.")

    if image_b64:
        content = []
        if user_message:
            content.append({"type": "text", "text": user_message})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image_mime_type};base64,{image_b64}"},
            }
        )
    else:
        content = user_message

    return {
        "model": DEFAULT_MODEL_NAME,
        "messages": [{"role": "user", "content": content}],
        "stream": True,
    }


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
