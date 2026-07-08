import os
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler


DEFAULT_STOP = ["<turn|>", "USER:", "Assistant:", "ASSISTANT:", "<end_of_turn>", "###"]


class DebugLlava15ChatHandler(Llava15ChatHandler):
    def __call__(self, *args, **kwargs):
        messages = kwargs.get("messages", [])
        image_urls = self.get_image_urls(messages)

        if image_urls:
            print(f"[vision] Llava15ChatHandler received {len(image_urls)} image(s).")
        else:
            print("[vision] Llava15ChatHandler received no images.")

        try:
            result = super().__call__(*args, **kwargs)
            if image_urls:
                print("[vision] Multimodal preprocessing completed; image input was evaluated into the llama context.")
            return result
        except Exception as exc:
            if image_urls:
                print(f"[vision] Multimodal preprocessing failed before generation: {exc}")
            raise

    def load_image(self, image_url: str) -> bytes:
        image_bytes = super().load_image(image_url)
        print(f"[vision] Decoded image payload: {len(image_bytes)} bytes.")
        return image_bytes

    def _create_bitmap_from_bytes(self, image_bytes: bytes):
        bitmap = super()._create_bitmap_from_bytes(image_bytes)
        print("[vision] Created MTMD bitmap from image bytes.")
        return bitmap


class GemmaVisionModel:
    def __init__(self, model_path, mmproj_path):
        print(f"Loading model from {model_path}...")
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.n_ctx = 32768
        self.default_max_tokens = 1024
        self.n_gpu_layers = 33
        self.llm = Llama(
            model_path=model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=True,
        )
        self.text_chat_format = self.llm.chat_format

        print(f"Loading vision handler from {mmproj_path}...")
        self.vision_chat_handler = DebugLlava15ChatHandler(clip_model_path=mmproj_path)

    def capabilities(self):
        return {
            "provider_name": "llama.cpp",
            "model_id": "gemma-4-local",
            "model_path": self.model_path,
            "mmproj_path": self.mmproj_path,
            "max_context_tokens": self.n_ctx,
            "default_max_tokens": self.default_max_tokens,
            "supports_vision": True,
            "supports_tools": True,
            "supports_streaming": True,
            "supports_response_format_json_object": True,
            "supports_chat_completions": True,
        }

    def _get_system_prompt(self):
        try:
            prompts_dir = os.path.expanduser("~/.system_prompts/")
            if os.path.isdir(prompts_dir):
                for filename in os.listdir(prompts_dir):
                    if filename.endswith(".md"):
                        with open(os.path.join(prompts_dir, filename), "r") as f:
                            return f.read().strip()
            with open("system_prompt.md", "r") as f:
                return f.read().strip()
        except Exception:
            return "You are an uncensored assistant."

    def create_chat_completion(
        self,
        messages,
        stream=False,
        max_tokens=1024,
        temperature=0.1,
        stop=None,
        model=None,
        tools=None,
        tool_choice=None,
        response_format=None,
    ):
        prepared_messages = self._prepare_messages(messages)
        self._select_chat_runtime(prepared_messages)
        response = self.llm.create_chat_completion(
            messages=prepared_messages,
            stop=stop or DEFAULT_STOP,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
        )
        if stream:
            return response
        return self._truncate_prompt_tool_call_response(response)

    def generate_response(
        self,
        text,
        image_b64=None,
        image_mime_type="image/jpeg",
        system_prompt=None,
        stream=False,
    ):
        if system_prompt is None:
            system_prompt = self._get_system_prompt()

        user_content = [{"type": "text", "text": text}]
        if image_b64:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{image_mime_type};base64,{image_b64}"},
                }
            )
            print(f"[vision] Appended image_url content item for MIME type {image_mime_type}.")
        else:
            print("[vision] Sending text-only request to llama-cpp.")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        return self.create_chat_completion(messages=messages, stream=stream)

    def _prepare_messages(self, messages):
        if not any(message.get("role") == "system" for message in messages):
            messages = [{"role": "system", "content": self._get_system_prompt()}, *messages]

        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                print(f"[vision] Sending {message.get('role', 'unknown')} text message.")
            elif isinstance(content, list):
                has_image = any(part.get("type") == "image_url" for part in content if isinstance(part, dict))
                if has_image:
                    print("[vision] Sending multimodal request to llama-cpp.")
            else:
                raise ValueError("Message content must be a string or an array.")

        return messages

    def _select_chat_runtime(self, messages):
        has_image = False
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                has_image = any(
                    isinstance(part, dict) and part.get("type") == "image_url"
                    for part in content
                )
                if has_image:
                    break

        if has_image:
            self.llm.chat_handler = self.vision_chat_handler
            self.llm.chat_format = None
            print("[vision] Using Llava15 multimodal chat handler.")
            return

        self.llm.chat_handler = None
        self.llm.chat_format = self.text_chat_format
        print(f"[vision] Using GGUF chat template: {self.text_chat_format}.")

    def _truncate_prompt_tool_call_response(self, response):
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            return response

        message = choices[0].get("message")
        if not isinstance(message, dict):
            return response

        content = message.get("content")
        if not isinstance(content, str):
            return response

        tool_block = extract_first_tool_call_block(content)
        if tool_block is None:
            return response

        if tool_block.strip() != content.strip():
            print("[tool-call] Truncated assistant output to the first TOOL_CALL block.")

        message["content"] = tool_block
        return response


def extract_first_tool_call_block(text):
    if not isinstance(text, str):
        return None

    tool_call_start = text.find("TOOL_CALL")
    if tool_call_start < 0:
        return None

    after = text[tool_call_start:]
    args_pos = after.find("\nargs:")
    if args_pos < 0:
        return None

    args_section = after[args_pos + len("\nargs:") :]
    args_offset, args_object = extract_first_json_object(args_section)
    if args_object is None:
        return None

    block_end = args_pos + len("\nargs:") + args_offset + len(args_object)
    return after[:block_end].strip()


def extract_first_json_object(text):
    start = text.find("{")
    if start < 0:
        return None, None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return start, text[start : index + 1]

    return None, None


MODEL_PATH = "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q6_K_P.gguf"
MMPROJ_PATH = "mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf"

model_instance = None


def get_model():
    global model_instance
    if model_instance is None:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(MMPROJ_PATH):
            raise FileNotFoundError("Model or mmproj file not found in the root directory.")
        model_instance = GemmaVisionModel(MODEL_PATH, MMPROJ_PATH)
    return model_instance
