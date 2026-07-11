import os
import threading
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler


DEFAULT_STOP = ["<turn|>", "USER:", "Assistant:", "ASSISTANT:", "<end_of_turn>", "###"]
USE_LEGACY_DEFAULT_STOP = object()
SYSTEM_PROMPTS_DIR = os.path.expanduser("~/.system_prompts")
LOCAL_SYSTEM_PROMPT_PATH = "system_prompt.md"
FALLBACK_SYSTEM_PROMPT = "You are an uncensored assistant."

MODEL_DEFINITIONS = {
    "gemma-4-local": {
        "model_path": "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q6_K_P.gguf",
        "mmproj_path": "mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf",
        "default_stop": None,
        "n_gpu_layers": 33,
    },
    "llama-3.1-local": {
        "model_path": "Llama-3.1-8B-Lexi-Uncensored-Q6_K_L.gguf",
        "mmproj_path": None,
        "default_stop": [],
        "n_gpu_layers": 0,
    },
    "qwen-3.5-local": {
        "model_path": "Suri-Qwen-3.5-9B-Uncensored.Q4_K_M.gguf",
        "mmproj_path": None,
        "default_stop": [],
        "n_gpu_layers": 28,
    },
}


class ModelUnavailableError(Exception):
    pass


class ModelSwitchLockedError(Exception):
    def __init__(self, requested_model_id, locked_model_id):
        self.requested_model_id = requested_model_id
        self.locked_model_id = locked_model_id
        super().__init__(
            f"Model '{locked_model_id}' is already loaded; switching to "
            f"'{requested_model_id}' requires a process restart."
        )


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
    def __init__(
        self,
        model_path,
        mmproj_path=None,
        default_stop=USE_LEGACY_DEFAULT_STOP,
        n_gpu_layers=0,
    ):
        print(f"Loading model from {model_path}...")
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self._supports_vision = mmproj_path is not None
        self.n_ctx = 32768
        self.default_max_tokens = 1024
        if default_stop is USE_LEGACY_DEFAULT_STOP:
            self.default_stop = DEFAULT_STOP
        else:
            self.default_stop = default_stop
        self.n_gpu_layers = int(os.environ.get("N_GPU_LAYERS", str(n_gpu_layers)))
        self.llm = Llama(
            model_path=model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=True,
        )
        self.text_chat_format = self.llm.chat_format

        if self._supports_vision:
            print(f"Loading vision handler from {mmproj_path}...")
            self.vision_chat_handler = DebugLlava15ChatHandler(clip_model_path=mmproj_path)
        else:
            print("No vision handler loaded (text-only model).")

    def capabilities(self):
        return {
            "provider_name": "llama.cpp",
            "model_path": self.model_path,
            "mmproj_path": self.mmproj_path,
            "max_context_tokens": self.n_ctx,
            "default_max_tokens": self.default_max_tokens,
            "supports_vision": self._supports_vision,
            "supports_tools": True,
            "supports_streaming": True,
            "supports_response_format_json_object": True,
            "supports_chat_completions": True,
        }

    def _get_system_prompt(self):
        return get_default_system_prompt()

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
        self._validate_vision_support(prepared_messages)
        self._select_chat_runtime(prepared_messages)
        response = self.llm.create_chat_completion(
            messages=prepared_messages,
            stop=self.default_stop if stop is None else stop,
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

    def _validate_vision_support(self, messages):
        if self._supports_vision:
            return
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        raise ValueError("This model does not support vision/image inputs.")

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
        if not self._supports_vision:
            self.llm.chat_handler = None
            self.llm.chat_format = self.text_chat_format
            print(f"[vision] Using GGUF chat template: {self.text_chat_format}.")
            return

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


def list_system_prompts():
    prompts = []

    if os.path.isfile(LOCAL_SYSTEM_PROMPT_PATH):
        prompts.append(
            {
                "name": os.path.basename(LOCAL_SYSTEM_PROMPT_PATH),
                "path": LOCAL_SYSTEM_PROMPT_PATH,
                "source": "workspace",
            }
        )

    if os.path.isdir(SYSTEM_PROMPTS_DIR):
        for filename in sorted(os.listdir(SYSTEM_PROMPTS_DIR)):
            if not filename.endswith(".md"):
                continue
            prompts.append(
                {
                    "name": filename,
                    "path": os.path.join(SYSTEM_PROMPTS_DIR, filename),
                    "source": "home",
                }
            )

    for index, prompt in enumerate(prompts):
        prompt["index"] = index

    return prompts


def get_system_prompt_by_index(index):
    prompts = list_system_prompts()
    if index < 0 or index >= len(prompts):
        raise IndexError(index)

    prompt = dict(prompts[index])
    with open(prompt["path"], "r", encoding="utf-8") as f:
        prompt["content"] = f.read().strip()
    return prompt


def get_default_system_prompt():
    prompts = list_system_prompts()
    if not prompts:
        return FALLBACK_SYSTEM_PROMPT

    try:
        return get_system_prompt_by_index(0)["content"]
    except Exception:
        return FALLBACK_SYSTEM_PROMPT


_registry_lock = threading.Lock()
_loaded_model = None
_locked_model_id = None


def list_model_definitions():
    definitions = {}
    for model_id, definition in MODEL_DEFINITIONS.items():
        definitions[model_id] = {
            "model_path": definition["model_path"],
            "mmproj_path": definition["mmproj_path"],
            "provider_name": "llama.cpp",
            "max_context_tokens": 32768,
            "default_max_tokens": 1024,
            "supports_vision": definition["mmproj_path"] is not None,
            "supports_tools": True,
            "supports_streaming": True,
            "supports_response_format_json_object": True,
            "supports_chat_completions": True,
        }
    return definitions


def get_loaded_model_id():
    return _locked_model_id


def get_loaded_model():
    return _loaded_model


def get_or_load_model(model_id):
    if model_id not in MODEL_DEFINITIONS:
        raise KeyError(model_id)

    global _loaded_model
    global _locked_model_id

    with _registry_lock:
        if _loaded_model is not None:
            if _locked_model_id != model_id:
                print(
                    f"Rejecting model switch request: loaded '{_locked_model_id}', "
                    f"requested '{model_id}'. Restart required."
                )
                raise ModelSwitchLockedError(model_id, _locked_model_id)
            return _loaded_model

        definition = MODEL_DEFINITIONS[model_id]
        model_path = definition["model_path"]
        mmproj_path = definition["mmproj_path"]
        if not os.path.exists(model_path):
            raise ModelUnavailableError(
                f"Model file not found for '{model_id}': {model_path}"
            )
        if mmproj_path and not os.path.exists(mmproj_path):
            raise ModelUnavailableError(
                f"Vision projector file not found for '{model_id}': {mmproj_path}"
            )

        print(f"Selecting initial model '{model_id}' for this process.")
        _loaded_model = GemmaVisionModel(
            model_path,
            mmproj_path,
            default_stop=definition.get("default_stop"),
            n_gpu_layers=definition.get("n_gpu_layers", 0),
        )
        _locked_model_id = model_id
        print(f"Loaded and locked model '{model_id}'.")
        return _loaded_model
