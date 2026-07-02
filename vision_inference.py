import os
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler


DEFAULT_STOP = ["USER:", "Assistant:", "<end_of_turn>", "###"]


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
        print(f"Loading vision handler from {mmproj_path}...")
        self.chat_handler = DebugLlava15ChatHandler(clip_model_path=mmproj_path)

        print(f"Loading model from {model_path}...")
        self.llm = Llama(
            model_path=model_path,
            chat_handler=self.chat_handler,
            n_ctx=8192,
            n_gpu_layers=33,
            verbose=True,
        )

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
    ):
        prepared_messages = self._prepare_messages(messages)
        return self.llm.create_chat_completion(
            messages=prepared_messages,
            stop=stop or DEFAULT_STOP,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )

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
