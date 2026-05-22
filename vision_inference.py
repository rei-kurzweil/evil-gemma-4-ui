import os
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler


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
        # n_gpu_layers=33 is a safer starting point for a 4B model on 8GB VRAM
        # when using a large mmproj and 8k context.
        self.llm = Llama(
            model_path=model_path,
            chat_handler=self.chat_handler,
            n_ctx=8192,
            n_gpu_layers=33,
            verbose=True
        )

    def _get_system_prompt(self):
        try:
            # try reading files from ~/.system_prompts/ first
            prompts_dir = os.path.expanduser("~/.system_prompts/")
            if os.path.isdir(prompts_dir):
                for filename in os.listdir(prompts_dir):
                    if filename.endswith(".md"):
                        with open(os.path.join(prompts_dir, filename), "r") as f:
                            return f.read().strip()
            # fallback to system_prompt.md in the current directory
            with open("system_prompt.md", "r") as f:
                return f.read().strip()
        except Exception:
            return "You are a an evil magical girl antagonist who is preparing to attack the user with a series of glowing orbs"

    def generate_response(self, text, image_b64=None, image_mime_type="image/jpeg", system_prompt=None, stream=False):
        if system_prompt is None:
            system_prompt = self._get_system_prompt()
            
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text}
                ]
            }
        ]
        
        if image_b64:
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:{image_mime_type};base64,{image_b64}"}
            })
            print(f"[vision] Appended image_url content item for MIME type {image_mime_type}.")
        else:
            print("[vision] Sending text-only request to llama-cpp.")

        return self.llm.create_chat_completion(
            messages=messages,
            stop=["USER:", "Assistant:", "<end_of_turn>", "###"],
            max_tokens=1024,
            temperature=0.7,
            stream=stream
        )

# Paths to the model and vision projector
MODEL_PATH = "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q6_K_P.gguf"
MMPROJ_PATH = "mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf"

# The singleton instance
model_instance = None

def get_model():
    global model_instance
    if model_instance is None:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(MMPROJ_PATH):
            raise FileNotFoundError("Model or mmproj file not found in the root directory.")
        model_instance = GemmaVisionModel(MODEL_PATH, MMPROJ_PATH)
    return model_instance
