import base64
import os
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

class GemmaVisionModel:
    def __init__(self, model_path, mmproj_path):
        print(f"Loading vision handler from {mmproj_path}...")
        self.chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path)
        
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

    def generate_response(self, text, image_b64=None, system_prompt="You are a helpful AI assistant.", stream=False):
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
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
            })

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
