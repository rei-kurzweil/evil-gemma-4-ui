# Gemma-4 Vision Support Setup

This project aims to run the Gemma-4-E4B model with vision support programmatically using Python and `llama-cpp-python`.

## Prerequisites

- Python 3.11+
- NVIDIA GPU (GTX 1080 detected)
- CUDA Toolkit installed (for hardware acceleration)

## Files

- Model: `Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q6_K_P.gguf`
- Vision Projector: `mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf`

## Architecture: Web UI for Image Uploads

To make image interaction easier, we will implement a simple Web UI.

### Components
1.  **Backend (Flask):** 
    - Handles image uploads and stores them temporarily.
    - Interfaces with the Gemma-4 model via `vision_inference.py`.
    - Returns analysis results as JSON.
2.  **Frontend (Vanilla HTML/JS):**
    - A clean interface for selecting/dropping images.
    - Input field for the prompt.
    - Real-time display of the model's response.

### Implementation Plan (Next Steps)
- Create `app.py` (Flask server).
- Create `templates/index.html` (UI).
- Create `static/style.css` (Styling).

## Setup Instructions

### 1. Create a Virtual Environment
...

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install llama-cpp-python with Vulkan Support

The GTX 1080 (Pascal) is not supported by the installed CUDA 13.2. We will use Vulkan for hardware acceleration instead.

```bash
# Ensure you have the Vulkan SDK or headers installed
CMAKE_ARGS="-DGGML_VULKAN=ON" pip install llama-cpp-python
```
*Note: We might also need `transformers` and `pillow` for image processing.*

### 3. Implementation Plan

The script `vision_inference.py` has been created. It:
1. Loads the main model using `Llama`.
2. Loads the `mmproj` (vision projector) via `Llava15ChatHandler`.
3. Processes an input image and a prompt.
4. Outputs the model's response.

### 4. Example Usage

```python
from vision_inference import analyze_image

result = analyze_image("test_image.jpg", "Describe this image.")
print(result)
```

## Running the Model

The original `llama-cli` command for reference:
```bash
llama-cli -m Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q6_K_P.gguf \
  --mmproj mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf \
  --jinja -c 8192 -ngl 99
```
