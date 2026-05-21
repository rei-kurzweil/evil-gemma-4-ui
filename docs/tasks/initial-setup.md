# Task: Initial Setup and Hardware Acceleration

**Date:** May 20, 2026
**Status:** Completed (Ready for Implementation)

## Overview
Successfully set up a Python environment to run the Gemma-4 vision model with hardware acceleration on an NVIDIA GTX 1080 (Pascal architecture).

## Hardware & Environment
- **GPU:** NVIDIA GeForce GTX 1080
- **OS:** EndeavourOS (Arch-based)
- **Python:** 3.14.4 (in `venv`)
- **CUDA:** 13.2 (Installed, but incompatible with GTX 1080 for `llama-cpp-python` builds)

## Challenges & Solutions

### 1. CUDA Incompatibility
- **Problem:** CUDA 13.2 dropped support for the Pascal architecture (`compute_61`), causing `llama-cpp-python` build failures.
- **Solution:** Switched to **Vulkan** acceleration, which is well-supported by both the OS and the hardware.

### 2. Missing Vulkan/SPIR-V Headers
- **Problem:** The build failed initially because development headers were missing.
- **Solution:** Installed headers via pacman:
  ```bash
  sudo pacman -S vulkan-headers spirv-headers
  ```

### 3. SPIR-V Header Path Issue
- **Problem:** Even with headers installed, the compiler couldn't find `spirv/unified1/spirv.hpp` because it was located under `/usr/include/spirv/unified1/`.
- **Solution:** Passed explicit include flags during the `pip install`:
  ```bash
  CMAKE_ARGS="-DGGML_VULKAN=ON -DCMAKE_CXX_FLAGS=-I/usr/include/spirv" pip install llama-cpp-python
  ```

## Current State
- `venv` is fully configured.
- `llama-cpp-python` is compiled with Vulkan support.
- `flask` and `pillow` are installed.
- `vision_inference.py` template is ready.

## Next Steps (Post-Reboot)
1. Implement the Flask backend (`app.py`).
2. Create the Web UI (templates and static files).
3. Test end-to-end inference through the browser.
