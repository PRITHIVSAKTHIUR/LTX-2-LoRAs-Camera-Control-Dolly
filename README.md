# **LTX-2-LoRAs-Camera-Control-Dolly**

> A Gradio-based demonstration for the Lightricks LTX-2 Distilled model, enhanced with specialized LoRA adapters for cinematic camera movements (dolly left/right/in/out, jib up/down, static). Generates animated videos from text prompts or input images, with optional prompt enhancement using Gemma-3-12b. Supports configurable duration, resolution, and GPU timeout for efficient generation.

## Features

- **Camera Control LoRAs**: 7 adapters for precise movements (e.g., "Dolly-left camera move, sliding left in an open space").
- **Text/Image-to-Video**: Start with prompts or images; auto-enhances prompts for better results.
- **Dynamic Settings**: Radio buttons for duration (3s/5s/10s), resolution (768x512/512x512/512x768), GPU timeout (120-300s).
- **Advanced Options**: Seed randomization, manual seed control.
- **Custom Theme**: OrangeRedTheme with animated radio buttons and responsive CSS.
- **Examples**: 7 pre-loaded scenarios with images, prompts, and adapters for quick testing.
- **Queueing**: Up to 30 concurrent jobs with progress tracking.
- **Memory Management**: Auto-clears cache after generations.

## Prerequisites

- Python 3.10 or higher.
- CUDA-compatible GPU (required for bfloat16 and Flash Attention 3).
- pip >= 23.0.0 (see pre-requirements.txt).
- Stable internet for initial model/LoRA downloads.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/PRITHIVSAKTHIUR/LTX-2-LoRAs-Camera-Control-Dolly.git
   cd LTX-2-LoRAs-Camera-Control-Dolly
   ```

2. Install pre-requirements:
   Create a `pre-requirements.txt` file with the following content, then run:
   ```
   pip install -r pre-requirements.txt
   ```

   **pre-requirements.txt content:**
   ```
   pip>=23.0.0
   ```

3. Install dependencies:
   Create a `requirements.txt` file with the following content, then run:
   ```
   pip install -r requirements.txt
   ```

   **requirements.txt content:**
   ```
   einops
   numpy
   torchaudio==2.8.0
   transformers
   safetensors
   accelerate
   flashpack==0.1.2
   scikit-image>=0.25.2
   av
   tqdm
   pillow
   scipy>=1.14
   flash-attn-3 @ https://huggingface.co/alexnasa/flash-attn-3/resolve/main/128/flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl
   bitsandbytes
   gradio
   ```

4. Start the application:
   ```
   python app.py
   ```
   The demo launches at `http://localhost:7860`.

## Usage

1. **Input Selection**:
   - Text-to-Image-3D: Enter prompt, generate image.
   - Image-to-3D: Upload image directly.

2. **Configure**:
   - Select LoRA for camera movement.
   - Choose duration/resolution/GPU timeout via animated radios.
   - Optional: Enhance prompt, set seed.

3. **Generate**: Click "Generate Video".

4. **Output**: View MP4 video; download if needed.

### Supported LoRAs

| LoRA Name                | Example Prompt                                      |
|--------------------------|-----------------------------------------------------|
| Camera-Control-Dolly-Left | "Dolly-left camera move, sliding left..."          |
| Camera-Control-Dolly-Right| "Dolly-right camera move, sliding right..."        |
| Camera-Control-Dolly-In  | "Slow dolly-in toward a face, gentle zoom-in..."   |
| Camera-Control-Dolly-Out | "Slow dolly-out with gentle zoom-out..."           |
| Camera-Control-Jib-Down  | "Jib-down camera move, smooth vertical descent..." |
| Camera-Control-Jib-Up    | "Jib-up camera move, smooth vertical ascent..."    |
| Camera-Control-Static    | "Static camera, locked-off shot..."                |

## Examples

| Input Image       | Prompt Example                                      | LoRA                         |
|-------------------|-----------------------------------------------------|------------------------------|
| examples/dolly_left.jpg | "Dolly-left camera move, sliding left in an open space..." | Camera-Control-Dolly-Left   |
| examples/dolly_in.jpg | "Slow dolly-in toward a face, gentle zoom-in..."   | Camera-Control-Dolly-In     |
| examples/dolly_right.jpg | "Dolly-right camera move in an open space..."      | Camera-Control-Dolly-Right  |
| examples/dolly_out.jpg | "Slow dolly-out with gentle zoom-out..."           | Camera-Control-Dolly-Out    |
| examples/Jib_down.jpg | "Jib-down camera move, smooth vertical descent..." | Camera-Control-Jib-Down     |
| examples/jib-up.jpg | "Jib-up camera move, smooth vertical ascent..."    | Camera-Control-Jib-Up       |
| examples/cam_static.jpg | "Static camera, locked-off shot..."                | Camera-Control-Static       |

## Troubleshooting

- **LoRA Loading**: First use downloads adapter; check console.
- **OOM**: Reduce duration/resolution; clear cache with `torch.cuda.empty_cache()`.
- **Flash Attention Fails**: Fallback to default; requires compatible CUDA.
- **No Video**: Ensure prompt descriptive; enhance enabled by default.
- **Queue Full**: Increase `max_size` in `demo.queue()`.
- **Gemma Encoder**: Loads on first use; handles optional images.

## Contributing

Contributions welcome! Add LoRAs to `ADAPTER_SPECS`, improve prompt enhancement, or enhance UI.

Repository: [https://github.com/PRITHIVSAKTHIUR/LTX-2-LoRAs-Camera-Control-Dolly.git](https://github.com/PRITHIVSAKTHIUR/LTX-2-LoRAs-Camera-Control-Dolly.git)

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

Built by Prithiv Sakthi. Report issues via the repository.
