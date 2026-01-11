import sys
import os
import gc
from pathlib import Path
import uuid
import tempfile
import time
import random
import numpy as np
import torch
import gradio as gr
import spaces
from typing import Iterable, Optional
from PIL import Image

# Gradio Theme Imports
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

# Add packages to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "packages" / "ltx-pipelines" / "src"))
sys.path.insert(0, str(current_dir / "packages" / "ltx-core" / "src"))

import flash_attn_interface
from huggingface_hub import hf_hub_download, snapshot_download

# LTX Imports
from ltx_pipelines.distilled import DistilledPipeline
from ltx_core.model.video_vae import TilingConfig
from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
from ltx_pipelines.utils import ModelLedger
from ltx_pipelines.utils.helpers import generate_enhanced_prompt
from ltx_pipelines.utils.constants import (
    DEFAULT_SEED,
    DEFAULT_1_STAGE_HEIGHT,
    DEFAULT_1_STAGE_WIDTH,
    DEFAULT_NUM_FRAMES,
    DEFAULT_FRAME_RATE,
    DEFAULT_LORA_STRENGTH,
)

# -----------------------------------------------------------------------------
# 1. OrangeRed Theme Configuration
# -----------------------------------------------------------------------------

colors.orange_red = colors.Color(
    name="orange_red",
    c50="#FFF0E5",
    c100="#FFE0CC",
    c200="#FFC299",
    c300="#FFA366",
    c400="#FF8533",
    c500="#FF4500",
    c600="#E63E00",
    c700="#CC3700",
    c800="#B33000",
    c900="#992900",
    c950="#802200",
)

class OrangeRedTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.orange_red,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_secondary_text_color="black",
            button_secondary_text_color_hover="white",
            button_secondary_background_fill="linear-gradient(90deg, *primary_300, *primary_300)",
            button_secondary_background_fill_hover="linear-gradient(90deg, *primary_400, *primary_400)",
            button_secondary_background_fill_dark="linear-gradient(90deg, *primary_500, *primary_600)",
            button_secondary_background_fill_hover_dark="linear-gradient(90deg, *primary_500, *primary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

orange_red_theme = OrangeRedTheme()

# -----------------------------------------------------------------------------
# 2. Configuration & Adapters
# -----------------------------------------------------------------------------

MAX_SEED = np.iinfo(np.int32).max

# HuggingFace Hub defaults
DEFAULT_REPO_ID = "Lightricks/LTX-2"
DEFAULT_GEMMA_REPO_ID = "unsloth/gemma-3-12b-it-qat-bnb-4bit"
DEFAULT_CHECKPOINT_FILENAME = "ltx-2-19b-dev.safetensors"
DEFAULT_DISTILLED_LORA_FILENAME = "ltx-2-19b-distilled-lora-384.safetensors"
DEFAULT_SPATIAL_UPSAMPLER_FILENAME = "ltx-2-spatial-upscaler-x2-1.0.safetensors"

# New Adapter Definitions
ADAPTER_SPECS = {
    "Camera-Control-Dolly-Left": {
        "repo": "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left",
        "weights": "ltx-2-19b-lora-camera-control-dolly-left.safetensors",
        "adapter_name": "camera-control-dolly-left"
    },
    "Camera-Control-Dolly-Right": {
        "repo": "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right",
        "weights": "ltx-2-19b-lora-camera-control-dolly-right.safetensors",
        "adapter_name": "camera-control-dolly-right"
    },
    "Camera-Control-Dolly-In": {
        "repo": "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In",
        "weights": "ltx-2-19b-lora-camera-control-dolly-in.safetensors",
        "adapter_name": "camera-control-dolly-in"
    },
    "Camera-Control-Dolly-Out": {
        "repo": "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out",
        "weights": "ltx-2-19b-lora-camera-control-dolly-out.safetensors",
        "adapter_name": "camera-control-dolly-out"
    },

# ---------------------------------------others ---------------------------------------
    
     "Camera-Control-Jib-Down": {
        "repo": "Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down",
        "weights": "ltx-2-19b-lora-camera-control-jib-down.safetensors",
        "adapter_name": "camera-control-jib-down"
    },
     "Camera-Control-Jib-Up": {
        "repo": "Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up",
        "weights": "ltx-2-19b-lora-camera-control-jib-up.safetensors",
        "adapter_name": "camera-control-jib-up"
    },
     "Camera-Control-Static": {
        "repo": "Lightricks/LTX-2-19b-LoRA-Camera-Control-Static",
        "weights": "ltx-2-19b-lora-camera-control-static.safetensors",
        "adapter_name": "camera-control-static"
    },
}

# -----------------------------------------------------------------------------
# 3. Model Loading Helper Functions
# -----------------------------------------------------------------------------

def get_hub_or_local_checkpoint(repo_id: Optional[str] = None, filename: Optional[str] = None):
    """Download from HuggingFace Hub or use local checkpoint."""
    if repo_id is None and filename is None:
        raise ValueError("Please supply at least one of `repo_id` or `filename`")

    if repo_id is not None:
        if filename is None:
            raise ValueError("If repo_id is specified, filename must also be specified.")
        print(f"Downloading {filename} from {repo_id}...")
        ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
        print(f"Downloaded to {ckpt_path}")
    else:
        ckpt_path = filename

    return ckpt_path

def download_gemma_model(repo_id: str):
    """Download the full Gemma model directory."""
    print(f"Downloading Gemma model from {repo_id}...")
    local_dir = snapshot_download(repo_id=repo_id)
    print(f"Gemma model downloaded to {local_dir}")
    return local_dir

# -----------------------------------------------------------------------------
# 4. Global Initialization (Text Encoder & Paths)
# -----------------------------------------------------------------------------

print("=" * 80)
print("Initializing LTX-2 Environment...")
print("=" * 80)

device = "cuda"

# Load Text Encoder Weights
checkpoint_path = get_hub_or_local_checkpoint(DEFAULT_REPO_ID, DEFAULT_CHECKPOINT_FILENAME)
gemma_local_path = download_gemma_model(DEFAULT_GEMMA_REPO_ID)
distilled_lora_path = get_hub_or_local_checkpoint(DEFAULT_REPO_ID, DEFAULT_DISTILLED_LORA_FILENAME)
spatial_upsampler_path = get_hub_or_local_checkpoint(DEFAULT_REPO_ID, DEFAULT_SPATIAL_UPSAMPLER_FILENAME)

print("Loading Gemma Text Encoder...")
model_ledger = ModelLedger(
    dtype=torch.bfloat16,
    device=device,
    checkpoint_path=checkpoint_path,
    gemma_root_path=DEFAULT_GEMMA_REPO_ID,
    local_files_only=False
)
text_encoder = model_ledger.text_encoder()
print("Text encoder loaded.")

# -----------------------------------------------------------------------------
# 5. Inference Logic
# -----------------------------------------------------------------------------

def encode_text_simple(text_encoder, prompt: str):
    """Simple text encoding without using pipeline_utils."""
    v_context, a_context, _ = text_encoder(prompt)
    return v_context, a_context

@spaces.GPU()
def encode_prompt(
    prompt: str,
    enhance_prompt: bool = True,
    input_image=None,
    seed: int = 42,
    negative_prompt: str = ""
):
    start_time = time.time()
    try:
        final_prompt = prompt
        if enhance_prompt:
            final_prompt = generate_enhanced_prompt(
                text_encoder=text_encoder,
                prompt=prompt,
                image_path=input_image if input_image is not None else None,
                seed=seed,
            )

        with torch.inference_mode():
            video_context, audio_context = encode_text_simple(text_encoder, final_prompt)

        video_context_negative = None
        audio_context_negative = None
        if negative_prompt:
            video_context_negative, audio_context_negative = encode_text_simple(text_encoder, negative_prompt)

        embedding_data = {
            "video_context": video_context.detach().cpu(),
            "audio_context": audio_context.detach().cpu(),
            "prompt": final_prompt,
            "original_prompt": prompt,
        }
        if video_context_negative is not None:
            embedding_data["video_context_negative"] = video_context_negative
            embedding_data["audio_context_negative"] = audio_context_negative
            embedding_data["negative_prompt"] = negative_prompt

        elapsed_time = time.time() - start_time
        status = f"✓ Encoded in {elapsed_time:.2f}s"
        return embedding_data, final_prompt, status

    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, prompt, error_msg

# Function to calculate timeout based on the arguments passed to the GPU decorated function
def calc_timeout_duration(input_image, prompt, lora_adapter, duration, enhance_prompt, seed, randomize_seed, height, width, gpu_timeout, progress=None):
    # Determine timeout based on the last argument (gpu_timeout) passed to generate_video
    try:
        return int(gpu_timeout)
    except:
        return 120

@spaces.GPU(duration=calc_timeout_duration)
def generate_video(
    input_image,
    prompt: str,
    lora_adapter: str,
    duration: float,
    enhance_prompt: bool = True,
    seed: int = 42,
    randomize_seed: bool = True,
    height: int = DEFAULT_1_STAGE_HEIGHT,
    width: int = DEFAULT_1_STAGE_WIDTH,
    gpu_timeout: int = 120,
    progress=gr.Progress(track_tqdm=True),
):
    gc.collect()
    torch.cuda.empty_cache()
    
    try:
        current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
        frame_rate = 24.0
        num_frames = int(duration * frame_rate) + 1

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
            output_path = tmpfile.name

        # Prepare Inputs
        images = []
        if input_image is not None:
            images = [(input_image, 0, 1.0)]

        # Encode Prompt
        embeddings, final_prompt, status = encode_prompt(
            prompt=prompt,
            enhance_prompt=enhance_prompt,
            input_image=input_image,
            seed=current_seed,
            negative_prompt="",
        )
        
        if embeddings is None:
            raise Exception("Failed to encode prompt")

        video_context = embeddings["video_context"].to("cuda", non_blocking=True)
        audio_context = embeddings["audio_context"].to("cuda", non_blocking=True)
        
        # ---------------------------
        # Configure LoRAs
        # ---------------------------
        # Always start with the base Distilled LoRA
        active_loras = [
            LoraPathStrengthAndSDOps(
                path=distilled_lora_path,
                strength=DEFAULT_LORA_STRENGTH,
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            )
        ]

        # Add additional selected Adapter
        if lora_adapter and lora_adapter != "None":
            spec = ADAPTER_SPECS.get(lora_adapter)
            if spec:
                print(f"Loading Adapter: {lora_adapter}")
                # Download on demand
                adapter_path = get_hub_or_local_checkpoint(repo_id=spec["repo"], filename=spec["weights"])
                
                # Append to list
                active_loras.append(
                    LoraPathStrengthAndSDOps(
                        path=adapter_path,
                        strength=0.8, # Default strength for style/camera LoRAs
                        sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
                    )
                )

        # ---------------------------
        # Instantiate Pipeline
        # ---------------------------
        # We instantiate the pipeline inside the GPU function to ensure LoRAs are applied correctly
        # for this specific run without global state pollution.
        pipeline = DistilledPipeline(
            device=torch.device("cuda"),
            checkpoint_path=checkpoint_path,
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_root=None, # Already handled externally
            loras=active_loras,
            fp8transformer=False,
            local_files_only=False,
        )
        
        # Explicitly link the pre-loaded encoder/transformer to avoid VRAM bloat
        pipeline._video_encoder = pipeline.model_ledger.video_encoder()
        pipeline._transformer = pipeline.model_ledger.transformer()

        # Run Generation
        pipeline(
            prompt=prompt,
            output_path=str(output_path),
            seed=current_seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            tiling_config=TilingConfig.default(),
            video_context=video_context,
            audio_context=audio_context,
        )

        del video_context, audio_context, pipeline
        gc.collect()
        torch.cuda.empty_cache()
        
        return str(output_path), current_seed

    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, current_seed

def generate_video_example(input_image, prompt, lora_adapter, duration):
    # We pass a default 120s timeout for examples
    output, seed = generate_video(
        input_image=input_image, 
        prompt=prompt, 
        lora_adapter=lora_adapter,
        duration=5.0, 
        enhance_prompt=True, 
        seed=42, 
        randomize_seed=True, 
        height=DEFAULT_1_STAGE_HEIGHT, 
        width=DEFAULT_1_STAGE_WIDTH,
        gpu_timeout=120
    )
    return output

# -----------------------------------------------------------------------------
# 6. UI Components
# -----------------------------------------------------------------------------

def apply_resolution(resolution: str):
    w, h = resolution.split("x")
    return int(w), int(h)

def apply_duration(duration: str):
    duration_s = int(duration[:-1])
    return duration_s

def apply_gpu_duration(val: str):
    return int(val)

class RadioAnimated(gr.HTML):
    def __init__(self, choices, value=None, **kwargs):
        if not choices or len(choices) < 2:
            raise ValueError("RadioAnimated requires at least 2 choices.")
        if value is None:
            value = choices[0]

        uid = uuid.uuid4().hex[:8]
        group_name = f"ra-{uid}"

        inputs_html = "\n".join(
            f"""
            <input class="ra-input" type="radio" name="{group_name}" id="{group_name}-{i}" value="{c}">
            <label class="ra-label" for="{group_name}-{i}">{c}</label>
            """
            for i, c in enumerate(choices)
        )

        html_template = f"""
        <div class="ra-wrap" data-ra="{uid}">
          <div class="ra-inner">
            <div class="ra-highlight"></div>
            {inputs_html}
          </div>
        </div>
        """

        js_on_load = r"""
        (() => {
          const wrap = element.querySelector('.ra-wrap');
          const inner = element.querySelector('.ra-inner');
          const highlight = element.querySelector('.ra-highlight');
          const inputs = Array.from(element.querySelectorAll('.ra-input'));

          if (!inputs.length) return;

          const choices = inputs.map(i => i.value);

          function setHighlightByIndex(idx) {
            const n = choices.length;
            const pct = 100 / n;
            highlight.style.width = `calc(${pct}% - 6px)`;
            highlight.style.transform = `translateX(${idx * 100}%)`;
          }

          function setCheckedByValue(val, shouldTrigger=false) {
            const idx = Math.max(0, choices.indexOf(val));
            inputs.forEach((inp, i) => { inp.checked = (i === idx); });
            setHighlightByIndex(idx);

            props.value = choices[idx];
            if (shouldTrigger) trigger('change', props.value);
          }

          setCheckedByValue(props.value ?? choices[0], false);

          inputs.forEach((inp) => {
            inp.addEventListener('change', () => {
              setCheckedByValue(inp.value, true);
            });
          });
        })();
        """

        super().__init__(
            value=value,
            html_template=html_template,
            js_on_load=js_on_load,
            **kwargs
        )

# -----------------------------------------------------------------------------
# 7. Gradio Application
# -----------------------------------------------------------------------------

css = """
    #col-container {
        margin: 0 auto;
        max-width: 1200px;
    }
    #step-column {
        padding: 20px;
        border-radius: 12px;
        background: var(--background-fill-secondary);
        border: 1px solid var(--border-color-primary);
        margin-bottom: 20px;
    }
    .button-gradient {
        background: linear-gradient(90deg, #FF4500, #E63E00);
        border: none;
        color: white;
        font-weight: bold;
    }
    .ra-wrap{ width: fit-content; }
    .ra-inner{
      position: relative; display: inline-flex; align-items: center; gap: 0; padding: 6px;
      background: var(--neutral-200); border-radius: 9999px; overflow: hidden;
    }
    .ra-input{ display: none; }
    .ra-label{
      position: relative; z-index: 2; padding: 8px 16px;
      font-family: inherit; font-size: 14px; font-weight: 600;
      color: var(--neutral-500); cursor: pointer; transition: color 0.2s; white-space: nowrap;
    }
    .ra-highlight{
      position: absolute; z-index: 1; top: 6px; left: 6px;
      height: calc(100% - 12px); border-radius: 9999px;
      background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      transition: transform 0.2s, width 0.2s;
    }
    .ra-input:checked + .ra-label{ color: black; }
    
    /* Dark mode adjustments for Radio */
    .dark .ra-inner { background: var(--neutral-800); }
    .dark .ra-label { color: var(--neutral-400); }
    .dark .ra-highlight { background: var(--neutral-600); }
    .dark .ra-input:checked + .ra-label { color: white; }

    #main-title h1 { font-size: 2.2em !important; }
"""

with gr.Blocks() as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# **LTX-2-LoRAs-Camera-Control-Dolly**", elem_id="main-title")
        gr.Markdown("Create cinematic video from text or image using [LTX-2 Distilled](https://huggingface.co/Lightricks/LTX-2/blob/main/ltx-2-19b-distilled-lora-384.safetensors) model. Select LoRA adapters for specific camera movements or styles.")
        
        with gr.Row():
            # Left Column: Inputs
            with gr.Column(elem_id="step-column"):
                input_image = gr.Image(
                    label="Input Image (Optional)",
                    type="filepath",
                    height=300
                )
                
                prompt = gr.Textbox(
                    label="Prompt",
                    value="Make this image come alive with cinematic motion...",
                    lines=3,
                    placeholder="Describe the motion and animation you want..."
                )
                
                with gr.Row():
                    lora_adapter = gr.Dropdown(
                        label="Camera Control [LoRA]",
                        choices=list(ADAPTER_SPECS.keys()),
                        value="Camera-Control-Dolly-Left",
                        info="Select a specific camera movement or style adapter."
                    )

                enhance_prompt = gr.Checkbox(label="Enhance Prompt", value=True, visible=False)
                
                generate_btn = gr.Button("Generate Video", variant="primary", elem_classes="button-gradient")

            # Right Column: Output & Settings
            with gr.Column(elem_id="step-column"):
                output_video = gr.Video(label="Generated Video", autoplay=True, height=360)
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Duration**")
                        radioanimated_duration = RadioAnimated(
                            choices=["3s", "5s", "10s"],
                            value="3s",
                            elem_id="radioanimated_duration"
                        )
                        duration = gr.Number(value=3.0, visible=False)
                        
                    with gr.Column():
                        gr.Markdown("**Resolution**")
                        radioanimated_resolution = RadioAnimated(
                            choices=["768x512", "512x512", "512x768"],
                            value=f"{DEFAULT_1_STAGE_WIDTH}x{DEFAULT_1_STAGE_HEIGHT}",
                            elem_id="radioanimated_resolution"
                        )
                        width = gr.Number(value=DEFAULT_1_STAGE_WIDTH, visible=False)
                        height = gr.Number(value=DEFAULT_1_STAGE_HEIGHT, visible=False)

                # New GPU Duration Row below Resolution
                with gr.Row():
                     with gr.Column():
                        gr.Markdown("**GPU Duration**")
                        radioanimated_gpu_duration = RadioAnimated(
                            choices=["120", "180", "240", "300"],
                            value="120",
                            elem_id="radioanimated_gpu_duration"
                        )
                        gpu_duration_state = gr.Number(value=120, visible=False)

                with gr.Accordion("Advanced Settings", open=False, visible=False):
                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, value=DEFAULT_SEED, step=1)
                    randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                    
    # Wire up events
    radioanimated_duration.change(fn=apply_duration, inputs=radioanimated_duration, outputs=[duration], api_visibility="private")
    radioanimated_resolution.change(fn=apply_resolution, inputs=radioanimated_resolution, outputs=[width, height], api_visibility="private")
    radioanimated_gpu_duration.change(fn=apply_gpu_duration, inputs=radioanimated_gpu_duration, outputs=[gpu_duration_state], api_visibility="private")

    generate_btn.click(
        fn=generate_video,
        inputs=[input_image, prompt, lora_adapter, duration, enhance_prompt, seed, randomize_seed, height, width, gpu_duration_state],
        outputs=[output_video, seed]
    )

    gr.Examples(
        examples=[
            ["examples/dolly_left.jpg", "Dolly-left camera move, sliding left in an open space, revealing a stationary car, off-frame left elements enter the shot, cinematic parallax", "Camera-Control-Dolly-Left"],
            ["examples/dolly_in.jpg", "Slow dolly-in toward a face, gentle zoom-in, cinematic parallax.", "Camera-Control-Dolly-In"],
            ["examples/dolly_right.jpg", "Dolly-right camera move in an open space, sliding right, cinematic parallax.", "Camera-Control-Dolly-Right"],
            ["examples/dolly_out.jpg", "Slow dolly-out with gentle zoom-out, cinematic parallax.", "Camera-Control-Dolly-Out"],
            ["examples/Jib_down.jpg", "Jib-down camera move, smooth vertical descent, cinematic reveal.", "Camera-Control-Jib-Down"],
            ["examples/jib-up.jpg", "Jib-up camera move, smooth vertical ascent, cinematic reveal.", "Camera-Control-Jib-Up"],
            ["examples/cam_static.jpg", "Static camera, locked-off shot, no camera movement.", "Camera-Control-Static"],
        ],
        fn=generate_video_example,
        inputs=[input_image, prompt, lora_adapter],
        outputs=[output_video],
        label="Examples",
        cache_examples=False
    )

if __name__ == "__main__":
    demo.launch(css=css, theme=orange_red_theme, ssr_mode=False, mcp_server=True)