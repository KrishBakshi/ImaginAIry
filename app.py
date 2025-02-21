import gradio as gr
import access_token as acc_tok
import utils
import time
import os
import gc
import numpy as np
import torch
import json
import logging
from PIL import Image, PngImagePlugin
from datetime import datetime
from google import genai
from google.genai import types
from config import (
    MODEL,
    MIN_IMAGE_SIZE,
    MAX_IMAGE_SIZE,
    USE_TORCH_COMPILE,
    ENABLE_CPU_OFFLOAD,
    OUTPUT_DIR,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_ASPECT_RATIO,
    examples,
    sampler_list,
    aspect_ratios,
    style_list,
)

# SDXL pipelines
from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers import DPMSolverMultistepScheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PyTorch settings for better performance and determinism
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True

# Access tokens
HF_TOKEN = acc_tok.access_token_sdxl
API_KEY = acc_tok.GOOGLE_API_KEY

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Model: {MODEL}")

class GenerationError(Exception):
    """Custom exception for generation errors"""
    pass

def validate_prompt(prompt: str) -> str:
    """Validate and clean up the input prompt."""
    if not isinstance(prompt, str):
        raise GenerationError("Prompt must be a string")
    try:
        # Ensure proper UTF-8 encoding/decoding
        prompt = prompt.encode('utf-8').decode('utf-8')
        # Add space between ! and ,
        prompt = prompt.replace("!,", "! ,")
    except UnicodeError:
        raise GenerationError("Invalid characters in prompt")
    
    # Only check if the prompt is completely empty or only whitespace
    if not prompt or prompt.isspace():
        raise GenerationError("Prompt cannot be empty")
    return prompt.strip()

def validate_dimensions(width: int, height: int) -> None:
    """Validate image dimensions."""
    if not MIN_IMAGE_SIZE <= width <= MAX_IMAGE_SIZE:
        raise GenerationError(f"Width must be between {MIN_IMAGE_SIZE} and {MAX_IMAGE_SIZE}")
        
    if not MIN_IMAGE_SIZE <= height <= MAX_IMAGE_SIZE:
        raise GenerationError(f"Height must be between {MIN_IMAGE_SIZE} and {MAX_IMAGE_SIZE}")

def prompt_augmentation(prompt: str) -> str:
    """Integreate with Gemini 2.0 flash for prompt augmentation."""

    sys_instruct = (
    "You are a creative prompt generator for image generation. Your task is to produce a single, comma-separated string "
    "that follows this structured format:\n\n"
    "1. Subject: A brief descriptor (e.g., \"1girl\", \"1boy\", \"4girls\", \"1other\", \"wano country from one piece\").\n"
    "2. Character and Source (if applicable): Include the character’s name and, if applicable, their series/source "
    "(e.g., \"souryuu asuka langley, neon genesis evangelion\"). For character-specific prompts, include detailed physical descriptions "
    "such as body, hair, eyes, and other traits.\n"
    "3. Clothing and Accessories: Describe the outfit details (e.g., \"red plugsuit\", \"black jacket\").\n"
    "4. Pose and Action: Specify the character’s pose or action (e.g., \"sitting, on throne, crossed legs\").\n"
    "5. Expressions and Focus: Note facial expressions or where the character is looking (e.g., \"head tilt, looking at viewer\").\n"
    "6. Background and Environment: For character prompts, describe the setting or background (e.g., \"outdoors\", \"cityscape\", \"building\"). "
    "For scenery prompts, if the subject is a landscape or scene (e.g., \"wano country from one piece\"), provide a detailed description "
    "of the entire landscape including its beauty, vegetation, architectural elements, and natural features.\n"
    "7. Effects: Include any visual or photographic effects (e.g., \"depth of field\", \"chromatic aberration\", \"lens flare\", \"high contrast\").\n"
    "8. Additional Elements: Add any extra details (e.g., \"holding weapon, lance of longinus \\(evangelion\\)\"). Use escaped parentheses "
    "(i.e. \"\\( ... \\)\") when including modifiers.\n\n"
    "Ensure that:\n"
    "- All elements are included as a comma-separated list with no extra punctuation or line breaks.\n"
    "- Escaped parentheses are used literally as \"\\( ... \\)\" for any additional modifiers.\n"
    "- The prompt is concise yet descriptive enough to evoke a vivid image.\n"
    "- For non-character prompts (such as landscapes or scenes), do not add any character-specific details, and ensure a detailed description "
    "of the scene is provided.\n\n"
    "Examples:\n\n"
    "Example 1 (Character Prompt):\n"
    "\"1girl, souryuu asuka langley, neon genesis evangelion, red plugsuit, sitting, on throne, crossed legs, head tilt, looking at viewer, holding weapon, lance of longinus \\(evangelion\\), depth of field, outdoors\"\n\n"
     "Example 2:\n"
    "\"kimi no nawa., building, cityscape, cloud, cloudy sky, gradient sky, lens flare, no humans, outdoors, power lines, scenery, shooting star, sky, sparkle, star \\(sky\\), starry sky, sunset, tree, utility pole\""
    )

    client = genai.Client(api_key=API_KEY)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=sys_instruct),
        contents=[prompt]
    )
    return response.text
     
def load_pipeline(model_name):

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=torch.float16,
    )
    pipeline = (
        StableDiffusionXLPipeline.from_single_file
        if MODEL.endswith(".safetensors")
        else StableDiffusionXLPipeline.from_pretrained
    )
    
    pipe = pipeline(
        model_name,
        vae=vae,
        torch_dtype=torch.float16,
        custom_pipeline="lpw_stable_diffusion_xl",
        use_safetensors=True,
        add_watermarker=False,
        use_auth_token=HF_TOKEN,
    )

    pipe.to(device)
    return pipe


def generate(
    prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    seed: int = 0,
    custom_width: int = 1024,
    custom_height: int = 1024,
    guidance_scale: float = 6.0,
    num_inference_steps: int = 25,
    sampler: str = "Euler a",
    aspect_ratio_selector: str = DEFAULT_ASPECT_RATIO,
    style_selector: str = "(None)",
    use_upscaler: bool = False,
    upscaler_strength: float = 0.55,
    upscale_by: float = 1.5,
    add_quality_tags: bool = True,
    enable_prompt_augmentation: bool = True,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
    ) -> tuple[list[str], dict]:
    
    if pipe is None:
        return "Model not loaded, please try again later."
    if not prompt:
        return "Prompt is required!"

    # Optimize performance
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_attention_slicing()
    pipe.enable_sequential_cpu_offload()

    start_time = time.time()
    upscaler_pipe = None
    backup_scheduler = None
    
    try:
        # Memory management
        torch.cuda.empty_cache()
        gc.collect()

        # Input validation
        prompt = validate_prompt(prompt)

        if enable_prompt_augmentation:
            prompt = prompt_augmentation(prompt)

        if negative_prompt:
            negative_prompt = negative_prompt.encode('utf-8').decode('utf-8')
        
        validate_dimensions(custom_width, custom_height)
        
        # Set up generation
        generator = utils.seed_everything(seed)
        width, height = utils.aspect_ratio_handler(
            aspect_ratio_selector,
            custom_width,
            custom_height,
        )

        # Process prompts
        if add_quality_tags:
            prompt = "{prompt}, masterpiece, high score, great score, absurdres".format(prompt=prompt)

        prompt, negative_prompt = utils.preprocess_prompt(
            styles, style_selector, prompt, negative_prompt
        )    

        width, height = utils.preprocess_image_dimensions(width, height)

        # Set up pipeline
        backup_scheduler = pipe.scheduler
        pipe.scheduler = utils.get_scheduler(pipe.scheduler.config, sampler)

        if use_upscaler:
            upscaler_pipe = StableDiffusionXLImg2ImgPipeline(**pipe.components)
            
        # Prepare metadata
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "resolution": f"{width} x {height}",
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "style_preset": style_selector,
            "seed": seed,
            "sampler": sampler,
            "Model": MODEL.split("/")[-1],
            "Model hash": "6327eca98b",
        }

        if use_upscaler:
            new_width = int(width * upscale_by)
            new_height = int(height * upscale_by)
            metadata["use_upscaler"] = {
                "upscale_method": "nearest-exact",
                "upscaler_strength": upscaler_strength,
                "upscale_by": upscale_by,
                "new_resolution": f"{new_width} x {new_height}",
            }
        else:
            metadata["use_upscaler"] = None
        
        logger.info(f"Starting generation with parameters: {json.dumps(metadata, indent=4)}")

        # Generate images
        if use_upscaler:
            latents = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                output_type="latent",
            ).images
            upscaled_latents = utils.upscale(latents, "nearest-exact", upscale_by)
            images = upscaler_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=upscaled_latents,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                strength=upscaler_strength,
                generator=generator,
                output_type="pil",
            ).images
        else:
            images = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                output_type="pil",
            ).images

        # Save images
        if images:
            total = len(images)
            image_paths = []
            for idx, image in enumerate(images, 1):
                progress(idx/total, desc="Saving images...")
                path = utils.save_image(image, metadata, OUTPUT_DIR)
                image_paths.append(path)
                logger.info(f"Image {idx}/{total} saved as {path}")

        generation_time = time.time() - start_time
        logger.info(f"Generation completed successfully in {generation_time:.2f} seconds")
        metadata["generation_time"] = f"{generation_time:.2f}s"
        
        return image_paths, metadata

    except GenerationError as e:
        logger.warning(f"Generation validation error: {str(e)}")
        raise gr.Error(str(e))
    except Exception as e:
        logger.exception("Unexpected error during generation")
        raise gr.Error(f"Generation failed: {str(e)}")
    finally:
        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
        if upscaler_pipe is not None:
            del upscaler_pipe
        
        if backup_scheduler is not None and pipe is not None:
            pipe.scheduler = backup_scheduler
            
        utils.free_memory()

# Model Loding and initialization
if torch.cuda.is_available():
    pipe = load_pipeline(MODEL)
    logger.info("Loaded on Device!")
else:
    pipe = None

# Process styles
styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}

css_path = "style.css"
with gr.Blocks(css=css_path) as demo:
    gr.Markdown("<h1 class='gradient-text'>imaginAIry</h1>")
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Group():
                prompt = gr.Text(
                    label="Prompt",
                    max_lines=5,
                    placeholder="Describe what you want to generate",
                    info="Enter your image generation prompt here. Be specific and descriptive for better results.",
                )
                negative_prompt = gr.Text(
                    label="Negative Prompt",
                    max_lines=5,
                    placeholder="Describe what you want to avoid",
                    value=DEFAULT_NEGATIVE_PROMPT,
                    info="Specify elements you don't want in the image.",
                )
                add_quality_tags = gr.Checkbox(
                    label="Quality Tags",
                    value=True,
                    info="Add quality-enhancing tags to your prompt automatically.",
                )
                enable_prompt_augmentation = gr.Checkbox(
                    label="Prompt Augmentation",
                    value=True,
                    info="Information, context, or instructions are added to a base prompt",
                )
            with gr.Accordion(label="More Settings", open=False):
                with gr.Group():
                    aspect_ratio_selector = gr.Radio(
                        label="Aspect Ratio",
                        choices=aspect_ratios,
                        value=DEFAULT_ASPECT_RATIO,
                        container=True,
                        info="Choose the dimensions of your image.",
                    )
                with gr.Group(visible=False) as custom_resolution:
                    with gr.Row():
                        custom_width = gr.Slider(
                            label="Width",
                            minimum=MIN_IMAGE_SIZE,
                            maximum=MAX_IMAGE_SIZE,
                            step=8,
                            value=1024,
                            info=f"Image width (must be between {MIN_IMAGE_SIZE} and {MAX_IMAGE_SIZE})",
                        )
                        custom_height = gr.Slider(
                            label="Height",
                            minimum=MIN_IMAGE_SIZE,
                            maximum=MAX_IMAGE_SIZE,
                            step=8,
                            value=1024,
                            info=f"Image height (must be between {MIN_IMAGE_SIZE} and {MAX_IMAGE_SIZE})",
                        )
                with gr.Group():
                    use_upscaler = gr.Checkbox(
                        label="Use Upscaler",
                        value=False,
                        info="Enable high-resolution upscaling.",
                    )
                    with gr.Row() as upscaler_row:
                        upscaler_strength = gr.Slider(
                            label="Strength",
                            minimum=0,
                            maximum=1,
                            step=0.05,
                            value=0.55,
                            visible=False,
                            info="Control how much the upscaler affects the final image.",
                        )
                        upscale_by = gr.Slider(
                            label="Upscale by",
                            minimum=1,
                            maximum=1.5,
                            step=0.1,
                            value=1.5,
                            visible=False,
                            info="Multiplier for the final image resolution.",
                        )
                with gr.Accordion(label="Advanced Parameters", open=False):
                    with gr.Group():
                        style_selector = gr.Dropdown(
                            label="Style Preset",
                            interactive=True,
                            choices=list(styles.keys()),
                            value="(None)",
                            info="Apply a predefined style to your generation.",
                        )
                    with gr.Group():
                        sampler = gr.Dropdown(
                            label="Sampler",
                            choices=sampler_list,
                            interactive=True,
                            value="Euler a",
                            info="Different samplers can produce varying results.",
                        )
                    with gr.Group():
                        seed = gr.Slider(
                            label="Seed",
                            minimum=0,
                            maximum=utils.MAX_SEED,
                            step=1,
                            value=0,
                            info="Set a specific seed for reproducible results.",
                        )
                        randomize_seed = gr.Checkbox(
                            label="Randomize seed",
                            value=True,
                            info="Generate a new random seed for each image.",
                        )
                    with gr.Group():
                        with gr.Row():
                            guidance_scale = gr.Slider(
                                label="Guidance scale",
                                minimum=1,
                                maximum=12,
                                step=0.1,
                                value=5.0,
                                info="Higher values make the image more closely match your prompt.",
                            )
                            num_inference_steps = gr.Slider(
                                label="Number of inference steps",
                                minimum=1,
                                maximum=50,
                                step=1,
                                value=28,
                                info="More steps generally mean higher quality but slower generation.",
                            )
        
        with gr.Column(scale=3):
            with gr.Blocks():
                run_button = gr.Button("Generate", variant="primary", elem_id="generate-button")
            result = gr.Gallery(
                label="Generated Images",
                columns=1,
                height='768px',
                preview=True,
                show_label=True,
            )
            with gr.Accordion(label="Generation Parameters", open=False):
                gr_metadata = gr.JSON(
                    label="Image Metadata",
                    show_label=True,
                )
            # gr.Examples(
            #     examples=examples,
            #     inputs=prompt,
            #     outputs=[result, gr_metadata],
            #     fn=lambda *args, **kwargs: generate(*args, use_upscaler=True, **kwargs),
            #     cache_examples=CACHE_EXAMPLES,
            # )
    

    use_upscaler.change(
        fn=lambda x: [gr.update(visible=x), gr.update(visible=x)],
        inputs=use_upscaler,
        outputs=[upscaler_strength, upscale_by],
        queue=False,
        api_name=False,
    )
    aspect_ratio_selector.change(
        fn=lambda x: gr.update(visible=x == "Custom"),
        inputs=aspect_ratio_selector,
        outputs=custom_resolution,
        queue=False,
        api_name=False,
    )

    # Combine all triggers including keyboard shortcuts
    gr.on(
        triggers=[
            prompt.submit,
            negative_prompt.submit,
            run_button.click,
        ],
        fn=utils.randomize_seed_fn,
        inputs=[seed, randomize_seed],
        outputs=seed,
        queue=False,
        api_name=False,
    ).then(
        fn=lambda: gr.update(interactive=False, value="Generating..."),
        outputs=run_button,
    ).then(
        fn=generate,
        inputs=[
            prompt,
            negative_prompt,
            seed,
            custom_width,
            custom_height,
            guidance_scale,
            num_inference_steps,
            sampler,
            aspect_ratio_selector,
            style_selector,
            use_upscaler,
            upscaler_strength,
            upscale_by,
            add_quality_tags,
            enable_prompt_augmentation
        ],
        outputs=[result, gr_metadata],
    ).then(
        fn=lambda: gr.update(interactive=True, value="Generate"),
        outputs=run_button,
    )


if __name__ =="__main__":
    demo.launch()