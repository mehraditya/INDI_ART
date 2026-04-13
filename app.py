"""
Indian Art SD LoRA API
Main application entry point
Optimized for HuggingFace Spaces ZeroGPU
"""
import os
import sys
import json
import gradio as gr
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.config import Config
from src.model import IndianArtGenerator
from src.utils import ensure_directories

# Global generator instance
generator = IndianArtGenerator()

# Hardcoded generation settings
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE = 7.5
DEFAULT_LORA_SCALE = 0.8
DEFAULT_SEED = -1  # Random


# def get_generator():
#     """Lazy initialization of model"""
#     if generator.pipe is None:
#         generator.load_model()
#     return generator

try:
    import spaces
    ZERO_GPU_AVAILABLE = True
except ImportError:
    ZERO_GPU_AVAILABLE = False

def conditional_gpu_decorator(fn):
    if ZERO_GPU_AVAILABLE:
        return spaces.GPU(duration=60)(fn)
    return fn

@conditional_gpu_decorator
def generate_api(prompt: str, art_style: str):
    """
    Generate image from prompt and art style.
    All generation parameters are hardcoded constants.
    """

    if not prompt or not prompt.strip():
        raise gr.Error("Prompt cannot be empty")
    if len(prompt) > 1000:
        raise gr.Error("Prompt too long (max 1000 chars)")

    # gen = get_generator()

    #Lazy-load inside GPU context, so ZEROGPU can map weights onto GPU
    if generator.pipe is None:
        generator.load_model() 
    

    image, metadata = generator.generate(
        prompt=prompt,
        negative_prompt="",  # Uses default negatives from config
        art_style=art_style,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        num_inference_steps=DEFAULT_STEPS,
        guidance_scale=DEFAULT_GUIDANCE,
        lora_scale=DEFAULT_LORA_SCALE,
        seed=DEFAULT_SEED,
        num_images=1
    )

    return image, json.dumps(metadata, indent=2)


def create_ui():
    """Create minimal Gradio interface"""

    css = """
    .gradio-container {max-width: 800px !important;}
    .image-output {height: auto !important; max-height: 600px;}
    """

    with gr.Blocks(
        title="Indian Art Generator",
        theme=gr.themes.Soft(),
        css=css
    ) as demo:

        gr.Markdown("""
        <div style="text-align: center;">
        <h1>Indian Traditional Art Generator</h1>
        <p>Describe your scene and select an art style</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="peacock surrounded by floral patterns",
                    lines=3
                )

                art_style = gr.Dropdown(
                    choices=list(Config.ART_STYLES.keys()),
                    value="none",
                    label="Art Style Prefix",
                    info="Select a style to enhance your prompt, or 'none' for raw prompt"
                )

                generate_btn = gr.Button(
                    "Generate",
                    variant="primary",
                    size="lg"
                )

                # Hidden constants display
                with gr.Accordion("Generation Settings"):
                    gr.Markdown(f"""
                    - Resolution: {DEFAULT_WIDTH}x{DEFAULT_HEIGHT}
                    - Steps: {DEFAULT_STEPS}
                    - Guidance: {DEFAULT_GUIDANCE}
                    - LoRA Scale: {DEFAULT_LORA_SCALE}
                    - Seed: Random
                    """)

            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Generated Artwork",
                    type="pil",
                    elem_classes=["image-output"],
                    format="png"
                )

                output_metadata = gr.JSON(
                    label="Metadata"
                )

        # Examples
        gr.Examples(
            examples=[
                ["peacock with elaborate geometric feathers", "madhubani"],
                ["village scene with farmers and traditional huts", "warli"],
                ["tiger in dense jungle with dotted patterns", "gond"],
                ["radha krishna under flowering tree", "pattachitra"],
                ["goddess lakshmi with gold ornaments", "tanjore"],
                ["elephant decorated for festival procession", "miniature"],
            ],
            inputs=[prompt, art_style],
            label="Example Prompts"
        )

        generate_btn.click(
            fn=generate_api,
            inputs=[prompt, art_style],
            outputs=[output_image, output_metadata]
        )

    return demo


if __name__ == "__main__":
    ensure_directories()
    Config.validate()

    print("Starting Indian Art Generator...")
    print(f"Base Model: {Config.BASE_MODEL}")
    print(f"LoRA Path: {Config.LORA_PATH or 'Not configured'}")
    print(f"Hardcoded settings: {DEFAULT_WIDTH}x{DEFAULT_HEIGHT}, {DEFAULT_STEPS} steps")

    demo = create_ui()

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=True,
        show_api=False,
        quiet=True
    )