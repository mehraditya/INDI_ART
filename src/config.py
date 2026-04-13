"""
Configuration settings for Indian Art Generator
"""
import os
from typing import Dict

class Config:
    """Application configuration"""

    # Model Settings
    BASE_MODEL = os.getenv("BASE_MODEL", "SG161222/Realistic_Vision_V5.1_noVAE")

    # LoRA Weights Path
    # Options:
    # 1. Local: "./models/..."
    # 2. HuggingFace Hub: "username/repo-name"
    LORA_PATH = os.getenv("LORA_PATH", "mehradity/lora_weights")
    LORA_ADAPTER_NAME = os.getenv("LORA_ADAPTER_NAME", "indian_art")

    # Generation Defaults
    DEFAULT_WIDTH = int(os.getenv("DEFAULT_WIDTH", "512"))
    DEFAULT_HEIGHT = int(os.getenv("DEFAULT_HEIGHT", "512"))
    DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "20" if os.getenv("DEVICE") == "cpu" else "30"))
    DEFAULT_GUIDANCE = float(os.getenv("DEFAULT_GUIDANCE", "7.5"))
    DEFAULT_LORA_SCALE = float(os.getenv("DEFAULT_LORA_SCALE", "0.8"))

    # Device
    DEVICE = os.getenv("DEVICE", "cuda")

    # Cache dir for models (use /tmp for HF Spaces)
    CACHE_DIR = os.getenv("CACHE_DIR", "/tmp/huggingface")

    # Art-specific Negative Prompts (Cultural Preservation)
    # Prevents Western art styles and modern elements
    DEFAULT_NEGATIVE_PROMPT = """photorealistic, 3d render, photography, western art style, 
    oil painting, renaissance, baroque, impressionist, modern abstract, 
    graffiti, street art, pop art, contemporary western, 
    urban elements, skyscrapers, cars, modern architecture, 
    blurry, low quality, distorted, deformed, ugly, duplicate, 
    bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limbs, 
    watermark, signature, text, logo, cropped, out of frame, 
    worst quality, low resolution, jpeg artifacts, noise, grainy, 
    grayscale, monochrome, boring, plain background, 
    western clothing, modern fashion, jeans, t-shirt, sunglasses, sneakers"""

    # Art Style Prefixes (Auto-enhance prompts)
    ART_STYLES: Dict[str, str] = {
        "madhubani": "madhubani painting, intricate double border, geometric patterns, natural dyes, no gaps, fine line work, traditional motifs,",

        "warli": "warli tribal art, white rice paste drawing on dark background, stick figures, traditional hut, daily life scene, harvest festival,",

        "gond": "gond painting, intricate dot patterns, bright flat colors, nature inspired, line work, patangarh style, dighna patterns,",

        "pattachitra": "pattachitra scroll art, mythological theme, bold outlines, natural colors, intricate border, odisha tradition, palm leaf texture,",

        "tanjore": "tanjore painting, gold foil work, rich colors, devotional iconography, embossed texture, tamil nadu style, semi-precious stones,",

        "miniature": "mughal miniature painting, fine detailing, persian influence, solid colors, intricate borders, court scene,",

        "kalamkari": "kalamkari art, hand drawn with kalam, natural dyes, mythological narratives, floral motifs, andhra pradesh style,",

        "phad": "phad painting, scroll narrative, bold lines, bright orange red colors, rajasthani folk, deity stories,",

        "mysore": "mysore painting, muted colors, intricate details, gold leaf, karnataka tradition, hindu gods,",

        "none": ""
    }

    # UI Configuration
    GRADIO_THEME = os.getenv("GRADIO_THEME", "soft")
    GRADIO_SHARE = os.getenv("GRADIO_SHARE", "false").lower() == "true"

    @classmethod
    def validate(cls):
        """Validate configuration"""
        if cls.LORA_PATH is None:
            print("⚠️ Warning: LORA_PATH not set. Using base SDXL model only.")
        return True
