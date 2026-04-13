"""
Model loading and inference logic
"""
import torch
import random
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
from typing import Optional, Tuple, Dict
import json
import os

from .config import Config


class IndianArtGenerator:
    """
    SD Generator with LoRA support for Indian Traditional Art
    """

    def __init__(self):
        self.pipe = None
        self.device = Config.DEVICE if torch.cuda.is_available() else "cpu"
        self.is_lora_loaded = False
        self.current_adapter_name = None

        print(f"Device: {self.device}")

    def load_model(self, lora_path: Optional[str] = None):
        """
        Load SD base model and apply LoRA weights

        Args:
            lora_path: Path to LoRA weights (overrides Config.LORA_PATH)
        """
        print(f"Loading base model: {Config.BASE_MODEL}")

        # Load pipeline with memory efficient settings
        self.pipe = StableDiffusionPipeline.from_pretrained(
            Config.BASE_MODEL,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            # variant="fp16" if self.device == "cuda" else None,
            use_safetensors=True,
            cache_dir=Config.CACHE_DIR,
            local_files_only=False
        )

        # Use DPM++ 2M Karras scheduler for quality generations
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True
        )

        # Move to device
        self.pipe = self.pipe.to(self.device)
        

        # Enable VAE slicing for memory efficiency (SD 1.5)
        self.pipe.enable_vae_slicing()

        # Enable CPU offloading if low VRAM (optional optimization)
        # self.pipe.enable_model_cpu_offload()

        # Load LoRA weights if provided
        weights_path = lora_path or Config.LORA_PATH
        if weights_path:
            self.load_lora(weights_path)
        else:
            print("No LoRA weights loaded (using base SD)")

        print("Model loaded successfully")

    
    def load_lora(self, lora_path: str = None, adapter_name: str = None):
        adapter = adapter_name or Config.LORA_ADAPTER_NAME
        path = lora_path or Config.LORA_PATH
        print(f"Loading LoRA from HF Hub: {path} (adapter: {adapter})")
        try:
            # HF Hub repo ID or local path - load_lora_weights handles both
            self.pipe.load_lora_weights(path, adapter_name=adapter)
            self.pipe.set_adapters([adapter], [1.0])  # List format for newer diffusers
            self.is_lora_loaded = True
            self.current_adapter_name = adapter
            print(f"LoRA '{adapter}' loaded successfully")
        except Exception as e:
            print(f"Error loading LoRA: {e}")
            self.is_lora_loaded = False
            raise  # Fail fast in production

    def set_lora_scale(self, scale: float = 0.8):
        """
        Adjust LoRA influence (0.0 = base model, 1.0 = full LoRA)

        Args:
            scale: Adapter weight between 0.0 and 1.0
        """
        if self.is_lora_loaded and self.current_adapter_name:
            self.pipe.set_adapters(self.current_adapter_name, [scale])

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        art_style: str = "none",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        lora_scale: float = 0.8,
        seed: int = -1,
        num_images: int = 1
    ) -> Tuple[Image.Image, Dict]:
        """
        Generate image with full parameter control

        Returns:
            (PIL Image, metadata dict)
        """


        if self.pipe is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Safety: Limit resolution to prevent OOM on free tiers
        max_pixels = 786432  # 768x768 max for SD 1.5 on CPU/ZeroGPU
        if width * height > max_pixels:
            raise ValueError(f"Resolution {width}x{height} exceeds safe limit. Max: 768x768")


        # Apply art style prefix
        if art_style != "none" and art_style in Config.ART_STYLES:
            enhanced_prompt = Config.ART_STYLES[art_style] + " " + prompt
        else:
            enhanced_prompt = prompt

        # Combine negative prompts
        full_negative = Config.DEFAULT_NEGATIVE_PROMPT
        if negative_prompt:
            full_negative += ", " + negative_prompt

        # Set LoRA scale
        if self.is_lora_loaded and self.current_adapter_name:
            self.set_lora_scale(lora_scale)

        # Handle seed
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        print(f"Generating: {enhanced_prompt[:60]}...")
        print(f"Steps: {num_inference_steps}, CFG: {guidance_scale}, "
              f"LoRA: {lora_scale}, Size: {width}x{height}, Seed: {seed}")

        # Generate
        result = self.pipe(
            prompt=enhanced_prompt,
            negative_prompt=full_negative,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            generator=generator
        )

        image = result.images[0]

        # Compile metadata
        metadata = {
            "prompt": enhanced_prompt,
            "original_prompt": prompt,
            "negative_prompt": full_negative,
            "art_style": art_style,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "lora_scale": lora_scale if self.is_lora_loaded else None,
            "seed": seed,
            "model": Config.BASE_MODEL,
            "lora_loaded": self.is_lora_loaded
        }

        return image, metadata

    def get_model_info(self) -> Dict:
        """Get current model status"""
        return {
            "base_model": Config.BASE_MODEL,
            "lora_path": Config.LORA_PATH,
            "lora_loaded": self.is_lora_loaded,
            "device": self.device,
            "model_loaded": self.pipe is not None
        }