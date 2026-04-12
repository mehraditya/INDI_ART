"""
Utility functions
"""
import os
from pathlib import Path


def ensure_directories():
    """Create necessary directories"""
    dirs = ["outputs", "models", "/tmp/huggingface"]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def save_image_with_metadata(image, metadata, output_dir="outputs"):
    """Save image and metadata file"""
    import json
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed = metadata.get("seed", "unknown")

    img_path = os.path.join(output_dir, f"art_{timestamp}_{seed}.png")
    meta_path = os.path.join(output_dir, f"art_{timestamp}_{seed}.json")

    image.save(img_path)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return img_path
