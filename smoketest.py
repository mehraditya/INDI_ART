"""
Smoke test for Indian Art Generator.
Run locally before deploying to verify the pipeline works.

Usage:
    python smoke_test.py

Tests CPU mode with 1-step generation to verify imports, model loading,
and image generation without waiting for full inference.
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(__file__))


def test_imports():
    """Verify all modules import cleanly."""
    try:
        from src.config import Config
        from src.model import IndianArtGenerator
        from src.utils import ensure_directories
        return Config, IndianArtGenerator
    except Exception as e:
        print(f"Import failed: {e}")
        traceback.print_exc()
        return None


def test_config(Config):
    """Validate configuration."""
    try:
        Config.validate()
        required = {"madhubani", "warli", "gond", "none"}
        missing = required - set(Config.ART_STYLES.keys())
        if missing:
            raise AssertionError(f"Missing keys: {missing}")
        return True
    except Exception as e:
        print(f"Config failed: {e}")
        return False


def test_model_init(IndianArtGenerator):
    """Test generator instantiation."""
    try:
        gen = IndianArtGenerator()
        assert gen.current_adapter_name is None
        return gen
    except Exception as e:
        print(f"Model init failed: {e}")
        return None


def test_model_load(gen):
    """Load model without LoRA (CPU mode)."""
    try:
        os.environ["DEVICE"] = "cpu"
        os.environ["LORA_PATH"] = ""
        gen.load_model(lora_path="")
        return True
    except Exception as e:
        print(f"Model load failed: {e}")
        traceback.print_exc()
        return False


def test_generation(gen):
    """Run minimal generation test."""
    try:
        img, meta = gen.generate(
            prompt="peacock with geometric feathers",
            art_style="madhubani",
            num_inference_steps=1,
            width=256,
            height=256,
            seed=42,
        )
        assert img is not None
        assert img.size == (256, 256)
        assert meta.get("seed") == 42
        img.save("/tmp/smoke_test_output.png")
        return True
    except Exception as e:
        print(f"Generation failed: {e}")
        traceback.print_exc()
        return False


def main():
    print("Smoke Test Starting")
    print("=" * 50)
    
    result = test_imports()
    if not result:
        print("FAILED: Imports")
        sys.exit(1)
    Config, IndianArtGenerator = result
    print("OK: Imports")
    
    if not test_config(Config):
        print("FAILED: Config")
        sys.exit(1)
    print("OK: Config")
    
    gen = test_model_init(IndianArtGenerator)
    if not gen:
        print("FAILED: Model Init")
        sys.exit(1)
    print("OK: Model Init")
    
    print("Loading model (downloads ~1.8GB on first run)...")
    if not test_model_load(gen):
        print("FAILED: Model Load")
        sys.exit(1)
    print("OK: Model Load")
    
    print("Running 1-step generation test...")
    if not test_generation(gen):
        print("FAILED: Generation")
        sys.exit(1)
    print("OK: Generation")
    print("Output saved to /tmp/smoke_test_output.png")
    
    print("=" * 50)
    print("All tests passed. Safe to deploy.")


if __name__ == "__main__":
    main()