# Indian Art SDXL LoRA API

Generate Indian traditional art (Madhubani, Warli, Gond, Pattachitra, Tanjore, etc.) using SDXL + LoRA. Optimized for Hugging Face Spaces (ZeroGPU).

---

## Features
- Built-in negative prompts to reduce Western/modern bias  
- Works on Hugging Face Spaces (ZeroGPU)  
- Supports LoRA (local or HF Hub)  
- Simple API + Gradio UI  

---

## Quick Start (Local)

```bash
git clone https://github.com/yourusername/INDI_ART.git
cd INDI_ART

python -m venv venv
venv\Scripts\activate  # or source venv/bin/activate

pip install -r requirements.txt
```

### Configure LoRA

.env:

```
LORA_PATH=your-username/indian-art-lora
```

or local:

```
LORA_PATH=./models/lora.safetensors
```

### Run

```bash
python app.py
```

- UI: http://localhost:7860  
- API: http://localhost:7860/api/predict/generate_api  

---

## Hugging Face Spaces (Recommended)

1. Create Space (Gradio, ZeroGPU)  
2. Connect to GitHub repo  
3. Set LORA_PATH if using HF weights  

App URL:
https://<username>-<space>.hf.space

---

## API Usage

### Python

```python
from gradio_client import Client

client = Client("https://<space>.hf.space")

result = client.predict(
    "peacock with geometric feathers",
    "madhubani",
    api_name="/generate_api"
)
```

### cURL

```bash
curl -X POST https://<space>.hf.space/api/predict/generate_api \
-H "Content-Type: application/json" \
-d '{"data": ["warli village scene", "warli"]}'
```

---

## Configuration

| Variable | Description | Default |
|----------|------------|--------|
| LORA_PATH | LoRA weights | None |
| BASE_MODEL | Base model | SDXL |
| DEFAULT_STEPS | Steps | 30 |
| DEFAULT_GUIDANCE | CFG | 7.5 |
| DEFAULT_LORA_SCALE | LoRA strength | 0.8 |
| DEVICE | cuda/cpu | cuda |
| CACHE_DIR | Cache dir | /tmp |

---

## Notes

- Use SD-compatible LoRA only  
- Keep resolution ≤1024  
- ZeroGPU has cold starts  

---

## Structure

```
src/
  config.py
  model.py
app.py
requirements.txt
Dockerfile
```

---

## License

MIT
