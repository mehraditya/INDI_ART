---
title: Indian Art Generator
emoji: 🎨
colorFrom: orange
colorTo: red
sdk: gradio
sdk_version: "4.42.0"
python_version: "3.12"
app_file: app.py
pinned: false
---

# Indian Art LoRA API

Generate Indian traditional art (Madhubani, Warli, Gond, Pattachitra, Tanjore, etc.) using SD 1.5 + LoRA.

---

## Features
- Built-in negative prompts to reduce Western/modern bias
- Supports LoRA (local or HF Hub)
- Simple API + Gradio UI

---

## Quick Start (Local)

```bash
git clone https://github.com/yourusername/INDI_ART.git
cd INDI_ART

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Configure LoRA

`.env`:
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
|----------|-------------|---------|
| LORA_PATH | LoRA weights (HF Hub ID or local path) | None |
| BASE_MODEL | Base model repo | Realistic_Vision_V5.1_noVAE |
| DEFAULT_STEPS | Inference steps | 30 |
| DEFAULT_GUIDANCE | CFG scale | 7.5 |
| DEFAULT_LORA_SCALE | LoRA strength | 0.8 |
| DEVICE | cuda or cpu | cuda |
| CACHE_DIR | Model cache directory | /tmp |

---

## Notes

- Uses SD 1.5 compatible LoRA (not SDXL)
- Keep resolution ≤ 768×768 on CPU spaces
- CPU spaces: expect 3–8 min per generation at 30 steps

---

## Structure

```
src/
  config.py
  model.py
  utils.py
app.py
requirements.txt
Dockerfile
```

---

## License

MIT