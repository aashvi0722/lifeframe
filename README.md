# 🎬 LifeFrame — Real-Time World to Storybook Engine

> Point your camera at anything. Pick a mood. Watch your world become a living, illustrated story

---

## ✨ What is LifeFrame?

LifeFrame is a real-time multimodal AI pipeline that transforms your live camera feed into a continuously evolving illustrated story — entirely on-device, no external APIs.

**How it works:**
1. 📷 **Live webcam** streams your surroundings in real time
2. 👁️ **LLaVA 7B** (Vision-Language Model) analyzes the scene AND writes the next story beat in a single inference call
3. ✍️ **Narrative continuity engine** maintains story coherence across beats using a sliding context window
4. 🎨 **Stable Diffusion v1.5** generates matching artwork in the genre's visual style
5. 🖼️ **Cinematic UI** displays AI art as a full background with story text overlaid

**100% local inference — zero external APIs — runs on a single consumer GPU**

---

## 🎥 Live Demo
<img width="1891" height="926" alt="Screenshot 2026-06-26 122527" src="https://github.com/user-attachments/assets/b3f22560-092b-4069-acde-4b674be01ae2" />


| Fantasy | Horror | Sci-Fi | Romance |
|---------|--------|--------|---------|
| Magical realm, enchanted forests | Gothic shadows, eerie presence | Neon cyberpunk, alien contact | Soft watercolor, dreamy scenes |

---

## 🏗️ Pipeline Architecture
📷 Live Webcam (OpenCV)
↓
🖼️ Frame Capture (base64 encoded JPEG)
↓
🧠 LLaVA 7B — Single Combined Call
[Scene Understanding + Story Generation]
↓
✍️ Continuity Engine (sliding 3-beat context window)
↓
🎨 Stable Diffusion v1.5 (FP16, CUDA)
[Genre-controlled style transfer]
↓
🖥️ Gradio Cinematic UI
[Art as background + Story overlay]

---

## ⚡ Performance

| Metric | Value |
|--------|-------|
| GPU | NVIDIA RTX 3050 Laptop |
| VRAM usage | ~4GB (FP16 quantization) |
| SD inference steps | 20 |
| Time per beat | ~20 seconds |
| LLaVA model size | 7B parameters (4-bit quantized) |
| SD model | Stable Diffusion v1.5 (FP16) |
| API dependencies | Zero |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Vision-Language Model | LLaVA 7B (4-bit quantized via Ollama) |
| Image Generation | Stable Diffusion v1.5 (FP16) |
| CV Pipeline | OpenCV |
| Model Serving | Ollama (local) |
| Deep Learning | PyTorch + CUDA |
| Diffusion Library | HuggingFace Diffusers |
| Web UI | Gradio |
| Language | Python 3.11 |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11
- NVIDIA GPU with CUDA support (4GB+ VRAM)
- [Ollama](https://ollama.com) installed

### Installation

```bash
# Clone the repo
git clone https://github.com/YOURUSERNAME/lifeframe.git
cd lifeframe

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Pull LLaVA model (~4GB)
ollama pull llava:7b-v1.5-q4_0
```

### Run

```bash
python ui.py
```

Open **http://127.0.0.1:7860** → allow camera access → pick a genre → click **Generate Next Beat**

---

## 🧠 Technical Highlights

### Combined LLaVA Inference
Instead of two separate model calls (scene description → story generation), LifeFrame uses a single LLaVA call that simultaneously understands the visual scene and generates the next story beat — cutting latency by ~50%.

### Narrative Continuity Engine
A sliding context window passes the last 3 story beats into every new LLaVA call, ensuring the story evolves coherently across unlimited beats rather than restarting each time.

### Genre-Controlled Style Transfer
Each genre maps to a carefully tuned Stable Diffusion prompt suffix — fantasy uses artstation illustration style, horror uses gothic dramatic lighting, sci-fi uses cyberpunk neon aesthetics, and romance uses soft watercolor rendering.

### FP16 Quantization
Stable Diffusion runs in 16-bit floating point precision, reducing VRAM usage by ~50% vs full FP32 — making it viable on consumer laptop GPUs like the RTX 3050.

---

## 📁 Project Structure

lifeframe/n
├── app.py # Core pipeline (terminal version)
├── ui.py # Gradio web UI (main app)
├── requirements.txt # Python dependencies
└── README.md

---

## 💡 Future Improvements

- [ ] Auto-mode: generate beats automatically every N seconds
- [ ] Export full story as PDF with artwork
- [ ] Support for more genres and art styles via LoRA
- [ ] Voice narration of story beats using TTS
- [ ] Mobile-friendly UI

---

## 👤 Author

**Aashvi** — 4th Year Computer Science Student  
Building at the intersection of Computer Vision, Generative AI, and real-time systems.

---

## 📄 License

MIT License — feel free to use, modify, and build on this project.
