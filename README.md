# 🎬 LifeFrame — Real-Time World to Storybook Engine

> Point your camera at anything. Pick a mood. Watch your world become a living, illustrated story.

![LifeFrame Demo](demo.png)

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
