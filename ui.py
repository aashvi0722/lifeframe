import gradio as gr
import cv2
import ollama
import base64
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import io
import threading
import time

# ── State ──────────────────────────────────────────────
story_so_far = []
is_running = False
current_genre = "fantasy"

genre_styles = {
    "fantasy":  "fantasy art, magical, ethereal, detailed illustration, artstation",
    "horror":   "dark horror art, eerie, gothic, dramatic lighting, scary",
    "romance":  "soft watercolor, warm tones, romantic, dreamy illustration",
    "sci-fi":   "cyberpunk, futuristic, neon lights, sci-fi concept art",
}

# ── Load SD once at startup ─────────────────────────────
print("🎨 Loading Stable Diffusion v1.5...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None
)
pipe = pipe.to("cuda")
print("✅ Stable Diffusion ready!")

# ── Core pipeline functions ─────────────────────────────
def capture_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame

def frame_to_base64(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def frame_to_pil(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def describe_scene(frame_b64):
    response = ollama.chat(
        model="llava:7b-v1.5-q4_0",
        messages=[{
            "role": "user",
            "content": "Describe what you see in this image in 2-3 sentences. Focus on objects, people, mood and setting.",
            "images": [frame_b64]
        }]
    )
    return response['message']['content']

def generate_story_beat(scene_description, genre):
    history = " ".join(story_so_far[-3:])
    prompt = f"""You are a {genre} story writer.
Previous story: {history}
New scene observed: {scene_description}
Continue the story in 3-4 sentences. Be creative and dramatic. Only output the story, nothing else."""
    response = ollama.chat(
        model="llava:7b-v1.5-q4_0",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']

def generate_art(story_beat, genre):
    style = genre_styles.get(genre, genre_styles["fantasy"])
    # Extract key visual elements for better art prompt
    prompt = f"{story_beat[:150]}, {style}, highly detailed, 8k"
    image = pipe(
        prompt,
        num_inference_steps=20,
        guidance_scale=7.5
    ).images[0]
    return image

# ── Gradio pipeline step ────────────────────────────────
def run_one_beat(genre):
    global story_so_far

    # 1. Capture
    frame = capture_frame()
    if frame is None:
        return None, "❌ Camera not found.", None

    cam_img = frame_to_pil(frame)

    # 2. Describe
    frame_b64 = frame_to_base64(frame)
    scene = describe_scene(frame_b64)

    # 3. Story
    beat = generate_story_beat(scene, genre)
    story_so_far.append(beat)
    full_story = "\n\n".join([f"**Beat {i+1}:** {b}" for i, b in enumerate(story_so_far)])

    # 4. Art
    art_img = generate_art(beat, genre)

    return cam_img, full_story, art_img

def reset_story():
    global story_so_far
    story_so_far = []
    return None, "Story reset. Press **Generate Next Beat** to begin!", None

# ── Gradio UI ───────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft(), title="LifeFrame") as demo:
    gr.Markdown("""
    # 🎬 LifeFrame — Real-Time World to Storybook
    *Point your camera at anything. Pick a mood. Watch your world become a story.*
    """)

    with gr.Row():
        genre_selector = gr.Dropdown(
            choices=["fantasy", "horror", "romance", "sci-fi"],
            value="fantasy",
            label="🎭 Story Genre",
            scale=1
        )
        generate_btn = gr.Button("📸 Generate Next Beat", variant="primary", scale=2)
        reset_btn = gr.Button("🔄 Reset Story", scale=1)

    with gr.Row():
        cam_output = gr.Image(label="📷 Live Capture", type="pil")
        story_output = gr.Markdown(label="📖 Your Story", value="Press **Generate Next Beat** to begin!")
        art_output = gr.Image(label="🎨 Generated Art", type="pil")

    generate_btn.click(
        fn=run_one_beat,
        inputs=[genre_selector],
        outputs=[cam_output, story_output, art_output]
    )

    reset_btn.click(
        fn=reset_story,
        inputs=[],
        outputs=[cam_output, story_output, art_output]
    )

if __name__ == "__main__":
    demo.launch()