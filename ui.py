import gradio as gr
import cv2
import ollama
import base64
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import io

# ── State ──────────────────────────────────────────────
story_so_far = []
current_genre = "fantasy"

genre_styles = {
    "fantasy":  "fantasy art, magical, ethereal, detailed illustration, artstation",
    "horror":   "dark horror art, eerie, gothic, dramatic lighting, scary",
    "romance":  "soft watercolor, warm tones, romantic, dreamy illustration",
    "sci-fi":   "cyberpunk, futuristic, neon lights, sci-fi concept art",
}

# ── Load SD ─────────────────────────────────────────────
print("🎨 Loading Stable Diffusion v1.5...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None
)
pipe = pipe.to("cuda")
print("✅ Ready!")

# ── Core functions ───────────────────────────────────────
def capture_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def frame_to_base64(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

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

def generate_story_beat(scene, genre):
    history = " ".join(story_so_far[-3:])
    prompt = f"""You are a {genre} story writer.
Previous story: {history}
New scene observed: {scene}
Continue the story in 3-4 sentences. Be creative and dramatic. Only output the story, nothing else."""
    response = ollama.chat(
        model="llava:7b-v1.5-q4_0",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']

def generate_art(beat, genre):
    style = genre_styles.get(genre, genre_styles["fantasy"])
    prompt = f"{beat[:150]}, {style}, highly detailed, 8k"
    return pipe(prompt, num_inference_steps=20, guidance_scale=7.5).images[0]

def pil_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def run_one_beat(genre):
    global story_so_far

    frame = capture_frame()
    if frame is None:
        return "<p>Camera not found.</p>"

    frame_b64 = frame_to_base64(frame)
    scene = describe_scene(frame_b64)
    beat = generate_story_beat(scene, genre)
    story_so_far.append(beat)
    art_img = generate_art(beat, genre)

    # Convert art to base64 for embedding in HTML
    art_b64 = pil_to_base64(art_img)

    # Build story HTML
    story_html = ""
    for i, b in enumerate(story_so_far):
        story_html += f'<p><strong>Beat {i+1}:</strong> {b}</p>'

    html = f"""
    <div style="
        position: relative;
        width: 100%;
        min-height: 600px;
        border-radius: 16px;
        overflow: hidden;
        font-family: Georgia, serif;
    ">
        <!-- Background Art -->
        <img src="data:image/png;base64,{art_b64}" style="
            position: absolute;
            top: 0; left: 0;
            width: 100%; height: 100%;
            object-fit: cover;
            z-index: 0;
        "/>

        <!-- Dark overlay -->
        <div style="
            position: absolute;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: linear-gradient(to bottom, rgba(0,0,0,0.3), rgba(0,0,0,0.85));
            z-index: 1;
        "></div>

        <!-- Story Text -->
        <div style="
            position: relative;
            z-index: 2;
            padding: 40px;
            color: white;
            text-shadow: 1px 1px 4px rgba(0,0,0,0.9);
        ">
            <h2 style="
                font-size: 1.8em;
                margin-bottom: 20px;
                letter-spacing: 2px;
                text-transform: uppercase;
                color: #f0d080;
            ">📖 LifeFrame Story</h2>
            {story_html}
        </div>
    </div>
    """
    return html

def reset_story():
    global story_so_far
    story_so_far = []
    return "<p style='color:white'>Story reset. Press <strong>Generate Next Beat</strong> to begin!</p>"

# ── Gradio UI ────────────────────────────────────────────
with gr.Blocks(title="LifeFrame") as demo:
    gr.HTML("""
    <div style="text-align:center; padding: 20px 0;">
        <h1 style="font-size:2.5em; margin:0;">🎬 LifeFrame</h1>
        <p style="opacity:0.7; font-style:italic;">Point your camera at anything. Pick a mood. Watch your world become a story.</p>
    </div>
    """)

    with gr.Row():
        genre_selector = gr.Dropdown(
            choices=["fantasy", "horror", "romance", "sci-fi"],
            value="fantasy",
            label="🎭 Genre",
            scale=1
        )
        generate_btn = gr.Button("📸 Generate Next Beat", variant="primary", scale=3)
        reset_btn = gr.Button("🔄 Reset", scale=1)

    story_display = gr.HTML(
        value="<div style='min-height:200px; display:flex; align-items:center; justify-content:center; opacity:0.5'>Press Generate Next Beat to begin your story...</div>"
    )

    generate_btn.click(fn=run_one_beat, inputs=[genre_selector], outputs=[story_display])
    reset_btn.click(fn=reset_story, inputs=[], outputs=[story_display])

if __name__ == "__main__":
    demo.launch()