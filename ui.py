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
def frame_to_base64(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def pil_to_base64_str(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def describe_and_generate_story(frame_b64, genre):
    """Combined call — describe scene AND generate story beat in one LLaVA call"""
    history = " ".join(story_so_far[-3:]) if story_so_far else "This is the beginning of the story."
    
    prompt = f"""Look at this image carefully. You are a {genre} story writer.

First, notice what you see: the objects, people, setting, mood, and atmosphere.

Then, based on what you see AND this previous story context: "{history}"

Write the NEXT 3-4 sentences of the {genre} story that naturally continues from what came before, incorporating elements from the current scene. Be vivid, dramatic, and creative. Output ONLY the story continuation, nothing else."""

    response = ollama.chat(
        model="llava:7b-v1.5-q4_0",
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [frame_b64]
        }]
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

def run_one_beat(webcam_frame, genre):
    global story_so_far

    if webcam_frame is None:
        return "<div style='color:white;padding:40px'>❌ No camera frame received. Allow camera access and try again.</div>"

    # Convert gradio webcam output (numpy array) to cv2 format
    frame = cv2.cvtColor(webcam_frame, cv2.COLOR_RGB2BGR)
    frame_b64 = frame_to_base64(frame)

    # Single combined LLaVA call — faster!
    beat = describe_and_generate_story(frame_b64, genre)
    story_so_far.append(beat)

    # Generate art
    art_img = generate_art(beat, genre)
    art_b64 = pil_to_base64(art_img)

    # Build story HTML
    story_html = ""
    for i, b in enumerate(story_so_far):
        story_html += f'<p style="margin-bottom:16px"><strong style="color:#f0d080">Beat {i+1}:</strong> {b}</p>'

    html = f"""
    <div style="
        position: relative;
        width: 100%;
        min-height: 600px;
        border-radius: 16px;
        overflow: hidden;
        font-family: Georgia, serif;
    ">
        <img src="data:image/png;base64,{art_b64}" style="
            position: absolute;
            top: 0; left: 0;
            width: 100%; height: 100%;
            object-fit: cover;
            z-index: 0;
        "/>
        <div style="
            position: absolute;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: linear-gradient(to bottom, rgba(0,0,0,0.2), rgba(0,0,0,0.88));
            z-index: 1;
        "></div>
        <div style="
            position: relative;
            z-index: 2;
            padding: 40px;
            color: white;
            text-shadow: 1px 1px 4px rgba(0,0,0,0.9);
            line-height: 1.8;
            font-size: 1.05em;
        ">
            <h2 style="
                font-size: 1.6em;
                margin-bottom: 24px;
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
    return "<div style='min-height:200px; display:flex; align-items:center; justify-content:center; opacity:0.5; color:white'>Story reset. Press Generate Next Beat to begin!</div>"

# ── Gradio UI ────────────────────────────────────────────
with gr.Blocks(title="LifeFrame") as demo:
    gr.HTML("""
    <div style="text-align:center; padding: 20px 0;">
        <h1 style="font-size:2.5em; margin:0;">🎬 LifeFrame</h1>
        <p style="opacity:0.7; font-style:italic;">Point your camera at anything. Pick a mood. Watch your world become a story.</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            # Live webcam — always on, asks for permission automatically
            webcam = gr.Image(
            sources=["webcam"],
            streaming=True,
            label="📷 Live Camera"
             )
            genre_selector = gr.Dropdown(
                choices=["fantasy", "horror", "romance", "sci-fi"],
                value="fantasy",
                label="🎭 Genre"
            )
            with gr.Row():
                generate_btn = gr.Button("📸 Generate Next Beat", variant="primary")
                reset_btn = gr.Button("🔄 Reset")

        with gr.Column(scale=2):
            story_display = gr.HTML(
                value="<div style='min-height:400px; display:flex; align-items:center; justify-content:center; opacity:0.5; color:white'>Allow camera access, then press Generate Next Beat...</div>"
            )

    generate_btn.click(
        fn=run_one_beat,
        inputs=[webcam, genre_selector],
        outputs=[story_display]
    )

    reset_btn.click(
        fn=reset_story,
        inputs=[],
        outputs=[story_display]
    )

if __name__ == "__main__":
    demo.launch()