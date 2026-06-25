import cv2
import ollama
import base64
import time
from PIL import Image
import io
import torch
from diffusers import StableDiffusionPipeline

# Story state
story_so_far = []
genre = "fantasy"

# Art style per genre
genre_styles = {
    "fantasy": "fantasy art, magical, ethereal, detailed illustration, artstation",
    "horror": "dark horror art, eerie, gothic, dramatic lighting, scary",
    "romance": "soft watercolor, warm tones, romantic, dreamy illustration",
    "sci-fi": "cyberpunk, futuristic, neon lights, sci-fi concept art",
}

print("🎨 Loading Stable Diffusion v1.5... (downloading ~4GB first time)")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None
)
pipe = pipe.to("cuda")
print("✅ Stable Diffusion ready!\n")

def capture_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Failed to capture frame")
        return None
    return frame

def frame_to_base64(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

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

def generate_story_beat(scene_description):
    history = " ".join(story_so_far[-3:])
    prompt = f"""You are a {genre} story writer. 
Previous story: {history}
New scene observed: {scene_description}
Continue the story in 3-4 sentences based on this new scene. Be creative and dramatic."""

    response = ollama.chat(
        model="llava:7b-v1.5-q4_0",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']

def generate_art(story_beat, beat_number):
    style = genre_styles.get(genre, genre_styles["fantasy"])
    art_prompt = f"{story_beat[:200]}, {style}"

    print(f"🎨 Generating art for beat {beat_number}...")
    image = pipe(
        art_prompt,
        num_inference_steps=20,
        guidance_scale=7.5
    ).images[0]

    filename = f"beat_{beat_number}.png"
    image.save(filename)
    print(f"✅ Art saved: {filename}")
    return filename

def run_pipeline():
    print(f"\n🎬 LifeFrame starting... Genre: {genre}")
    print("Press Ctrl+C to stop\n")
    beat_number = 1

    while True:
        try:
            print("📸 Capturing frame...")
            frame = capture_frame()
            if frame is None:
                continue

            print("👁️  Analyzing scene...")
            frame_b64 = frame_to_base64(frame)
            scene = describe_scene(frame_b64)
            print(f"Scene: {scene}\n")

            print("✍️  Generating story beat...")
            beat = generate_story_beat(scene)
            story_so_far.append(beat)
            print(f"Story: {beat}\n")

            art_file = generate_art(beat, beat_number)
            beat_number += 1

            print("-" * 60)
            time.sleep(10)

        except KeyboardInterrupt:
            print("\n\n📖 Your story so far:")
            for i, beat in enumerate(story_so_far):
                print(f"\nBeat {i+1}: {beat}")
            break
        except Exception as e:
            print(f"Error: {e}, retrying...")
            time.sleep(3)

if __name__ == "__main__":
    run_pipeline()