
import cv2
import ollama
import base64
import time
from PIL import Image
import io

# Story state
story_so_far = []
genre = "fantasy"  # we'll make this dynamic later

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
    history = " ".join(story_so_far[-3:])  # last 3 beats for continuity
    
    prompt = f"""You are a {genre} story writer. 
Previous story: {history}
New scene observed: {scene_description}
Continue the story in 3-4 sentences based on this new scene. Be creative and dramatic."""

    response = ollama.chat(
        model="llava:7b-v1.5-q4_0",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']

def run_pipeline():
    print(f"\n🎬 LifeFrame starting... Genre: {genre}")
    print("Press Ctrl+C to stop\n")
    
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