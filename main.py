import threading
import cv2
import speech_recognition as sr
import os
import time
import requests
import json
import logging
from flask import Flask, render_template, Response, stream_with_context
from ultralytics import YOLO
from picamera2 import Picamera2

# --- CONFIGURATION ---
app = Flask(__name__)
# Suppress Flask logs in terminal to keep it clean
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

OLLAMA_MODEL = "qwen2.5:0.5b"  # Fast model
MIC_INDEX = 1                  # Your USB Mic Index
WAKE_WORD = "computer"         # Or "hi", "robot"

# Global Shared Variables
current_scene_objects = []     # List of things the robot sees
video_frame = None             # The current video image
CURRENT_STATE = "IDLE"         # UI State: IDLE, LISTENING, THINKING, SPEAKING

# --- PART 1: FLASK WEB SERVER ---
@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    """Stream video to browser"""
    global video_frame
    while True:
        if video_frame is not None:
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', video_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.04) # 25 FPS

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status_feed')
def status_feed():
    """Push state updates to the UI"""
    def event_stream():
        last_state = ""
        while True:
            global CURRENT_STATE
            if CURRENT_STATE != last_state:
                yield f"data: {CURRENT_STATE}\n\n"
                last_state = CURRENT_STATE
            time.sleep(0.1)
    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")


# --- PART 2: VISION (EYES) ---
def vision_loop():
    global video_frame, current_scene_objects
    print(" [EYES] Starting Camera (Picamera2)...")
    
    # Initialize YOLO
    model = YOLO("yolov8n.pt")
    
    # Initialize Picamera2
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    ))
    picam2.start()

    while True:
        try:
            # Capture from Pi Camera
            frame = picam2.capture_array()
            
            # Since Picamera captures RGB, OpenCV expects BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # YOLO Inference
            results = model(frame, verbose=False, stream=True)
            
            detected = []
            for r in results:
                frame = r.plot() # Draw boxes
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    detected.append(model.names[cls_id])
            
            # Update Globals
            current_scene_objects = list(set(detected))
            video_frame = frame
            
        except Exception as e:
            print(f"Vision Error: {e}")
            time.sleep(1)


# --- PART 3: VOICE & BRAIN (EARS & MOUTH) ---
def speak(text):
    global CURRENT_STATE
    if not text.strip(): return
    
    CURRENT_STATE = "SPEAKING"
    print(f"Robot: {text}")
    
    safe_text = text.replace("'", "").replace('"', "")
    os.system(f'espeak -ven+m3 -s160 "{safe_text}" 2>/dev/null')
    
    # Wait for audio to release (Critical fix for "Stuck Listening")
    time.sleep(0.5)
    CURRENT_STATE = "IDLE"

def stream_response(prompt):
    global CURRENT_STATE
    CURRENT_STATE = "THINKING"
    
    # Build prompt with visual context
    vision_context = ", ".join(current_scene_objects)
    if not vision_context: vision_context = "nothing clearly"
    
    full_prompt = (
        f"You are a robot. You see: {vision_context}. "
        f"User said: {prompt}. Be brief."
    )
    
    url = "http://localhost:11434/api/generate"
    data = {"model": OLLAMA_MODEL, "prompt": full_prompt, "stream": True}
    
    print(" [BRAIN] Thinking...", end="", flush=True)
    
    try:
        response = requests.post(url, json=data, stream=True)
        buffer = ""
        
        for line in response.iter_lines():
            if line:
                json_chunk = json.loads(line.decode('utf-8'))
                if 'response' in json_chunk:
                    word = json_chunk['response']
                    buffer += word
                    # Speak continuously on punctuation
                    if word in ['.', '?', '!', '\n']:
                        speak(buffer)
                        buffer = ""
                        CURRENT_STATE = "SPEAKING" # Keep state updated
        
        if buffer.strip(): speak(buffer)
        
    except Exception as e:
        print(f"Ollama Error: {e}")

    CURRENT_STATE = "IDLE"

def voice_loop():
    global CURRENT_STATE
    recognizer = sr.Recognizer()
    mic = sr.Microphone(device_index=MIC_INDEX)
    
    print(" [EARS] Calibrating Microphone...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        recognizer.dynamic_energy_threshold = True
    
    print(" [SYSTEM] Robot Online.")

    while True:
        try:
            with mic as source:
                # Don't change state to "Listening" on the UI until we actually detect something
                # to avoid flashing green constantly
                
                # Listen (Blocking)
                try:
                    audio = recognizer.listen(source, timeout=None, phrase_time_limit=5)
                except sr.WaitTimeoutError:
                    continue

                # Recognize
                try:
                    text = recognizer.recognize_google(audio).lower()
                    
                    # Only activate if wake word is heard OR if we just want it to reply to everything
                    # For this demo, let's reply to everything to make testing easy
                    print(f"You: {text}")
                    CURRENT_STATE = "LISTENING"
                    
                    if WAKE_WORD in text or True: # Remove 'or True' to enforce wake word
                        stream_response(text)
                        
                except sr.UnknownValueError:
                    pass # Ignore noise
                except sr.RequestError:
                    print("Internet down?")

        except Exception as e:
            print(f"Voice Loop Error: {e}")
            CURRENT_STATE = "IDLE"

# --- MAIN ENTRY POINT ---
if __name__ == "__main__":
    # 1. Start Vision Thread
    t_vision = threading.Thread(target=vision_loop, daemon=True)
    t_vision.start()

    # 2. Start Voice Thread
    t_voice = threading.Thread(target=voice_loop, daemon=True)
    t_voice.start()

    # 3. Start Web Server (Blocks Main Thread)
    # Open http://localhost:5000 in your browser
    app.run(host='0.0.0.0', port=5000, debug=False)
