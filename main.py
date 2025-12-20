import threading
import cv2
import speech_recognition as sr
import os
import time
import requests
from flask import Flask, render_template, Response, stream_with_context
from ultralytics import YOLO

# --- CONFIGURATION ---
app = Flask(__name__)
MODEL_PATH = "yolov8n.pt"
OLLAMA_MODEL = "phi3:mini"
WAKE_WORD = "computer"

# Global Variables
current_scene_objects = []
video_frame = None
CURRENT_STATE = "IDLE"  # Options: IDLE, LISTENING, THINKING, SPEAKING

# --- FLASK ROUTES (THE WEB SERVER) ---
@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    """Streams the camera feed to the browser."""
    global video_frame
    while True:
        if video_frame is not None:
            ret, buffer = cv2.imencode('.jpg', video_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.05)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status_feed')
def status_feed():
    """Server-Sent Events to update the robot face."""
    def event_stream():
        last_state = ""
        while True:
            global CURRENT_STATE
            # Only send update if state changed to save bandwidth
            if CURRENT_STATE != last_state:
                yield f"data: {CURRENT_STATE}\n\n"
                last_state = CURRENT_STATE
            time.sleep(0.1)
    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")

# --- AI LOGIC ---
def vision_loop():
    global current_scene_objects, video_frame
    model = YOLO(MODEL_PATH)
    # GStreamer pipeline for Raspberry Pi 5 Libcamera
    gst_str = (
        "libcamerasrc ! "
        "video/x-raw, width=640, height=480, framerate=30/1 ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink"
    )
    cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


    while True:
        ret, frame = cap.read()
        if not ret: continue

        # Run object detection
        results = model(frame, verbose=False, stream=True)
        detected = []
        for r in results:
            # Draw boxes on the frame for the web view
            frame = r.plot() 
            for box in r.boxes:
                class_id = int(box.cls[0])
                detected.append(model.names[class_id])
        
        video_frame = frame  # Update global frame for web streaming
        current_scene_objects = list(set(detected))

def speak(text):
    global CURRENT_STATE
    CURRENT_STATE = "SPEAKING"
    # Escaping quotes to prevent shell errors
    clean_text = text.replace('"', '').replace("'", "")
    # Using piper via command line or simple espeak for demo
    # Ensure 'espeak' is installed: sudo apt install espeak
    os.system(f'espeak "{clean_text}"') 
    CURRENT_STATE = "IDLE"

def listen_and_respond():
    global CURRENT_STATE
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("System Ready. Open browser at http://localhost:5000")
    
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        
        while True:
            try:
                # 1. Listen for Wake Word
                CURRENT_STATE = "IDLE"
                # Short timeout to keep loop responsive
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)
                
                # Check for wake word (simple logic)
                try:
                    text = recognizer.recognize_google(audio).lower()
                    print(f"Heard: {text}")
                except sr.UnknownValueError:
                    continue

                if WAKE_WORD in text:
                    CURRENT_STATE = "LISTENING"
                    # Play a small beep sound here if desired
                    print("Wake word detected! Listening for command...")
                    
                    # Listen again for the actual command
                    audio_cmd = recognizer.listen(source, timeout=5)
                    command = recognizer.recognize_google(audio_cmd).lower()
                    print(f"Command: {command}")
                    
                    # 2. Think
                    CURRENT_STATE = "THINKING"
                    scene_desc = ", ".join(current_scene_objects) if current_scene_objects else "nothing specific"
                    
                    prompt = (
                        f"You are a helpful robot. You see: {scene_desc}. "
                        f"User said: {command}. Answer briefly."
                    )
                    
                    response = requests.post('http://localhost:11434/api/generate', 
                        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False})
                    
                    reply = response.json().get('response', "I am not sure.")
                    print(f"Reply: {reply}")
                    
                    # 3. Speak
                    speak(reply)

            except Exception as e:
                # print(f"Loop error: {e}")
                CURRENT_STATE = "IDLE"

if __name__ == "__main__":
    # Start Vision Thread
    t_vision = threading.Thread(target=vision_loop, daemon=True)
    t_vision.start()

    # Start AI Logic Thread
    t_ai = threading.Thread(target=listen_and_respond, daemon=True)
    t_ai.start()

    # Run Web Server (Blocks main thread)
    # host='0.0.0.0' makes it accessible from other devices on the network
    app.run(host='0.0.0.0', port=5000, debug=False)
