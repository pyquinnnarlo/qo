import threading
import cv2
import speech_recognition as sr
import os
import time
import logging
from flask import Flask, render_template, Response, stream_with_context
from ultralytics import YOLO
from picamera2 import Picamera2
from openai import OpenAI  # <--- NEW IMPORT

# --- CONFIGURATION ---
app = Flask(__name__)
# Suppress Flask logs
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# !!! ENTER YOUR OPENAI API KEY HERE !!!
OPENAI_API_KEY = "sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx" 
OPENAI_MODEL = "gpt-4o-mini" # Fast, cheap, and very smart

# Setup OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY)

MIC_INDEX = 1                  # Your USB Mic Index
WAKE_WORD = "computer"         

# Global Shared Variables
current_scene_objects = []     
video_frame = None             
CURRENT_STATE = "IDLE"         

# --- PART 1: FLASK WEB SERVER ---
@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    """Stream video to browser"""
    global video_frame
    while True:
        if video_frame is not None:
            ret, buffer = cv2.imencode('.jpg', video_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.04) 

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
    
    model = YOLO("yolov8n.pt")
    
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    ))
    picam2.start()

    while True:
        try:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            results = model(frame, verbose=False, stream=True)
            
            detected = []
            for r in results:
                frame = r.plot() 
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    detected.append(model.names[cls_id])
            
            current_scene_objects = list(set(detected))
            video_frame = frame
            
        except Exception as e:
            print(f"Vision Error: {e}")
            time.sleep(1)


# --- PART 3: VOICE & BRAIN (OpenAI) ---
def speak(text):
    global CURRENT_STATE
    if not text.strip(): return
    
    CURRENT_STATE = "SPEAKING"
    print(f"Robot: {text}")
    
    safe_text = text.replace("'", "").replace('"', "")
    os.system(f'espeak -ven+m3 -s160 "{safe_text}" 2>/dev/null')
    
    # Wait for audio to release (Critical for Pi audio drivers)
    time.sleep(0.5)
    CURRENT_STATE = "IDLE"

def stream_response(prompt):
    global CURRENT_STATE
    CURRENT_STATE = "THINKING"
    
    # Context
    vision_context = ", ".join(current_scene_objects)
    if not vision_context: vision_context = "nothing clearly"
    
    # System Instruction
    system_prompt = (
        f"You are a helpful robot assistant. "
        f"Through your camera, you see: {vision_context}. "
        f"Keep your answers conversational and brief (1-2 sentences)."
    )
    
    print(" [BRAIN] Sending to OpenAI...", end="", flush=True)
    
    try:
        # OpenAI Streaming Call
        stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            stream=True,
        )
        
        buffer = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                word = chunk.choices[0].delta.content
                buffer += word
                
                # Speak immediately on punctuation
                if word in ['.', '?', '!', '\n']:
                    speak(buffer)
                    buffer = ""
                    CURRENT_STATE = "SPEAKING"
        
        # Flush remaining buffer
        if buffer.strip(): speak(buffer)
        
    except Exception as e:
        print(f"\nOpenAI Error: {e}")
        speak("I am having trouble connecting to the internet.")

    CURRENT_STATE = "IDLE"

def voice_loop():
    global CURRENT_STATE
    recognizer = sr.Recognizer()
    mic = sr.Microphone(device_index=MIC_INDEX)
    
    print(" [EARS] Calibrating Microphone...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        recognizer.dynamic_energy_threshold = True
    
    print(" [SYSTEM] Robot Online via OpenAI.")

    while True:
        try:
            with mic as source:
                try:
                    audio = recognizer.listen(source, timeout=None, phrase_time_limit=5)
                except sr.WaitTimeoutError:
                    continue

                try:
                    text = recognizer.recognize_google(audio).lower()
                    print(f"You: {text}")
                    
                    # Wake word check
                    if WAKE_WORD in text or True: # 'or True' makes it reply to everything
                        CURRENT_STATE = "LISTENING"
                        stream_response(text)
                        
                except sr.UnknownValueError:
                    pass 
                except sr.RequestError:
                    print("Speech API Error")

        except Exception as e:
            print(f"Voice Loop Error: {e}")
            CURRENT_STATE = "IDLE"

# --- MAIN ---
if __name__ == "__main__":
    t_vision = threading.Thread(target=vision_loop, daemon=True)
    t_vision.start()

    t_voice = threading.Thread(target=voice_loop, daemon=True)
    t_voice.start()

    app.run(host='0.0.0.0', port=5000, debug=False)
