import threading
import cv2
import speech_recognition as sr
import os
import time
import json
import face_recognition
import numpy as np
import logging
import hashlib
import re
from flask import Flask, render_template, Response
from gtts import gTTS
from ctypes import *
from contextlib import contextmanager

# --- ALSA ERROR SUPPRESSION (Cleans up terminal output) ---
ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
def py_error_handler(filename, line, function, err, fmt):
    pass
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

@contextmanager
def no_alsa_errors():
    asound = cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)
    yield
    asound.snd_lib_error_set_handler(None)

# --- CONFIGURATION ---
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

MIC_INDEX = None # System Default Input

# Global Variables
video_frame = None
known_face_encodings = []
known_face_names = []
current_detected_name = None  
current_status = "SYSTEM ONLINE"

# --- STEP 1: LOAD DATABASE ---
def load_student_data():
    print(" [DB] Loading Student Faces...")
    global known_face_encodings, known_face_names
    
    path = "student_pics"
    if not os.path.exists(path):
        os.makedirs(path)
        print(" [WARNING] 'student_pics' folder missing. Please create it.")
        return

    for file in os.listdir(path):
        if file.endswith((".jpg", ".png", ".jpeg")):
            img = face_recognition.load_image_file(f"{path}/{file}")
            encoding = face_recognition.face_encodings(img)
            
            if len(encoding) > 0:
                known_face_encodings.append(encoding[0])
                name = os.path.splitext(file)[0]
                known_face_names.append(name)
                print(f"   + Loaded: {name}")
    
    print(f" [DB] Database Ready. Loaded {len(known_face_names)} students.")

def get_student_info(name):
    """Returns the full dictionary for a student from JSON"""
    try:
        with open('students.json', 'r') as f:
            db = json.load(f)
        return db.get(name) 
    except:
        return None

# --- AUDIO FUNCTIONS ---
def update_status(status):
    global current_status
    current_status = status

def speak(text):
    global current_status
    print(f"Robot: {text}")
    
    prev_status = current_status
    update_status(f"SPEAKING: {text}")
    
    # Hash text for caching filename
    filename = hashlib.md5(text.encode()).hexdigest() + ".mp3"
    file_path = f"audio_cache/{filename}"
    
    if not os.path.exists("audio_cache"):
        os.makedirs("audio_cache")

    if not os.path.exists(file_path):
        try:
            # FIX: Add padding ". . . " so the Pi audio driver doesn't cut off the start
            padded_text = ". . . " + text
            tts = gTTS(text=padded_text, lang='en', tld='co.uk')
            tts.save(file_path)
        except Exception as e:
            print(f" [ERROR] TTS Failed: {e}")
            update_status(prev_status)
            return

    # Short delay to ensure file write and process priority
    time.sleep(0.1)
    os.system(f"mpg321 -q {file_path}")
    
    update_status(prev_status)

def listen(prompt_text=None):
    """Generic listener with timeout handling"""
    if prompt_text:
        speak(prompt_text)

    update_status("LISTENING")
    r = sr.Recognizer()
    
    with no_alsa_errors():
        try:
            mic = sr.Microphone(device_index=MIC_INDEX)
            with mic as source:
                r.adjust_for_ambient_noise(source, duration=0.5)
                # Listen for up to 5 seconds
                audio = r.listen(source, timeout=5, phrase_time_limit=5)
                
                update_status("THINKING")
                text = r.recognize_google(audio).lower()
                print(f" [User said]: {text}")
                return text
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            speak("I did not understand.")
            return None
        except Exception:
            speak("Audio error.")
            return None
        finally:
            update_status("IDLE")

def clean_id(text):
    """Normalizes ID strings for comparison"""
    if not text: return ""
    return re.sub(r'[\s-]', '', text).lower()

# --- LOGIC LOOP (THE ADMINISTRATOR) ---
def exam_logic_loop():
    global current_detected_name
    
    time.sleep(5) # Give camera time to warm up
    speak("Examination Proctor System Online.")
    
    while True:
        update_status("WAITING FOR STUDENT")
        
        # 1. Wait for a face
        if current_detected_name is None:
            # Remind occasionally
            if int(time.time()) % 20 == 0: 
                speak("Please step forward for identification.")
            time.sleep(1)
            continue
            
        # 2. Face Detected
        name = current_detected_name
        update_status(f"IDENTIFYING: {name}")
        
        # --- SECURITY CHECK: REJECT UNKNOWN ---
        if name == "Unknown":
            speak("Face not recognized.")
            time.sleep(0.5)
            update_status("ACCESS DENIED")
            speak("Access Denied. Please see a human administrator.")
            
            # Wait for them to leave
            while current_detected_name is not None:
                time.sleep(1)
            continue 
            
        # 3. Known Face Found
        speak(f"Biometric match found. Hello {name}.")
        
        # Retrieve Data
        student_data = get_student_info(name)
        
        if not student_data:
            speak(f"Error. No data found for {name}.")
            while current_detected_name is not None: time.sleep(1)
            continue

        # 4. Two-Factor Auth (Voice ID)
        correct_id = str(student_data['id']).lower()
        spoken_id = listen("Please state your Student I D.")
        
        if spoken_id and clean_id(spoken_id) in clean_id(correct_id):
            speak("Identity confirmed.")
        else:
            update_status("AUTH FAILED")
            speak(f"Authentication Failed. Records expect I D {correct_id}.")
            speak("Please step aside.")
            while current_detected_name is not None: time.sleep(1)
            continue

        # 5. Log Test Entry
        test_name = listen("What test are you writing?")
        if test_name:
            speak(f"Logging entry for {test_name}.")
        
        # 6. Check Registration & Give Rules
        if student_data['registered']:
            update_status("ACCESS GRANTED")
            speak("Registration Verified. Listen strictly to the rules.")
            time.sleep(0.3)
            speak("1. No electronic devices allowed.")
            time.sleep(0.3)
            speak("2. Keep your eyes on your own paper.")
            time.sleep(0.3)
            speak("Violation results in immediate failure.")
            time.sleep(0.5)
            speak("You may enter. Good luck.")
            
            while current_detected_name is not None: time.sleep(1)
            speak("Next student.")
            
        else:
            update_status("ACCESS DENIED")
            speak(f"Alert. {name}, you are NOT registered for this exam.")
            speak("Please leave the area immediately.")
            while current_detected_name is not None: time.sleep(1)

# --- FLASK & CAMERA SETUP ---
@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    global video_frame
    while True:
        if video_frame is not None:
            ret, buffer = cv2.imencode('.jpg', video_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.04)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status_feed')
def status_feed():
    def generate():
        while True:
            yield f"data: {current_status}\n\n"
            time.sleep(0.5)
    return Response(generate(), mimetype="text/event-stream")

def vision_loop():
    global video_frame, current_detected_name
    print(" [VISION] Starting Camera via GStreamer...")
    
    # Pi 5 Pipeline
    pipeline = (
        "libcamerasrc ! "
        "video/x-raw, width=640, height=480, framerate=30/1, format=YUY2 ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! appsink"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print(" [ERROR] Camera Failed. Check ribbon cable.")
        return

    frame_count = 0
    cached_names = []
    cached_locs = []

    while True:
        ret, frame = cap.read()
        if not ret: time.sleep(0.1); continue
        
        frame_count += 1
        # Process every 5th frame to prevent CPU overload/crash
        if frame_count % 5 == 0: 
            small = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            cached_locs = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(rgb, cached_locs)
            
            cached_names = []
            detected = None
            for enc in encodings:
                matches = face_recognition.compare_faces(known_face_encodings, enc)
                name = "Unknown"
                dists = face_recognition.face_distance(known_face_encodings, enc)
                if len(dists) > 0:
                    best_idx = np.argmin(dists)
                    if matches[best_idx]: name = known_face_names[best_idx]
                cached_names.append(name)
                detected = name
            current_detected_name = detected

        # Draw HUD elements on frame
        for (t, r, b, l), name in zip(cached_locs, cached_names):
            t*=4; r*=4; b*=4; l*=4
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            cv2.putText(frame, name, (l, b+20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 1)
        
        video_frame = frame

if __name__ == "__main__":
    load_student_data()
    # Clean cache on startup
    if os.path.exists("audio_cache"):
        import shutil
        shutil.rmtree("audio_cache")
        
    threading.Thread(target=vision_loop, daemon=True).start()
    threading.Thread(target=exam_logic_loop, daemon=True).start()
    
    app.run(host='0.0.0.0', port=5000, debug=False)
