

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

# --- ALSA ERROR SUPPRESSION ---
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

MIC_INDEX = None # System Default

# Global Variables
video_frame = None
known_face_encodings = []
known_face_names = []
current_detected_name = None  
current_status = "SYSTEM READY"

# --- STEP 1: LOAD DATABASE ---
def load_student_data():
    print(" [DB] Loading Student Faces...")
    global known_face_encodings, known_face_names
    
    path = "student_pics"
    if not os.path.exists(path):
        os.makedirs(path)
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
    
    print(f" [DB] System Ready.")

def get_student_info(name):
    """Returns the full dictionary for a student"""
    try:
        with open('students.json', 'r') as f:
            db = json.load(f)
        return db.get(name) # Returns {registered: true, id: ...} or None
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
    
    # Cache audio to make response faster
    filename = hashlib.md5(text.encode()).hexdigest() + ".mp3"
    file_path = f"audio_cache/{filename}"
    
    if not os.path.exists("audio_cache"):
        os.makedirs("audio_cache")

    if not os.path.exists(file_path):
        try:
            tts = gTTS(text=text, lang='en', tld='co.uk')
            tts.save(file_path)
        except Exception as e:
            print(f" [ERROR] TTS Failed: {e}")
            update_status(prev_status)
            return

    os.system(f"mpg321 -q {file_path}")
    update_status(prev_status)

def listen(prompt_text=None):
    """Generic listener"""
    if prompt_text:
        speak(prompt_text)

    update_status("LISTENING")
    r = sr.Recognizer()
    
    with no_alsa_errors():
        try:
            mic = sr.Microphone(device_index=MIC_INDEX)
            with mic as source:
                r.adjust_for_ambient_noise(source, duration=0.5)
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
    """Removes spaces and dashes to compare IDs (e.g., 'stu 001' -> 'stu001')"""
    if not text: return ""
    return re.sub(r'[\s-]', '', text).lower()

# --- LOGIC LOOP (STRICT SECURITY MODE) ---
def exam_logic_loop():
    global current_detected_name
    
    time.sleep(3) 
    speak("Examination Proctor System Online.")
    
    while True:
        update_status("WAITING FOR STUDENT")
        
        # 1. Wait for a face
        if current_detected_name is None:
            if int(time.time()) % 10 == 0: 
                speak("Please step forward for identification.")
            time.sleep(1)
            continue
            
        # 2. Face Detected - STRICT CHECK
        name = current_detected_name
        update_status(f"IDENTIFYING: {name}")
        
        # --- SECURITY FIX: REJECT UNKNOWN FACES ---
        if name == "Unknown":
            speak("Face not recognized.")
            time.sleep(0.5)
            speak("Access Denied. Your face does not match our records.")
            speak("Please step aside and see a human administrator.")
            
            # Lock the system until this person leaves the camera view
            while current_detected_name is not None:
                time.sleep(1)
            continue # Restart loop for the next person
            
        # 3. Face is Known - Proceed to Verification
        speak(f"Biometric match found. Hello {name}.")
        
        # Retrieve Data
        student_data = get_student_info(name)
        
        if not student_data:
            # Face matches image file, but name isn't in JSON
            speak(f"Error. Student {name} has no registration data file.")
            while current_detected_name is not None: time.sleep(1)
            continue

        # 4. Two-Factor Authentication (Face + Voice ID)
        correct_id = student_data['id'].lower()
        spoken_id = listen("To confirm your identity, please state your Student I D.")
        
        # Check ID
        if spoken_id and clean_id(spoken_id) == clean_id(correct_id):
            speak("Identity confirmed.")
        else:
            speak(f"Authentication Failed. I heard {spoken_id}, but records expect {correct_id}.")
            speak("Access Denied.")
            while current_detected_name is not None: time.sleep(1)
            continue

        # 5. Ask for Test (Administrative Log)
        test_name = listen("What test are you writing today?")
        if test_name:
            speak(f"Logging entry for {test_name}.")
        
        # 6. Final Registration Check & Rules
        if student_data['registered']:
            speak("Registration Verified. Listen strictly to the rules.")
            time.sleep(0.3)
            speak("1. No electronic devices. Phones, watches, and glasses must be removed.")
            time.sleep(0.3)
            speak("2. Keep your eyes on your own paper.")
            time.sleep(0.3)
            speak("Violation will result in immediate failure.")
            time.sleep(0.5)
            speak("You may enter. Good luck.")
            
            # Wait for student to enter
            while current_detected_name is not None: time.sleep(1)
            speak("Next student.")
            
        else:
            speak(f"Alert. {name}, you are NOT registered for this exam.")
            speak("Please leave the area immediately.")
            while current_detected_name is not None: time.sleep(1)


# --- WEB & VISION SETUP ---
@app.route('/')
def index(): return render_template('index.html')

def generate_frames():
    global video_frame
    while True:
        if video_frame is not None:
            ret, buffer = cv2.imencode('.jpg', video_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.04)

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status_feed')
def status_feed():
    def generate():
        while True:
            yield f"data: {current_status}\n\n"
            time.sleep(0.5)
    return Response(generate(), mimetype="text/event-stream")

def vision_loop():
    global video_frame, current_detected_name
    print(" [VISION] Starting Camera...")
    
    pipeline = "libcamerasrc ! video/x-raw, width=640, height=480, framerate=30/1, format=YUY2 ! videoconvert ! video/x-raw, format=BGR ! appsink"
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened(): return

    frame_count = 0
    cached_names = []
    cached_locs = []

    while True:
        ret, frame = cap.read()
        if not ret: time.sleep(0.1); continue
        
        frame_count += 1
        if frame_count % 5 == 0: # Process every 5th frame
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
                    if matches[np.argmin(dists)]: name = known_face_names[np.argmin(dists)]
                cached_names.append(name)
                detected = name
            current_detected_name = detected

        for (t, r, b, l), name in zip(cached_locs, cached_names):
            t*=4; r*=4; b*=4; l*=4
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            cv2.putText(frame, name, (l, b+20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 1)
        video_frame = frame

if __name__ == "__main__":
    load_student_data()
    threading.Thread(target=vision_loop, daemon=True).start()
    threading.Thread(target=exam_logic_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False)
