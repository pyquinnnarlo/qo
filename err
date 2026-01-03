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
from flask import Flask, render_template, Response
from gtts import gTTS  # NEW: Google Text-to-Speech
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

# MIC_INDEX = None uses the System Default (configured in Pi Desktop)
MIC_INDEX = None 

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
    
    print(f" [DB] System Ready. Known students: {len(known_face_names)}")

def check_registration(name):
    try:
        with open('students.json', 'r') as f:
            db = json.load(f)
        student = db.get(name)
        if student:
            return student.get('registered', False)
        return None 
    except:
        return None

# --- AUDIO FUNCTIONS (UPDATED FOR REAL VOICE) ---
def update_status(status):
    global current_status
    current_status = status

def speak(text):
    global current_status
    print(f"Robot: {text}")
    
    # Update UI to make mouth move
    prev_status = current_status
    update_status(f"SPEAKING: {text}")
    
    # 1. Create a unique filename based on the text (Caching)
    # This prevents downloading "Please look at the camera" 100 times
    filename = hashlib.md5(text.encode()).hexdigest() + ".mp3"
    file_path = f"audio_cache/{filename}"
    
    if not os.path.exists("audio_cache"):
        os.makedirs("audio_cache")

    # 2. Generate Audio if it doesn't exist
    if not os.path.exists(file_path):
        try:
            # lang='en', tld='co.uk' gives a nice British English voice
            tts = gTTS(text=text, lang='en', tld='co.uk')
            tts.save(file_path)
        except Exception as e:
            print(f" [ERROR] Could not generate TTS: {e}")
            update_status(prev_status)
            return

    # 3. Play the MP3 file using mpg321 (Quiet mode)
    os.system(f"mpg321 -q {file_path}")
    
    update_status(prev_status)

def listen_for_name():
    """Asks for name with error handling"""
    update_status("LISTENING")
    speak("Face not recognized. Please state your name.")
    
    r = sr.Recognizer()
    
    with no_alsa_errors():
        try:
            mic = sr.Microphone(device_index=MIC_INDEX)
            with mic as source:
                r.adjust_for_ambient_noise(source, duration=0.5)
                # Reduced timeout to prevent hanging
                audio = r.listen(source, timeout=4, phrase_time_limit=4)
                
                update_status("THINKING")
                name = r.recognize_google(audio).lower()
                return name
        except sr.WaitTimeoutError:
            print(" [AUDIO] Timeout.")
            return None
        except sr.UnknownValueError:
            speak("I did not understand.")
            return None
        except Exception as e:
            print(f" [AUDIO] Error: {e}")
            return None
        finally:
            update_status("IDLE")

# --- WEB SERVER ---
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

# --- LOGIC LOOP ---
def exam_logic_loop():
    global current_detected_name
    
    time.sleep(3) 
    speak("System Online.")
    
    while True:
        update_status("WAITING FOR FACE")
        
        # 1. Wait for Face
        if current_detected_name is None:
            speak("Please look at the camera.")
            # Wait loop to avoid repeating too fast
            for _ in range(6): 
                if current_detected_name is not None: break
                time.sleep(1)
            continue
            
        # 2. Process Face
        name = current_detected_name
        update_status(f"PROCESSING: {name}")
        
        if name == "Unknown":
            spoken_name = listen_for_name()
            if spoken_name:
                speak(f"I heard {spoken_name}.")
                name = spoken_name.replace(" ", "_")
        
        # 3. Validation
        is_registered = check_registration(name)
        
        if is_registered is True:
            speak(f"Hello {name}.")
            time.sleep(0.2)
            speak("You are registered. You may enter.")
            
            # Wait for them to leave
            while current_detected_name == name: time.sleep(1)
            speak("Next student.")
            
        elif is_registered is False:
            speak(f"Alert. {name}, you are NOT registered.")
            while current_detected_name == name: time.sleep(1)
                
        elif is_registered is None:
            if name != "Unknown":
                speak(f"I cannot find registration for {name}.")
                time.sleep(2)
        
        time.sleep(1) 

# --- VISION LOOP ---
def vision_loop():
    global video_frame, current_detected_name
    
    print(" [VISION] Starting Camera...")
    
    pipeline = (
        "libcamerasrc ! "
        "video/x-raw, width=640, height=480, framerate=30/1, format=YUY2 ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! appsink"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print(" [ERROR] Camera Failed.")
        return

    frame_count = 0
    cached_face_locations = []
    cached_face_names = []

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        
        frame_count += 1
        
        # --- OPTIMIZATION: Only detect faces every 5th frame ---
        if frame_count % 5 == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            cached_face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, cached_face_locations)
            
            cached_face_names = []
            detected_name = None

            if face_encodings:
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]
                    
                    cached_face_names.append(name)
                    detected_name = name
            
            current_detected_name = detected_name

        for (top, right, bottom, left), name in zip(cached_face_locations, cached_face_names):
            top *= 4; right *= 4; bottom *= 4; left *= 4
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, str(name), (left, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        video_frame = frame

if __name__ == "__main__":
    load_student_data()
    
    t_vis = threading.Thread(target=vision_loop, daemon=True)
    t_vis.start()
    
    t_log = threading.Thread(target=exam_logic_loop, daemon=True)
    t_log.start()
    
    app.run(host='0.0.0.0', port=5000, debug=False)
