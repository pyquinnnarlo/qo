import threading
import cv2
import speech_recognition as sr
import os
import time
import json
import face_recognition
import numpy as np
import logging
from flask import Flask, render_template, Response

# --- CONFIGURATION ---
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# *** UPDATE THIS AFTER RUNNING check_mic.py ***
MIC_INDEX = 1  # Change this to the correct index found!

# Global Variables
video_frame = None
known_face_encodings = []
known_face_names = []
current_detected_name = None  
current_status = "SYSTEM READY"

# --- STEP 1: LOAD DATABASE & FACES ---
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

# --- AUDIO FUNCTIONS ---
def update_status(status):
    global current_status
    current_status = status

def speak(text):
    global current_status
    print(f"Robot: {text}")
    prev_status = current_status
    update_status(f"SPEAKING: {text}")
    
    safe_text = text.replace("'", "").replace('"', "")
    os.system(f'espeak -ven+m3 -s160 "{safe_text}" 2>/dev/null')
    
    update_status(prev_status)

def listen_for_name():
    """Asks for name with error handling"""
    update_status("LISTENING")
    speak("Face not recognized. Please state your name clearly.")
    
    r = sr.Recognizer()
    try:
        # Use device_index=None if MIC_INDEX fails, or set specific index
        mic = sr.Microphone(device_index=MIC_INDEX)
        
        with mic as source:
            r.adjust_for_ambient_noise(source, duration=0.5)
            # Listen with a timeout so we don't hang forever
            audio = r.listen(source, timeout=5, phrase_time_limit=5)
            
            update_status("THINKING")
            name = r.recognize_google(audio).lower()
            return name
    except sr.WaitTimeoutError:
        print(" [AUDIO] No speech detected.")
        return None
    except sr.RequestError:
        print(" [AUDIO] Network error (Google API).")
        speak("I cannot connect to the internet.")
        return None
    except Exception as e:
        print(f" [AUDIO] Microphone Error: {e}")
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
            # Wait 5 seconds before speaking again
            for _ in range(5): 
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
            speak(f"Hello {name}. You may enter.")
            # Wait for them to leave frame
            while current_detected_name == name: time.sleep(1)
            speak("Next.")
            
        elif is_registered is False:
            speak(f"{name}, you are NOT registered.")
            while current_detected_name == name: time.sleep(1)
                
        elif is_registered is None:
            if name != "Unknown":
                speak(f"{name} not in database.")
                time.sleep(2)
        
        time.sleep(1) 

# --- VISION LOOP (OPTIMIZED) ---
def vision_loop():
    global video_frame, current_detected_name
    
    print(" [VISION] Starting Camera...")
    
    # Pi 5 / Libcamera Pipeline
    pipeline = (
        "libcamerasrc ! "
        "video/x-raw, width=640, height=480, framerate=30/1, format=YUY2 ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! appsink"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print(" [ERROR] Camera Failed. Check connections.")
        return

    frame_count = 0
    
    # Cache the results to display them on skipped frames
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
            
            # Detect
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

        # --- DRAWING (Uses cached data on skipped frames) ---
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
