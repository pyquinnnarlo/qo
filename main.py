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
import csv
from datetime import datetime
from flask import Flask, render_template, Response
from gtts import gTTS
from ctypes import *
from contextlib import contextmanager
from gpiozero import LED 

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

# -- HARDWARE CONFIG (GPIO) --
try:
    red_led = LED(17)
    green_led = LED(27)
except Exception as e:
    print(f" [WARN] GPIO Init Failed (Are you on a Pi?): {e}")
    class MockLED:
        def on(self): pass
        def off(self): pass
        def blink(self, *args, **kwargs): pass
    red_led = MockLED()
    green_led = MockLED()

# Global Variables
video_frame = None
known_face_encodings = []
known_face_names = []
current_detected_name = None  
current_status = "SYSTEM READY"

# --- HELPER: LED CONTROLS ---
def led_busy():
    green_led.off()
    red_led.on()

def led_granted():
    red_led.off()
    green_led.on()

def led_denied():
    green_led.off()
    red_led.blink(on_time=0.1, off_time=0.1)

# --- NEW: LOGGING WITH PHOTOS ---
def log_attendance(name, student_id, test_name, status, frame=None):
    """Saves entry attempts to CSV and saves a photo of the person"""
    
    # 1. Define Log Filename
    log_filename = f"attendance_log_{datetime.now().strftime('%Y-%m-%d')}.csv"
    
    # 2. Save the Photo (if frame is provided)
    photo_path = "N/A"
    if frame is not None:
        # Create directory if missing
        if not os.path.exists("attendance_photos"):
            os.makedirs("attendance_photos")
        
        # Clean the name for filename safety
        safe_name = name.replace(" ", "_")
        timestamp_str = datetime.now().strftime("%H-%M-%S")
        
        # Save Image: attendance_photos/Unknown_12-30-01.jpg
        photo_filename = f"{status}_{safe_name}_{timestamp_str}.jpg"
        photo_path = os.path.join("attendance_photos", photo_filename)
        
        try:
            cv2.imwrite(photo_path, frame)
        except Exception as e:
            print(f" [ERROR] Could not save photo: {e}")

    # 3. Append to CSV
    file_exists = os.path.exists(log_filename)
    
    with open(log_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write Header if new file
        if not file_exists:
            writer.writerow(["Timestamp", "Name", "Student ID", "Test Name", "Status", "Photo Path"])
            
        timestamp = datetime.now().strftime("%H:%M:%S")
        writer.writerow([timestamp, name, student_id, test_name, status, photo_path])
        print(f" [ADMIN] Logged: {status} for {name} (Photo Saved)")

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
    
    filename = hashlib.md5(text.encode()).hexdigest() + ".mp3"
    file_path = f"audio_cache/{filename}"
    
    if not os.path.exists("audio_cache"): os.makedirs("audio_cache")

    if not os.path.exists(file_path):
        try:
            padded_text = ". . . " + text
            tts = gTTS(text=padded_text, lang='en', tld='co.uk')
            tts.save(file_path)
        except Exception as e:
            print(f" [ERROR] TTS Failed: {e}")
            update_status(prev_status)
            return

    time.sleep(0.1)
    os.system(f"mpg321 -q {file_path}")
    update_status(prev_status)

def listen(prompt_text=None):
    if prompt_text: speak(prompt_text)
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
    if not text: return ""
    return re.sub(r'[\s-]', '', text).lower()

# --- LOGIC LOOP (ADMINISTRATOR) ---
def exam_logic_loop():
    global current_detected_name, video_frame
    
    led_busy()
    time.sleep(3) 
    speak("Examination Proctor System Online.")
    
    while True:
        update_status("WAITING FOR STUDENT")
        led_busy()
        
        # 1. Wait for a face
        if current_detected_name is None:
            if int(time.time()) % 15 == 0:
                speak("Please step forward for identification.")
            time.sleep(1)
            continue
            
        # 2. Face Detected
        name = current_detected_name
        update_status(f"IDENTIFYING: {name}")
        
        # --- SECURITY FIX: REJECT UNKNOWN FACES ---
        if name == "Unknown":
            led_denied()
            speak("Face not recognized.")
            time.sleep(0.5)
            speak("Access Denied.")
            
            # LOGGING WITH PHOTO
            log_attendance("Unknown", "N/A", "N/A", "DENIED_FACE_MISMATCH", video_frame)
            
            speak("Please step aside.")
            while current_detected_name is not None: time.sleep(1)
            continue 
            
        # 3. Face is Known
        speak(f"Biometric match found. Hello {name}.")
        
        student_data = get_student_info(name)
        if not student_data:
            led_denied()
            speak(f"Error. Student {name} has no registration data.")
            # LOGGING WITH PHOTO
            log_attendance(name, "Unknown", "N/A", "DENIED_NO_DATA", video_frame)
            while current_detected_name is not None: time.sleep(1)
            continue

        # 4. Voice ID Check
        correct_id = student_data['id'].lower()
        spoken_id = listen("To confirm your identity, please state your Student I D.")
        
        if spoken_id and clean_id(spoken_id) == clean_id(correct_id):
            speak("Identity confirmed.")
        else:
            led_denied()
            speak(f"Authentication Failed.")
            speak("Access Denied.")
            
            # LOGGING WITH PHOTO
            log_attendance(name, student_data['id'], "N/A", "DENIED_WRONG_VOICE_ID", video_frame)
            
            while current_detected_name is not None: time.sleep(1)
            continue

        # 5. Ask for Test
        test_name = listen("What test are you writing today?")
        if not test_name: test_name = "Not Stated"
        speak(f"Logging entry for {test_name}.")
        
        # 6. Final Registration Check
        if student_data['registered']:
            speak("Registration Verified. Listen strictly to the rules.")
            time.sleep(0.3)
            speak("1. No electronic devices.")
            time.sleep(0.3)
            speak("2. Keep your eyes on your own paper.")
            time.sleep(0.5)
            speak("You may enter. Good luck.")
            
            led_granted()
            # LOGGING WITH PHOTO (Success)
            log_attendance(name, student_data['id'], test_name, "ALLOWED_ENTRY", video_frame)
            
            while current_detected_name is not None: time.sleep(1)
            speak("Next student.")
            
        else:
            led_denied()
            speak(f"Alert. {name}, you are NOT registered for this exam.")
            
            # LOGGING WITH PHOTO (Denied)
            log_attendance(name, student_data['id'], test_name, "DENIED_NOT_REGISTERED", video_frame)
            
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
