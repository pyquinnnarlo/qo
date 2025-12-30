the camera not turning on the Raspberry Pi 5:
import threading
import cv2
import speech_recognition as sr
import os
import time
import json
import face_recognition
import numpy as np
import logging
from flask import Flask, render_template, Response, stream_with_context

--- CONFIGURATION ---

app = Flask(name)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

MIC_INDEX = 1  # Your USB Mic Index

Global Variables

video_frame = None
known_face_encodings = []
known_face_names = []
current_detected_name = None
last_processed_name = None  # To prevent repeating the same check 5 times a second

--- STEP 1: LOAD DATABASE & FACES ---

def load_student_data():
print(" [DB] Loading Student Faces...")
global known_face_encodings, known_face_names

code
Code
download
content_copy
expand_less
path = "student_pics"
if not os.path.exists(path):
    os.makedirs(path)
    print(" ! Created folder 'student_pics'. Please put .jpg files there!")
    return

# Loop through every image in the folder
for file in os.listdir(path):
    if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
        # Load image
        img = face_recognition.load_image_file(f"{path}/{file}")
        # Encode face (Get the AI numbers describing the face)
        encoding = face_recognition.face_encodings(img)
        
        if len(encoding) > 0:
            known_face_encodings.append(encoding[0])
            # Use filename as name (remove .jpg)
            name = os.path.splitext(file)[0]
            known_face_names.append(name)
            print(f"   + Loaded: {name}")

print(f" [DB] System Ready. Known students: {len(known_face_names)}")

def check_registration(name):
"""Checks JSON to see if student is registered"""
try:
with open('students.json', 'r') as f:
db = json.load(f)

code
Code
download
content_copy
expand_less
student = db.get(name)
    if student:
        return student['registered']
    return None # Student not in database file
except:
    return None
--- AUDIO FUNCTIONS ---

def speak(text):
print(f"Robot: {text}")
safe_text = text.replace("'", "").replace('"', "")
os.system(f'espeak -ven+m3 -s160 "{safe_text}" 2>/dev/null')
time.sleep(0.5) # Wait for audio to release

def listen_for_name():
"""Asks for name if face is not recognized"""
speak("Face not recognized. Please state your name clearly.")
r = sr.Recognizer()
mic = sr.Microphone(device_index=MIC_INDEX)

code
Code
download
content_copy
expand_less
with mic as source:
    r.adjust_for_ambient_noise(source, duration=1)
    try:
        audio = r.listen(source, timeout=5)
        name = r.recognize_google(audio).lower()
        return name
    except:
        return None
--- WEB SERVER ---

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

--- LOGIC LOOP (THE BRAIN) ---

def exam_logic_loop():
global current_detected_name, last_processed_name

code
Code
download
content_copy
expand_less
while True:
    # Wait until we see someone new
    if current_detected_name and current_detected_name != last_processed_name:
        
        name = current_detected_name
        print(f" [LOGIC] Processing student: {name}")
        
        if name == "Unknown":
            # Scenario A: Face not in database
            spoken_name = listen_for_name()
            if spoken_name:
                speak(f"I heard {spoken_name}. Checking database manually.")
                # You could add manual logic here
                name = spoken_name.replace(" ", "_") # format to match DB keys
        
        # Scenario B: Known Face (or manually spoken name)
        is_registered = check_registration(name)
        
        if is_registered is True:
            speak(f"Hello {name}. You are registered for this test.")
            time.sleep(0.2)
            speak("Please be aware. Malpractice is strictly prohibited.")
            speak("No phones, no smartwatches, and keep your eyes on your own paper.")
            speak("You may enter.")
            last_processed_name = name # Mark as done so we don't repeat
            
        elif is_registered is False:
            speak(f"Alert. {name}, you are NOT registered for this exam.")
            speak("Please leave the hall immediately.")
            last_processed_name = name
            
        elif is_registered is None:
            # Recognized face, but not in JSON file
            speak(f"I recognize you, {name}, but I cannot find your registration data.")
        
        # Reset after a while to allow re-checking?
        # For now, we keep last_processed_name to avoid loops.
        
    time.sleep(0.5)
--- VISION LOOP ---

def vision_loop():
global video_frame, current_detected_name

code
Code
download
content_copy
expand_less
# Use OpenCV Capture (simpler for Face Rec than Picamera2 raw data)
cap = cv2.VideoCapture(0) # Or use the GStreamer pipeline if on Pi 5

while True:
    ret, frame = cap.read()
    if not ret: continue
    
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Find all the faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        face_names.append(name)
        
        # Update global for the logic loop
        current_detected_name = name
    
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up since we scaled down by 1/4
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    video_frame = frame
--- MAIN ENTRY ---

if name == "main":
# Load data first
load_student_data()

code
Code
download
content_copy
expand_less
# Start Vision
t_vis = threading.Thread(target=vision_loop, daemon=True)
t_vis.start()

# Start Logic
t_log = threading.Thread(target=exam_logic_loop, daemon=True)
t_log.start()

# Start Web
app.run(host='0.0.0.0', port=5000, debug=False)

terminal:
(venv) fyplg@FYPLG:~/Desktop/pi/exam $ python main.py
/home/fyplg/Desktop/pi/venv/lib/python3.13/site-packages/face_recognition_models/init.py:7: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
from pkg_resources import resource_filename
[DB] Loading Student Faces...

Loaded: john_doe
[DB] System Ready. Known students: 1

Serving Flask app 'main'

Debug mode: off
