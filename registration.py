import threading
import cv2
import os
import time
import json
import face_recognition
import numpy as np
import logging
import hashlib
import subprocess
from flask import Flask, render_template, Response, request, jsonify
from gtts import gTTS

# --- CONFIGURATION ---
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Global Variables
video_frame = None
frame_lock = threading.Lock()  # Mutex for frame access
db_lock = threading.Lock()     # Mutex for students.json + known_face_* updates

known_face_encodings = []
known_face_names = []

current_detected_name = None
current_face_location = None
current_status = "SYSTEM READY"

# Flags for Thread Safety
scan_active = True  # Controls if Dlib runs or not
pending_registration_data = None
registration_frame_buffer = None

# --- DATABASE ---
def _read_students_db(json_file="students.json"):
    if not os.path.exists(json_file):
        return {}
    try:
        with open(json_file, "r") as f:
            content = f.read().strip()
            return json.loads(content) if content else {}
    except:
        return {}

def _sanitize_id(student_id: str) -> str:
    sid = str(student_id).strip()
    sid = "".join(ch for ch in sid if ch.isalnum() or ch in ("-", "_"))
    return sid or "UNKNOWN_ID"

def load_student_data():
    print(" [DB] Loading Student Faces...")
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []

    path = "student_pics"
    if not os.path.exists(path):
        os.makedirs(path)

    for file in os.listdir(path):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            try:
                img = face_recognition.load_image_file(f"{path}/{file}")
                encs = face_recognition.face_encodings(img)
                if encs:
                    # File format: Name__StudentID.jpeg
                    base = os.path.splitext(file)[0]
                    display_name = base.split("__")[0]  # show only name
                    known_face_encodings.append(encs[0])
                    known_face_names.append(display_name)
                    print(f"   + Loaded: {file}")
            except:
                pass

def save_new_student(name, student_id, frame):
    """
    Prevent duplicate registration by:
      1) student_id already exists in students.json
      2) captured face already matches an existing encoding
    Returns: (ok: bool, message: str)
    """
    try:
        if frame is None:
            return False, "Camera frame missing."

        clean_name = name.strip().replace(" ", "_")
        sid_clean = _sanitize_id(student_id)

        # Face encoding from captured frame (convert BGR -> RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        new_encs = face_recognition.face_encodings(rgb_frame)
        if not new_encs:
            return False, "No clear face found. Try again."
        new_enc = new_encs[0]

        with db_lock:
            # Load DB
            json_file = "students.json"
            db = _read_students_db(json_file)

            # 1) Duplicate ID check
            if sid_clean in db:
                existing_name = db[sid_clean].get("full_name") or db[sid_clean].get("name_key") or "student"
                return False, f"Duplicate blocked: ID already registered for {existing_name}."

            # 2) Duplicate face check
            if known_face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, new_enc, tolerance=0.45)
                if True in matches:
                    idx = matches.index(True)
                    existing_label = known_face_names[idx] if idx < len(known_face_names) else "student"
                    return False, f"Duplicate blocked: Face already registered as {existing_label.replace('_',' ')}."

            # Ensure folder exists
            if not os.path.exists("student_pics"):
                os.makedirs("student_pics")

            # Save as Name__ID to prevent overwriting and support same-name different IDs
            img_path = f"student_pics/{clean_name}__{sid_clean}.jpeg"
            cv2.imwrite(img_path, frame)

            # Write DB keyed by student_id (unique)
            db[sid_clean] = {
                "full_name": name,
                "student_id": sid_clean,
                "name_key": clean_name,
                "image_path": img_path,
                "registered": True,
                "timestamp": time.time()
            }

            with open(json_file, "w") as f:
                json.dump(db, f, indent=4)

            # Update in-memory lists
            known_face_encodings.append(new_enc)
            known_face_names.append(clean_name)

        return True, "Registration saved."

    except Exception as e:
        print(f" [ERROR] Save failed: {e}")
        return False, "Save failed due to an internal error."

# --- UTILS ---
def get_current_frame():
    """Thread-safe way to get a copy of the current frame"""
    with frame_lock:
        if video_frame is not None:
            return video_frame.copy()
    return None

def check_lighting(frame, face_loc):
    if frame is None or face_loc is None:
        return False, 0
    try:
        top, right, bottom, left = face_loc
        h, w, _ = frame.shape
        top = max(0, top)
        left = max(0, left)
        bottom = min(h, bottom)
        right = min(w, right)

        face_roi = frame[top:bottom, left:right]
        if face_roi.size == 0:
            return False, 0

        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        return avg_brightness > 50, avg_brightness  # Threshold
    except:
        return False, 0

def speak(text):
    global current_status
    print(f"Robot: {text}")
    filename = hashlib.md5(text.encode()).hexdigest() + ".mp3"
    file_path = f"audio_cache/{filename}"

    if not os.path.exists("audio_cache"):
        os.makedirs("audio_cache")
    if not os.path.exists(file_path):
        try:
            tts = gTTS(text=text, lang='en', tld='co.uk')
            tts.save(file_path)
        except:
            return

    try:
        subprocess.run(["mpg123", "-q", file_path], timeout=5)
    except Exception:
        pass

def update_status(status):
    global current_status
    current_status = status

# --- FLASK ---
@app.route('/')
def index():
    return render_template('register.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with frame_lock:
                if video_frame is not None:
                    ret, buf = cv2.imencode('.jpg', video_frame)
                    frame_bytes = buf.tobytes() if ret else None
                else:
                    frame_bytes = None

            if frame_bytes:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            time.sleep(0.05)  # Cap at 20 FPS for stability
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status_feed')
def status_feed():
    def generate():
        while True:
            yield f"data: {current_status}\n\n"
            time.sleep(0.2)
    return Response(generate(), mimetype="text/event-stream")

@app.route('/register_student', methods=['POST'])
def register_student():
    global pending_registration_data
    data = request.json or {}

    if not data.get('name') or not data.get('id'):
        return jsonify({"status": "error"}), 400

    # Prevent double-submit spam while one registration is pending
    if pending_registration_data is not None:
        return jsonify({"status": "busy", "message": "Registration already pending."}), 409

    pending_registration_data = data
    return jsonify({"status": "received"}), 200

# --- LOGIC LOOP ---
def logic_loop():
    global current_detected_name, pending_registration_data, registration_frame_buffer, current_face_location, scan_active

    time.sleep(2)
    speak("System Online.")

    while True:
        update_status("WAITING FOR STUDENT")
        scan_active = True  # Enable Vision

        if current_detected_name is None:
            time.sleep(0.5)
            continue

        detected = current_detected_name

        # 1. KNOWN STUDENT
        if detected != "Unknown":
            update_status(f"WELCOME BACK {detected}")
            speak(f"Hello {detected.replace('_', ' ')}.")
            time.sleep(1)
            while current_detected_name is not None:
                time.sleep(0.5)
            continue

        # 2. NEW STUDENT
        if detected == "Unknown":
            update_status("NEW STUDENT DETECTED")
            speak("Hello. I do not recognize you.")
            time.sleep(0.5)

            update_status("PREPARING FOR PHOTO")
            speak("Please look directly at the camera.")
            time.sleep(0.5)
            speak("Please remove any hats, glasses, or hair covering your face.")
            time.sleep(3)

            # --- LIGHTING CHECK ---
            frame_snapshot = get_current_frame()
            if frame_snapshot is not None and current_face_location is not None:
                is_bright, _ = check_lighting(frame_snapshot, current_face_location)
                if not is_bright:
                    update_status("LIGHTING TOO DARK")
                    speak("The lighting is too dark. Please find better light.")
                    time.sleep(3)
                    continue

            # --- CAPTURE ---
            speak("Hold still.")
            time.sleep(1.0)

            registration_frame_buffer = get_current_frame()
            if registration_frame_buffer is None:
                speak("Camera error.")
                continue

            speak("Face captured.")
            update_status("CAPTURED - PAUSING VISION")

            # --- INPUT PHASE (PAUSE VISION) ---
            scan_active = False
            time.sleep(0.5)

            speak("Please enter your Name and ID on the screen.")
            update_status("INPUT REQUIRED")

            timer = 0
            while pending_registration_data is None:
                time.sleep(0.5)
                timer += 1
                if timer > 120:
                    break  # ~60 seconds

            if pending_registration_data:
                name = pending_registration_data['name']
                sid = pending_registration_data['id']

                update_status("SAVING...")
                ok, msg = save_new_student(name, sid, registration_frame_buffer)
                if ok:
                    update_status("REGISTRATION COMPLETE")
                    speak("Registration successful. Thank you.")
                else:
                    update_status("REGISTRATION BLOCKED")
                    speak(msg)

                pending_registration_data = None
                registration_frame_buffer = None

                scan_active = True
                speak("Please step aside.")
                while current_detected_name is not None:
                    time.sleep(1)
            else:
                update_status("TIMEOUT")
                speak("Registration timed out. Resetting.")
                pending_registration_data = None
                registration_frame_buffer = None
                scan_active = True
                time.sleep(2)

# --- VISION LOOP ---
def vision_loop():
    global video_frame, current_detected_name, current_face_location, scan_active

    pipelines = [
        "libcamerasrc ! video/x-raw, width=640, height=480, framerate=15/1, format=YUY2 ! videoconvert ! video/x-raw, format=BGR ! appsink drop=true max-buffers=1 sync=false",
        0
    ]

    cap = None
    for pipeline in pipelines:
        if isinstance(pipeline, str):
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            cap = cv2.VideoCapture(pipeline)
        if cap.isOpened():
            break

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame_count += 1

        # Only do heavy face recognition if allowed
        if scan_active and frame_count % 5 == 0:
            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            locs = face_recognition.face_locations(rgb)
            encs = face_recognition.face_encodings(rgb, locs)

            det_name = None
            curr_loc = None

            if locs and encs:
                enc = encs[0]
                loc = locs[0]
                curr_loc = (loc[0] * 4, loc[1] * 4, loc[2] * 4, loc[3] * 4)

                name = "Unknown"
                if known_face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, enc, tolerance=0.5)
                    dists = face_recognition.face_distance(known_face_encodings, enc)
                    if len(dists) > 0:
                        best = np.argmin(dists)
                        if matches[best]:
                            name = known_face_names[best]

                det_name = name

            current_detected_name = det_name
            current_face_location = curr_loc

        # Draw last known box/location
        if current_face_location:
            top, right, bottom, left = current_face_location
            color = (0, 255, 0) if current_detected_name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            if current_detected_name:
                cv2.putText(frame, str(current_detected_name), (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        final_frame = frame.copy()
        with frame_lock:
            video_frame = final_frame

if __name__ == "__main__":
    load_student_data()
    threading.Thread(target=vision_loop, daemon=True).start()
    threading.Thread(target=logic_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
