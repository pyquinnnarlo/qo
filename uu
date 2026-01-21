# =========================
# REGISTRATION APP (UPDATED)
# Multi-Face Blocking Added
# =========================

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

UNKNOWN_LABEL = "Unknown"
MULTI_LABEL = "MULTIPLE"  # internal label for multi-face state

# Global Variables
video_frame = None
frame_lock = threading.Lock()  # Mutex for frame access
db_lock = threading.Lock()     # Mutex for students.json + known_face_* updates

known_face_encodings = []
known_face_names = []

current_detected_name = None
current_face_location = None          # single-face best location
current_face_locations = []           # all face boxes
current_face_count = 0                # number of faces
current_status = "SYSTEM READY"

# Flags for Thread Safety
scan_active = True
pending_registration_data = None
registration_frame_buffer = None

# --- DB HELPERS ---
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

# --- DATABASE ---
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
                img = face_recognition.load_image_file(os.path.join(path, file))
                encs = face_recognition.face_encodings(img)
                if encs:
                    base = os.path.splitext(file)[0]
                    display_name = base.split("__")[0]  # Name__ID -> Name
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
      3) multi-face / no-face capture is blocked
    Returns: (ok: bool, message: str)
    """
    try:
        if frame is None:
            return False, "Camera frame missing."

        clean_name = name.strip().replace(" ", "_")
        sid_clean = _sanitize_id(student_id)

        # Face encoding from captured frame (convert BGR -> RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb_frame)
        if len(locs) != 1:
            if len(locs) == 0:
                return False, "No face found. Try again."
            return False, "Multiple faces detected. One student at a time."

        new_encs = face_recognition.face_encodings(rgb_frame, locs)
        if not new_encs:
            return False, "Face not clear. Try again."
        new_enc = new_encs[0]

        with db_lock:
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

            if not os.path.exists("student_pics"):
                os.makedirs("student_pics")

            # Save as Name__ID
            img_path = f"student_pics/{clean_name}__{sid_clean}.jpeg"
            cv2.imwrite(img_path, frame)

            # DB keyed by student_id
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

            known_face_encodings.append(new_enc)
            known_face_names.append(clean_name)

        return True, "Registration saved."
    except Exception as e:
        print(f" [ERROR] Save failed: {e}")
        return False, "Save failed due to an internal error."

# --- UTILS ---
def get_current_frame():
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
        top = max(0, top); left = max(0, left)
        bottom = min(h, bottom); right = min(w, right)

        face_roi = frame[top:bottom, left:right]
        if face_roi.size == 0:
            return False, 0

        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        return avg_brightness > 50, avg_brightness
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
        subprocess.run(["mpg123", "-q", file_path], timeout=6)
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
                frame = video_frame.copy() if video_frame is not None else None

            if frame is not None:
                ret, buf = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

            time.sleep(0.05)
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

    if pending_registration_data is not None:
        return jsonify({"status": "busy", "message": "Registration already pending."}), 409

    pending_registration_data = data
    return jsonify({"status": "received"}), 200

# --- LOGIC LOOP ---
def logic_loop():
    global current_detected_name, pending_registration_data, registration_frame_buffer
    global current_face_location, current_face_count, scan_active

    time.sleep(2)
    speak("System Online.")

    last_multi_warn = 0

    while True:
        update_status("WAITING FOR STUDENT")
        scan_active = True

        if current_detected_name is None:
            time.sleep(0.5)
            continue

        # MULTI-FACE BLOCK (GLOBAL)
        if current_detected_name == MULTI_LABEL or current_face_count > 1:
            update_status("MULTIPLE FACES - ONE STUDENT ONLY")
            if time.time() - last_multi_warn > 6:
                speak("Multiple faces detected. One student at a time. Please step back.")
                last_multi_warn = time.time()
            while current_face_count > 1:
                time.sleep(0.5)
            continue

        detected = current_detected_name

        # 1) KNOWN STUDENT
        if detected != UNKNOWN_LABEL:
            update_status(f"WELCOME BACK {detected}")
            speak(f"Hello {detected.replace('_', ' ')}.")
            time.sleep(1)
            while current_detected_name is not None:
                time.sleep(0.5)
            continue

        # 2) NEW STUDENT
        update_status("NEW STUDENT DETECTED")
        speak("Hello. I do not recognize you.")
        time.sleep(0.5)

        # Ensure exactly 1 face before instructions/capture
        if current_face_count != 1:
            update_status("ONE FACE REQUIRED")
            speak("One student only. Please make sure only one face is in view.")
            while current_face_count != 1:
                time.sleep(0.5)
            continue

        update_status("PREPARING FOR PHOTO")
        speak("Please look directly at the camera.")
        time.sleep(0.5)
        speak("Please remove any hats, glasses, or hair covering your face.")
        time.sleep(3)

        # Multi-face check again right before capture
        if current_face_count != 1:
            update_status("MULTIPLE FACES - CAPTURE BLOCKED")
            speak("Multiple faces detected. Capture blocked. One student at a time.")
            continue

        # Lighting check uses single-face location
        frame_snapshot = get_current_frame()
        if frame_snapshot is not None and current_face_location is not None:
            is_bright, _ = check_lighting(frame_snapshot, current_face_location)
            if not is_bright:
                update_status("LIGHTING TOO DARK")
                speak("The lighting is too dark. Please find better light.")
                time.sleep(3)
                continue

        speak("Hold still.")
        time.sleep(1.0)

        registration_frame_buffer = get_current_frame()
        if registration_frame_buffer is None:
            speak("Camera error.")
            continue

        # HARD BLOCK: confirm exactly 1 face in the captured frame
        rgb_cap = cv2.cvtColor(registration_frame_buffer, cv2.COLOR_BGR2RGB)
        cap_locs = face_recognition.face_locations(rgb_cap)
        if len(cap_locs) != 1:
            update_status("CAPTURE INVALID")
            if len(cap_locs) == 0:
                speak("No face found. Please try again.")
            else:
                speak("Multiple faces detected. One student at a time.")
            registration_frame_buffer = None
            continue

        speak("Face captured.")
        update_status("CAPTURED - PAUSING VISION")

        scan_active = False
        time.sleep(0.5)

        speak("Please enter your Name and ID on the screen.")
        update_status("INPUT REQUIRED")

        timer = 0
        while pending_registration_data is None:
            time.sleep(0.5)
            timer += 1
            if timer > 120:
                break

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
    global video_frame, current_detected_name
    global current_face_location, current_face_locations, current_face_count, scan_active

    pipelines = [
        "libcamerasrc ! video/x-raw, width=640, height=480, framerate=15/1, format=YUY2 ! videoconvert ! video/x-raw, format=BGR ! appsink drop=true max-buffers=1 sync=false",
        0
    ]

    cap = None
    for pipeline in pipelines:
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER) if isinstance(pipeline, str) else cv2.VideoCapture(pipeline)
        if cap.isOpened():
            break

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame_count += 1

        if scan_active and frame_count % 5 == 0:
            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            locs = face_recognition.face_locations(rgb)
            current_face_count = len(locs)

            # scale locs to full size for drawing/lighting
            scaled_locs = [(t*4, r*4, b*4, l*4) for (t, r, b, l) in locs]
            current_face_locations = scaled_locs
            current_face_location = scaled_locs[0] if scaled_locs else None

            # multi-face -> block at logic layer
            if current_face_count > 1:
                current_detected_name = MULTI_LABEL
            elif current_face_count == 1:
                encs = face_recognition.face_encodings(rgb, locs)
                name = UNKNOWN_LABEL
                if encs and known_face_encodings:
                    enc = encs[0]
                    matches = face_recognition.compare_faces(known_face_encodings, enc, tolerance=0.5)
                    dists = face_recognition.face_distance(known_face_encodings, enc)
                    if len(dists) > 0:
                        best = np.argmin(dists)
                        if matches[best]:
                            name = known_face_names[best]
                current_detected_name = name
            else:
                current_detected_name = None

        # Draw all boxes (even when scan paused; reuse last)
        if current_face_locations:
            for (top, right, bottom, left) in current_face_locations:
                if current_face_count > 1 or current_detected_name == MULTI_LABEL:
                    color = (0, 165, 255)  # orange for multi
                    label = "MULTIPLE"
                else:
                    color = (0, 255, 0) if current_detected_name != UNKNOWN_LABEL else (0, 0, 255)
                    label = str(current_detected_name) if current_detected_name else ""

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                if label:
                    cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        with frame_lock:
            video_frame = frame.copy()

if __name__ == "__main__":
    load_student_data()
    threading.Thread(target=vision_loop, daemon=True).start()
    threading.Thread(target=logic_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
