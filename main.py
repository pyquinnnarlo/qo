# ======================
# ENTRIES APP (FULL UPDATED - ACCURACY FIX)
# Improvements Applied:
# - Detect On Small Frame (Fast) But Encode On Full Frame (Accurate)
# - Stricter Threshold + Winner Margin (Best Must Beat Runner-Up)
# - Require N Consistent Hits Before Confirming Identity
# - Use Unique Labels Name__ID (Speak/Display Only Name)
# - Keep Native Single-Threading + OpenCV Single-Thread
# - Serialize All face_recognition (dlib) Calls With A Global Lock
# ======================

import os

# --- Prevent Native Thread Races (dlib/OpenMP/OpenBLAS) ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import threading
import cv2
cv2.setNumThreads(1)

import speech_recognition as sr
import time
import json
import face_recognition
import numpy as np
import logging
import hashlib
import subprocess
from flask import Flask, render_template, Response
from gtts import gTTS
from ctypes import *
from contextlib import contextmanager

# --- Serialize All face_recognition (dlib) Calls ---
fr_lock = threading.Lock()

# --- ALSA ERROR SUPPRESSION ---
ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
def py_error_handler(filename, line, function, err, fmt):
    pass
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

@contextmanager
def no_alsa_errors():
    try:
        asound = cdll.LoadLibrary("libasound.so")
        asound.snd_lib_error_set_handler(c_error_handler)
        yield
        asound.snd_lib_error_set_handler(None)
    except:
        yield

# --- CONFIGURATION ---
app = Flask(__name__)
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

UNKNOWN_LABEL = "Unknown"
MULTI_LABEL = "MULTIPLE"

# --- RECOGNITION TUNING (IMPORTANT) ---
MATCH_THRESHOLD = 0.43   # lower = stricter, reduces false positives
MATCH_MARGIN    = 0.06   # best must beat runner-up by this margin
CONFIRM_HITS    = 3      # same identity must win this many scans before accepted

MIC_INDEX = None

# Thread safety
frame_lock = threading.Lock()
db_lock = threading.Lock()

# Global Variables
video_frame = None
known_face_encodings = []
known_face_names = []   # UNIQUE LABELS: Name__ID
current_detected_name = None
current_face_count = 0
current_status = "SYSTEM READY"

# Confirmation filter state
candidate_label = None
candidate_hits = 0

# --- DB HELPERS (MATCH UPDATED REGISTRATION CODE) ---
def _read_students_db(json_file="students.json"):
    if not os.path.exists(json_file):
        return {}
    try:
        with open(json_file, "r") as f:
            content = f.read().strip()
            return json.loads(content) if content else {}
    except:
        return {}

def _normalize_name_key(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")

def _display_name(label: str) -> str:
    if not label:
        return ""
    base = label.split("__")[0] if "__" in label else label
    return base.replace("_", " ")

def load_student_data():
    """
    Loads face encodings from student_pics/Name__StudentID.jpeg
    Stores recognition label as UNIQUE label: Name__ID (from filename base).
    """
    print(" [DB] Loading Student Faces...")
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []

    path = "student_pics"
    if not os.path.exists(path):
        os.makedirs(path)
        print(" [DB] 'student_pics' folder created. Add images there.")
        return

    for file in os.listdir(path):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            try:
                img = face_recognition.load_image_file(os.path.join(path, file))

                with fr_lock:
                    encodings = face_recognition.face_encodings(img)

                if encodings:
                    base = os.path.splitext(file)[0]
                    label = base  # UNIQUE: Name__ID
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(label)
                    print(f"   + Loaded: {label}")
                else:
                    print(f"   - Warning: No face found in {file}")
            except Exception as e:
                print(f"   - Error loading {file}: {e}")

    print(f" [DB] System Ready. Known students: {len(known_face_names)}")

def _best_match_label(new_enc: np.ndarray):
    """
    Returns (label or None, best_dist, second_best_dist)
    Uses strict threshold + margin logic.
    """
    if not known_face_encodings:
        return None, None, None

    with fr_lock:
        dists = face_recognition.face_distance(known_face_encodings, new_enc)

    if len(dists) == 0:
        return None, None, None

    best_idx = int(np.argmin(dists))
    best_dist = float(dists[best_idx])

    if len(dists) > 1:
        sorted_d = np.sort(dists)
        second_best = float(sorted_d[1])
    else:
        second_best = 999.0

    if (best_dist < MATCH_THRESHOLD) and ((second_best - best_dist) > MATCH_MARGIN):
        return known_face_names[best_idx], best_dist, second_best

    return None, best_dist, second_best

def check_registration_by_face_label(face_label: str):
    """
    students.json is keyed by student_id in updated registration code.
    For entries, validate by matching record['student_id'] (from label __ID).
    Falls back to matching record['name_key'] if needed.
    """
    try:
        with db_lock:
            db = _read_students_db("students.json")

        if not db:
            return None

        label = str(face_label)
        student_id = None
        if "__" in label:
            student_id = label.split("__", 1)[1].strip()

        # Primary: match by student_id (most reliable)
        if student_id and student_id in db and isinstance(db[student_id], dict):
            return bool(db[student_id].get("registered", False))

        # Fallback: match by name_key if student_id missing
        target_name_key = _normalize_name_key(label.split("__")[0] if "__" in label else label)
        for _, rec in db.items():
            if not isinstance(rec, dict):
                continue
            if _normalize_name_key(rec.get("name_key", "")) == target_name_key:
                return bool(rec.get("registered", False))

        return None
    except Exception as e:
        print(f" [DB ERROR] {e}")
        return None

# --- FRAME HELPERS ---
def get_current_frame():
    with frame_lock:
        if video_frame is not None:
            return video_frame.copy()
    return None

def save_denied_photo(frame, reason="unknown"):
    """
    Saves a screenshot of the denied student for audit.
    - Saves full frame + metadata json sidecar.
    - Only saves if exactly 1 face is present.
    """
    try:
        if frame is None:
            return False

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with fr_lock:
            locs = face_recognition.face_locations(
                rgb, number_of_times_to_upsample=0, model="hog"
            )

        if len(locs) != 1:
            return False

        folder = "denied_pics"
        if not os.path.exists(folder):
            os.makedirs(folder)

        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"DENIED_{reason}_{ts}.jpg"
        path = os.path.join(folder, filename)

        cv2.imwrite(path, frame)

        meta = {
            "timestamp_unix": time.time(),
            "timestamp_local": ts,
            "reason": reason,
            "faces_detected": len(locs),
            "image_path": path,
        }
        with open(os.path.join(folder, f"DENIED_{reason}_{ts}.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f" [DENIED] Saved: {path}")
        return True
    except Exception as e:
        print(f" [DENIED ERROR] {e}")
        return False

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

    if not os.path.exists("audio_cache"):
        os.makedirs("audio_cache")

    if not os.path.exists(file_path):
        try:
            tts = gTTS(text=text, lang="en", tld="co.uk")
            tts.save(file_path)
        except Exception as e:
            print(f" [ERROR] Could not generate TTS: {e}")
            update_status(prev_status)
            return

    try:
        subprocess.run(["mpg123", "-q", file_path], timeout=6)
    except Exception:
        pass

    update_status(prev_status)

def listen_for_name():
    update_status("LISTENING")
    speak("Face not recognized. Please state your name.")

    r = sr.Recognizer()

    with no_alsa_errors():
        try:
            mic = sr.Microphone(device_index=MIC_INDEX)
            with mic as source:
                r.adjust_for_ambient_noise(source, duration=0.5)
                audio = r.listen(source, timeout=4, phrase_time_limit=4)

                update_status("THINKING")
                spoken = r.recognize_sphinx(audio).lower()
                return _normalize_name_key(spoken)
        except:
            return None
        finally:
            update_status("IDLE")

# --- WEB SERVER ---
@app.route("/")
def index():
    return render_template("index.html")

def generate_frames():
    global video_frame
    while True:
        with frame_lock:
            frame = video_frame.copy() if video_frame is not None else None

        if frame is not None:
            try:
                ret, buffer = cv2.imencode(".jpg", frame)
                if ret:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
            except:
                pass

        time.sleep(0.04)

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status_feed")
def status_feed():
    def generate():
        while True:
            yield f"data: {current_status}\n\n"
            time.sleep(0.5)
    return Response(generate(), mimetype="text/event-stream")

# --- LOGIC LOOP ---
def exam_logic_loop():
    global current_detected_name, current_face_count

    time.sleep(3)
    speak("System Online.")

    last_processed = None
    last_processed_time = 0
    last_multi_warn = 0

    while True:
        update_status("WAITING FOR FACE")

        if current_detected_name is None:
            time.sleep(0.4)
            continue

        # MULTI-FACE BLOCK
        if current_detected_name == MULTI_LABEL or current_face_count > 1:
            update_status("MULTIPLE FACES - ONE STUDENT ONLY")
            if time.time() - last_multi_warn > 6:
                speak("Multiple faces detected. One student at a time. Please step back.")
                last_multi_warn = time.time()
            while current_face_count > 1:
                time.sleep(0.5)
            continue

        # Debounce
        if current_detected_name == last_processed and (time.time() - last_processed_time) < 8:
            time.sleep(0.5)
            continue

        update_status("INSTRUCTING STUDENT")
        speak("Hi. Look directly at the camera. Remove anything covering your face.")
        time.sleep(2)

        if current_detected_name is None:
            speak("Face lost. Process reset.")
            continue

        # Multi-face check again
        if current_face_count > 1 or current_detected_name == MULTI_LABEL:
            update_status("MULTIPLE FACES - BLOCKED")
            speak("Multiple faces detected. One student at a time.")
            continue

        label = current_detected_name  # UNIQUE label: Name__ID
        update_status(f"PROCESSING: {label}")

        # --- DENY IF NOT RECOGNIZED ---
        if label == UNKNOWN_LABEL:
            update_status("DENIED - NOT RECOGNIZED")
            snap = get_current_frame()
            save_denied_photo(snap, reason="not_recognized")

            speak("Access denied. You are not recognized.")
            speak("Please try again. If the problem continues, contact the administration.")

            last_processed = "DENIED_UNKNOWN"
            last_processed_time = time.time()

            while current_detected_name == UNKNOWN_LABEL:
                time.sleep(0.8)

            continue

        # If recognized but not in DB / not registered -> deny and snapshot
        is_registered = check_registration_by_face_label(label)

        if is_registered is True:
            speak(f"Hello {_display_name(label)}.")
            time.sleep(0.2)
            speak("You are registered.")

            update_status("READING RULES")
            speak("Attention. Please listen to the exam rules.")
            time.sleep(0.2)
            speak("No phones or smart watches are allowed in the exam hall or during the exam.")
            time.sleep(0.2)
            speak("Failure to follow these rules will result in the dismissal of the test.")
            time.sleep(0.2)

            speak("You may enter now.")

            last_processed = label
            last_processed_time = time.time()

            while current_detected_name == label:
                time.sleep(1)

            speak("Next student.")

        else:
            reason = "not_registered" if is_registered is False else "not_found_in_db"
            update_status("DENIED - NOT REGISTERED")
            snap = get_current_frame()
            save_denied_photo(snap, reason=reason)

            speak(f"Access denied. {_display_name(label)}, you are not registered.")
            speak("Please try again or contact the administration.")

            last_processed = f"DENIED_{label}"
            last_processed_time = time.time()

            while current_detected_name == label:
                time.sleep(1)

        time.sleep(0.5)

# --- VISION LOOP ---
def vision_loop():
    global video_frame, current_detected_name, current_face_count
    global candidate_label, candidate_hits

    print(" [VISION] Starting Camera (USB/V4L2)...")

    pipelines = [
        "libcamerasrc ! video/x-raw, width=640, height=480, framerate=15/1, format=YUY2 ! videoconvert ! video/x-raw, format=BGR ! appsink drop=true max-buffers=1 sync=false",
        0
    ]

    cap = None
    opened_with = None
    for pipeline in pipelines:
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER) if isinstance(pipeline, str) else cv2.VideoCapture(pipeline)
        if cap.isOpened():
            opened_with = pipeline
            break

    if not cap or not cap.isOpened():
        print(" [ERROR] Could not open any camera.")
        return

    if isinstance(opened_with, str):
        print(" [VISION] Camera opened (GStreamer).")
    else:
        print(" [VISION] Camera opened (OpenCV V4L2): index 0")

    frame_count = 0
    cached_face_locations_small = []
    cached_face_names = []

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            frame_count += 1

            # Reduce Scan Frequency
            if frame_count % 10 == 0:
                # Detect on small frame
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                with fr_lock:
                    cached_face_locations_small = face_recognition.face_locations(
                        rgb_small, number_of_times_to_upsample=0, model="hog"
                    )

                current_face_count = len(cached_face_locations_small)
                cached_face_names = []

                if current_face_count > 1:
                    current_detected_name = MULTI_LABEL
                    cached_face_names = [MULTI_LABEL] * current_face_count
                    candidate_label = None
                    candidate_hits = 0

                elif current_face_count == 1:
                    # Convert small-loc -> full-loc
                    (t, r, b, l) = cached_face_locations_small[0]
                    full_loc = (t*4, r*4, b*4, l*4)

                    # Encode on FULL frame for accuracy
                    rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    with fr_lock:
                        encs_full = face_recognition.face_encodings(rgb_full, [full_loc])

                    name = UNKNOWN_LABEL
                    if encs_full and known_face_encodings:
                        label, _, _ = _best_match_label(encs_full[0])
                        if label is not None:
                            name = label

                    # Confirmation filter
                    if name == UNKNOWN_LABEL:
                        candidate_label = None
                        candidate_hits = 0
                        current_detected_name = UNKNOWN_LABEL
                        cached_face_names = [UNKNOWN_LABEL]
                    else:
                        if candidate_label == name:
                            candidate_hits += 1
                        else:
                            candidate_label = name
                            candidate_hits = 1

                        if candidate_hits >= CONFIRM_HITS:
                            current_detected_name = name
                            cached_face_names = [name]
                        else:
                            current_detected_name = UNKNOWN_LABEL
                            cached_face_names = [UNKNOWN_LABEL]

                else:
                    current_detected_name = None
                    cached_face_names = []
                    candidate_label = None
                    candidate_hits = 0

            # Draw rectangles from cached results (scale from small to full)
            for (top, right, bottom, left), name in zip(cached_face_locations_small, cached_face_names):
                top *= 4; right *= 4; bottom *= 4; left *= 4

                if name == MULTI_LABEL:
                    color = (0, 165, 255)
                    label_txt = "MULTIPLE"
                else:
                    color = (0, 255, 0) if name not in (UNKNOWN_LABEL, None) else (0, 0, 255)
                    if name not in (UNKNOWN_LABEL, None, MULTI_LABEL):
                        label_txt = _display_name(str(name))
                    else:
                        label_txt = UNKNOWN_LABEL if name == UNKNOWN_LABEL else ""

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, label_txt, (left + 6, bottom - 6),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            with frame_lock:
                video_frame = frame.copy()

        except Exception as e:
            print(f" [VISION WARNING] {e}")
            time.sleep(0.1)

if __name__ == "__main__":
    load_student_data()
    threading.Thread(target=vision_loop, daemon=True).start()
    threading.Thread(target=exam_logic_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
