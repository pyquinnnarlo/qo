pip3 install adafruit-circuitpython-fingerprint pyserial
[Student_Face_Registration_Protocol.pdf](https://github.com/user-attachments/files/24772021/Student_Face_Registration_Protocol.pdf)
[Facial_Recognition_Protocol_Operations_Manual.pdf](https://github.com/user-attachments/files/24772016/Facial_Recognition_Protocol_Operations_Manual.pdf)


```bash
sudo apt update

# Python + build tooling
sudo apt install -y python3 python3-pip python3-venv build-essential cmake pkg-config

# OpenCV runtime deps
sudo apt install -y libatlas-base-dev libjpeg-dev libpng-dev libtiff5-dev libopenblas-dev

# dlib/face_recognition deps
sudo apt install -y libboost-all-dev

# Audio (mpg123 playback + ALSA)
sudo apt install -y mpg123 alsa-utils libasound2

# SpeechRecognition (optional for PocketSphinx)
sudo apt install -y swig

# If you use the GStreamer pipeline (libcamerasrc ...):
sudo apt install -y gstreamer1.0-tools \
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
  gstreamer1.0-libav

pip install flask numpy opencv-python gTTS SpeechRecognition face_recognition
pip install pocketsphinx


```![Abraham_Sheriff__89610](https://github.com/user-attachments/assets/532533fb-0489-4b04-8578-dc1e86316f1d)

