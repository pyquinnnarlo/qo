pip uninstall opencv-python opencv-contrib-python -y
python -m venv --system-site-packages venv
source venv/bin/activate
pip install face_recognition Flask SpeechRecognition pyaudio
