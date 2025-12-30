sudo apt install gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-tools -y

# Raspberry Pi 5 Pipeline (Forces YUY2 format to fix "Internal data stream error")
pipeline = (
    "libcamerasrc ! "
    "video/x-raw, width=640, height=480, framerate=30/1, format=YUY2 ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink"
)
