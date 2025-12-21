from ultralytics import YOLO
from picamera2 import Picamera2
import cv2

# Load YOLOv8 nano model (fastest)
model = YOLO("yolov8n.pt")

# Initialize camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
))
picam2.start()

print("Starting YOLO object detection... Press Q to quit")

while True:
    frame = picam2.capture_array()

    # Run YOLO inference
    results = model(frame, verbose=False)

    # Draw results
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
picam2.stop()
