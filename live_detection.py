import cv2
import torch
from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "my_model" / "my_model.pt"
CONF = 0.5
IMGSZ = 320
CAMERA_INDEX = 0

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"デバイス: {device}")

model = YOLO(str(MODEL_PATH))
model.to(device)
print(f"クラス: {model.names}")
print("終了: q キー")

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, imgsz=IMGSZ, conf=CONF, verbose=False, device=device)
    annotated = results[0].plot()

    cv2.imshow("YOLO Live", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
