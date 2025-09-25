from ultralytics import YOLO
import cv2

MODEL_PATH = "C:/Users/Arthur/source/repos/FRC-Programas/object-detection/build-season-volley-project/models/train3-INT8.onnx"
model = YOLO(MODEL_PATH)

VIDEO_PATH = "C:/Users/Arthur/source/repos/FRC-Programas/object-detection/build-season-volley-project/videos/video-6.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

# DESLOCAMENTO EM GRAUS:
# (deslocamente em px / resolucao em px) * FOV

"""
FOV = 2 * arctan (W ou H / 2) / D
W ou H = tamanho do objeto
D = distância da câmera do objeto   
"""
camera_resolution = {
    "x": int,
    "y": int
}
if cap.isOpened():
    camera_resolution["x"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    camera_resolution["y"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(camera_resolution)

while cap.isOpened():
    ret, frame = cap.read()
    results = model(source=frame)

    for objects in results:
        boxes = objects.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            classes = int(box.cls[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cam_x, cam_y = int(camera_resolution["x"] / 2), int(camera_resolution["y"] / 2)

            print(f"cls: {classes}")
            img = cv2.rectangle(frame, (cx, cy), (cx, cy), (255, 0, 253), 5)
            img = cv2.rectangle(frame, (cam_x, cam_y), (cam_x, cam_y), (255, 0, 255), 5)

            cv2.imshow("",img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()