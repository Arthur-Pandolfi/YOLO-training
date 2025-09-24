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

while cap.isOpened():
    ret, frame = cap.read()
    results = model(source=frame)

    for objects in results:
        obj = objects.boxes

        for dados in obj:
            x1, y1, x2, y2 = dados.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            classes = int(dados.cls[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            print(cx, cy)
            print(f"cls: {classes}")
            img = cv2.rectangle(frame, (cx, cy), (cx, cy), (255, 0, 253), 5)

            cv2.imshow("",frame)
            cv2.waitKey(1000)


cap.release()
cv2.destroyAllWindows()