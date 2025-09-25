import cv2
from time import sleep
from networktables import NetworkTables
from utils import get_infos
from ultralytics import YOLO

MODEL_PATH = "./models/train3-INT8.onnx"
model = YOLO(MODEL_PATH, task="detect")

cap = cv2.VideoCapture(1)

"""
FOV = 2 * arctan (W ou H / 2) / D
W ou H = tamanho do objeto
D = distância da câmera do objeto   
"""

#MIN CONF, QUANTIDADE

# ROBORIO_IP = "roborio-9485-frc.local"
class Vision:
    def __init__(self):
        self.min_conf = 0
        self.detection_quantity = 0

    def valueChanged(self, key, value, isNew):
        MIN_CONF_KEY = "//raspberry/detections/config/min_conf"

        if key == MIN_CONF_KEY:
            self.min_conf = float(value)
    
    def main(self):
        ROBORIO_IP = "localhost"
        NetworkTables.initialize(server=ROBORIO_IP)
        while not NetworkTables.isConnected(): pass

        raspberry_table = NetworkTables.getTable("raspberry")
        detections = raspberry_table.getSubTable("detections")
        locations_table = detections.getSubTable("locations")
        tx_entry = locations_table.getEntry("tx")
        ty_entry = locations_table.getEntry("ty")
        cls_entry = locations_table.getEntry("cls")

        config_table = detections.getSubTable("config")
        min_conf_entry = config_table.getEntry("min_conf")

        min_conf_entry.setNumber(1)

        NetworkTables.addEntryListener(self.valueChanged)

        camera_resolution = {
            "x": int,
            "y": int
        }

        if cap.isOpened():
            camera_resolution["x"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            camera_resolution["y"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        h = 1
        w = 2
        d = 2.25

        w_degrees_fov = get_infos.Calcs.calculate_width_fov(w, d, False)
        h_degrees_fov = get_infos.Calcs.calculate_heigth_fov(h, d, False)

        calcs = get_infos.Calcs(
            vertical_fov=h_degrees_fov,
            horizontal_fov=w_degrees_fov,
            camera_resolution=(camera_resolution["x"], camera_resolution["y"])
        )

        while cap.isOpened():
            ret, frame = cap.read()
            results = model(source=frame, verbose=False)

            if not ret:
                break

            counter = -1
            teste = {
                "0": 1
            }
            detections_conf = []
            has_detection = False


            for objects in results:
                counter += 1
                boxes = objects.boxes

                if len(boxes) == 0:
                    tx, ty, cls = 0, 0, ""
                    tx_entry.setNumber(tx)
                    ty_entry.setNumber(ty)
                    cls_entry.setString(cls)
                    continue

                for box in boxes:
                    if float(box.conf) <= self.min_conf:
                        continue

                    has_detection = True

                    teste.update(
                        {
                            counter: box
                        }
                    )

                    detections_conf.append(float(box.conf))
            
            if has_detection:
                higher_value = max(detections_conf)
                higher_value_index = detections_conf.index(higher_value)

                box = teste[higher_value_index]

                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cam_x, cam_y = int(camera_resolution["x"] / 2), int(camera_resolution["y"] / 2)

                cls = int(box.cls[0])
                vertical_pixel_offset = cam_x - cx
                horizontal_pixel_offset = cam_y - cy

                ty = calcs.get_horizontal_angle_offset(
                    horizontal_pixel_offset,
                    False
                )

                tx = calcs.get_vertical_angle_offset(
                    vertical_pixel_offset,
                    False
                )

                tx = round(tx, 2)
                ty = round(ty, 2)

                tx_entry.setNumber(tx)
                ty_entry.setNumber(ty)
                cls_entry.setString("ball")

                print(f"\ntx: {tx}")
                print(f"ty: {ty}")
                print(f"cls: {cls}\n")

                cv2.putText(frame, f"tx: {tx}", (x1, y1-30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(frame, f"ty: {ty}", (x1, y1-20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(frame, f"cls: {cls}", (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 253), 2)
                cv2.rectangle(frame, (cx, cy), (cx, cy), (255, 0, 253), 5)
                cv2.rectangle(frame, (cam_x, cam_y), (cam_x, cam_y), (0, 0, 0), 8)

                cv2.imshow("", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

Vision().main()