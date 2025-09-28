import os
import cv2
from time import sleep
from typing import Union
from utils import get_infos
from ultralytics import YOLO
from cv2.typing import MatLike
from networktables import NetworkTables
from networktables.entry import NetworkTableEntry

StrOrBytesPath = Union[str, bytes, os.PathLike[str], os.PathLike[bytes]]
class Vision:

    def __init__(self, h: float, w: float, d: float, model_path: StrOrBytesPath, camera_index: int = 0, roborio_ip: Union[int, str] = "localhost"):
        """
        This class initalizes the vision code, that run inferences and do everything needed

        Atributes
        ---------
        h : float
            The heigth (in meters) of the object used to calculate the FOV
        w : float
            The width (in meters) of the object used to calculate the FOV
        d : float
            The distance (in meters) of the object to the camera used to calculate the FOV
        model_path : StrOrBytesPath
            The path to the model used to run inferecences
        camera_index : int
            The index of the camera used to detect objects 
        roborio_ip : int | str
            The RoboRIO IP or hostname (roborio-TEAM-frc.local)
        """

        self.min_conf = 0
        self.ROBORIO_IP = roborio_ip
        self.MODEL_PATH = model_path
        self.CAMERA_INDEX = camera_index
        self.CAMERA_RESOLUTION = {}

        self.cam_x, cam_y = 0, 0
        self.w_degrees_fov = get_infos.Calcs.calculate_width_fov(w, d, False)
        self.h_degrees_fov = get_infos.Calcs.calculate_heigth_fov(h, d, False)

        self.objects = { 0: "ball" }

    def _valueChanged(self, key, value, isNew):
        """
        The function used to check if a entry value has been changed
        """

        MIN_CONF_KEY = "//raspberry/detections/config/min_conf"

        if key == MIN_CONF_KEY:
            self.min_conf = float(value)

    def show_detected_image(self, img: MatLike, tx: float, ty: float, cls: int,
                   box_x1: int,box_y1: int, box_x2: int, box_y2: int,
                   cx:int, cy:int, cam_x: int, cam_y: int) -> MatLike:
        
        """
        Used to show a image if has a Target

        :param img: The camera image
        :param tx: The "Target X" in degrees
        :param ty: The "Target Y" in degrees
        :param cls: The class ID of the target
        :param box_x1: The first X axis value of the rectangle
        :param box_y1: The first Y axis value of the rectangle
        :param box_x2: The second X axis value of the rectangle
        :param box_y2: The second Y axis value of the rectangle
        :param cx: The middle of the X axis of the target
        :param cy: The middle of the Y axis of the target
        :param cam_x: The middle of the camera on the X axis 
        :param cam_y: The middle of the camera on the Y axis 

        :return: The final image
        """

        cv2.putText(img, f"tx: {tx}", (box_x1, box_y1-30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(img, f"ty: {ty}", (box_x1, box_y1-20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(img, f"cls: {cls}", (box_x1, box_y1-5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.rectangle(img, (cx, cy), (cx, cy), (255, 0, 253), 5)
        cv2.rectangle(img, (cam_x, cam_y), (cam_x, cam_y), (0, 0, 0), 8)
        cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), (255, 0, 253), 2)
        cv2.imshow("", img)
        return img
    
    def show_not_detected_image(self, img: MatLike, cam_x: int, cam_y: int) -> MatLike:
        """
        Used to show a image if dosen't have a Target

        :param img: The camera image
        :param cam_x: The middle of the camera on the X axis 
        :param cam_y: The middle of the camera on the Y axis 

        :return: The final image
        """

        cv2.rectangle(img, (cam_x, cam_y), (cam_x, cam_y), (0, 0, 0), 8)
        cv2.imshow("", img)
        return img

    def initialize_network_tables(self) -> NetworkTableEntry:
        """
        Used to initialize the Network Tables (get the entrys and the tables)

        :return: All the entrys that will be utilized
        """

        NetworkTables.initialize(server=self.ROBORIO_IP)
        while not NetworkTables.isConnected(): pass

        raspberry_table = os.environ.get("RASPBERRY_NAME")

        raspberry_table = NetworkTables.getTable(raspberry_table)
        detections = raspberry_table.getSubTable("detections")
        locations_table = detections.getSubTable("locations")
        tx_entry = locations_table.getEntry("tx")
        ty_entry = locations_table.getEntry("ty")
        tv_entry = locations_table.getEntry("tv")
    
        cls_entry = locations_table.getEntry("cls")
        cls_name_entry = locations_table.getEntry("cls_name")

        config_table = detections.getSubTable("config")
        min_conf_entry = config_table.getEntry("min_conf")

        min_conf_entry.setNumber(1)

        NetworkTables.addEntryListener(self._valueChanged)

        return tx_entry, ty_entry, tv_entry, cls_entry, cls_name_entry

    def main(self, show_image: bool = False, model_task: str = "detect", verbose_model: bool = False):
        """
        The main function that runs the principal code
        """

        model = YOLO(self.MODEL_PATH, task=model_task)
        cap = cv2.VideoCapture(self.CAMERA_INDEX)

        camera_resolution = { "x": int, "y": int }
        tx_entry, ty_entry, tv_entry, cls_entry, cls_name_entry = self.initialize_network_tables()

        calcs = get_infos.Calcs(
            vertical_fov=self.h_degrees_fov,
            horizontal_fov=self.w_degrees_fov,
            camera_resolution=camera_resolution
        )

        if show_image:
            print("Press 'q' to quit")
            sleep(5)

        if cap.isOpened():
            camera_resolution["x"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            camera_resolution["y"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self.cam_x, self.cam_y = int(camera_resolution["x"] // 2), int(camera_resolution["y"] // 2)
        else:
            exit(f"Cannot open the cmaera at the index {self.CAMERA_INDEX}")

        while cap.isOpened():
            ret, frame = cap.read()
            results = model(source=frame, verbose=verbose_model)

            if not ret:
                break

            counter = -1
            boxes_detected = {}
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
                    boxes_detected.update( { counter: box } )
                    detections_conf.append(float(box.conf))
                del(boxes)
            
            if has_detection:
                tv_entry.setBoolean(True)
                higher_value = max(detections_conf)
                higher_value_index = detections_conf.index(higher_value)

                box = boxes_detected[higher_value_index]

                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                vertical_pixel_offset = self.cam_x - cx
                horizontal_pixel_offset = self.cam_y - cy

                tx = calcs.get_vertical_angle_offset( vertical_pixel_offset, False )
                ty = calcs.get_horizontal_angle_offset( horizontal_pixel_offset, False )

                tx = round(tx, 2)
                ty = round(ty, 2)
                cls = int(box.cls[0])
                cls_name = self.objects[cls]

                tx_entry.setNumber(tx)
                ty_entry.setNumber(ty)
                cls_name_entry.setString(cls_name)
                
                print(f"Object Detected!")
                print(f"tx: {tx}")
                print(f"ty: {ty}")
                print(f"cls: {cls} / cls_name: {cls_name}")
                print()

                if show_image:
                    self.show_detected_image(
                        img=frame, tx=tx, ty=ty, cls=cls, box_x1=x1, box_y1=y1,
                        box_x2=x2, box_y2=y2, cx=cx, cy=cy, cam_x=self.cam_x, cam_y=self.cam_y
                    )

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
            else:
                tv_entry.setBoolean(False)
                print("No detections")
                print()
                if show_image:
                    self.show_not_detected_image( frame, self.cam_x, self.cam_y )

                if cv2.waitKey(1) & 0x0FF == ord('q'):
                        break

        cap.release()
        cv2.destroyAllWindows()

h = 1
w = 2
d = 2.25

Vision(h, w, d, model_path="./models/train3-INT8.onnx").main(True)
