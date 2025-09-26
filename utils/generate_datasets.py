import os
import cv2
import yt_dlp
import shutil
import numpy as np
from PIL import Image
from time import sleep
from sys import platform
from typing import Union
from ultralytics import YOLO
from moviepy import VideoFileClip

OS_TYPE = platform

if OS_TYPE == "linux":
    CLEAR_COMMAND = "clear"
else:
    CLEAR_COMMAND = "cls"

StrOrBytesPath = Union[str, bytes, os.PathLike[str], os.PathLike[bytes]]

class Videos:
    def __init__(self):
        """
        This class downloads, and manipulate videos in your filesystem
        """

    def download_video(self, url: str):
        """
        Download a single video from youtube

        :param url: The URL of the video
        """

        ydl_opts = {
            'format': 'bestvideo+bestaudio[ext=m4a]/best',
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }],
            'merge_output_format': 'mp4',
            'outtmpl': '%(title)s.%(ext)s',
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    
    def download_videos(self, videos: list[str]) -> None:
        """
        Download several videos from youtube

        :param videos: A list that contains all the videos that will be downloaded
        """
        try:
            for x in range(len(videos)):
                print(f"Downloading the video: {x+1}")
                self.download_video(videos[x])
                os.system('cls')

            print("ALl videos download with succes!")
        
        except Exception as e:
            print("\033[31mError downloading the video,\033[0m {}".format(e))

    def move_videos(self, path: StrOrBytesPath, path_to_save: StrOrBytesPath) -> None:
        """
        Move all the videos of a path to another path

        Works with this videos extensions:
            .mp4,
            .mov,
            .avi,
            .mkv,
            .flv,
            .webm,
            .wmv
        
        :param path: The path that includes the videos
        :param path_to_save: The path that the videos that will be saved
        """

        EXTENSIONS = [
            ".mp4",
            ".mov",
            ".avi",
            ".mkv",
            ".flv",
            ".webm",
            ".wmv"
        ]

        archives = os.listdir(path)
        videos_to_move = []

        if not os.path.exists(path_to_save):
            os.mkdir(path_to_save)

        for archive in archives:
            if os.path.splitext(archive)[1] in EXTENSIONS:
                videos_to_move.append(archive)
        
        for video in videos_to_move:
            print("moving videos")
            shutil.move(video, path_to_save)
        print("all videos moved")

class Generate:
    def __init__(self, model_path: StrOrBytesPath, videos_extension: str = ".mp4", images_extension: str = ".jpg",):
        self.model = YOLO(model_path)
        self.videos_extension = videos_extension
        self.images_extensions = images_extension

    def convert_videos(self, videos_path: Union[StrOrBytesPath, list[StrOrBytesPath]], delete: bool = False):
        """
        Convert all videos to the specified extensions in the class creation

        Suported formats:
        .mp4, .mov, .avi, .mkv, .flw, .webm, .wmv
        """

        CODECS = {
            ".mp4":  "libx264",
            ".mov":  "libx264",
            ".avi":  "mpeg4",
            ".mkv":  "libx264",
            ".flv":  "flv",
            ".webm": "libvpx",
            ".wmv":  "wmv2"
        }

        if type(videos_path) == list:
            for video_path in videos_path:
                videos = os.listdir(video_path)
                for video in videos:
                    if video.endswith(self.videos_extension):
                        continue
                    
                    video_name = video_path + "/" + os.path.splitext(video)[0] + self.videos_extension
                    clip = VideoFileClip(video_path + "/" + video)
                    clip.write_videofile(video_name, codec=CODECS[self.videos_extension])    
                    clip.close()

                    if delete:
                        os.remove(video_path + "/" + video)

        elif type(videos_path) == str:
            videos = os.listdir(videos_path)
            for video in videos:
                if video.endswith(self.videos_extension):
                    continue

                video_name = video_path + "/" + os.path.splitext(video)[0] + self.videos_extension
                clip = VideoFileClip(video_path + "/" + video)
                clip.write_videofile(video_name, codec=CODECS[self.videos_extension])    
                clip.close()

                if delete:
                    os.remove(videos_path + "/" + video)
                
    def convert_images(self, images_path: Union[str, list[str]], delete:bool = False):
        """
        Convert all images to the specified extensions in the class creation
        """
        # IMAGE_PATH = "C:\\Users\\Arthur\\source\\repos\\FRC-Programas\\object-detection\\build-season-volley-project\\images.jpeg"
        # OUTPUT_PATH = "C:\\Users\\Arthur\\source\\repos\\FRC-Programas\\object-detection\\build-season-volley-project\\images.png"
        # img = Image.open(IMAGE_PATH)
        # img.save(OUTPUT_PATH, "PNG")

        if type(images_path) == list:
            for image_path in images_path:
                for image in image_path:
                    images = os.listdir(image_path)
                    extension_to_save = self.images_extensions.removeprefix(".")

                    for image in images:
                        if image.endswith(self.images_extensions):
                            continue

                        current_image_path = image_path + "/" + image
                        image_to_save_path = image_path + "/" + os.path.splitext(image)[0] + self.images_extensions
                        img = Image.open(current_image_path)
                        img.save(image_to_save_path, extension_to_save)

                        if delete:
                            os.remove(current_image_path)


        elif type(images_path) == str:
            images = os.listdir(images_path)
            extension_to_save = self.images_extensions.removeprefix(".")

            for image in images:
                if image.endswith(self.images_extensions):
                    continue

                current_image_path = images_path + "/" + image
                image_to_save_path = images_path + "/" + os.path.splitext(image)[0] + self.images_extensions
                img = Image.open(current_image_path)
                img.save(image_to_save_path, extension_to_save)

                if delete:
                    os.remove(current_image_path)

    def extract_frames_with_pause(self, videos_path: StrOrBytesPath, start_counter=0):
        """
        Separete the frames of a video with a pause

        Args:
        - videos_path: the path containing the videos to separete the frames
        - start_counter: the start of counter of videos name
        """
        
        videos = os.listdir(videos_path)
        
        # Select only videos on the directory
        for archive in videos:
            if not archive.endswith(self.videos_extension):
                videos.remove(archive)

        # Rename the videos 
        counter = start_counter    
        for video in videos:
            counter += 1
            os.rename(f"{videos_path}/{video}", f"{videos_path}/video-{counter}" + self.videos_extension)

        counter = start_counter
        videos = os.listdir(videos_path)

        # Select only videos on the directory
        for archive in videos:
            if not archive.endswith(self.videos_extension):
                videos.remove(archive)

        # Extract the frames
        for video in videos:
            # Create a folder for the extracted frames
            print(f"Extracting the frames of: {video}")
            current_video_frames_path = f"{videos_path}/{video.removesuffix(self.videos_extension)}-frames"

            if not os.path.exists(current_video_frames_path):
                os.mkdir(current_video_frames_path)

            counter += 1
            current_frame = 0
            cap = cv2.VideoCapture(f"{videos_path}/{video}")
            fps = cap.get(cv2.CAP_PROP_FPS)

            
            while cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame * fps)
                ret, frame = cap.read()
                if not ret:
                    break

                if current_frame == 0:
                    print(f"Total frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")                   
                    sleep(2)        
                
                print(f"Current frame: {current_frame}")
                path = current_video_frames_path + '/' + f'video{counter}-' + 'frame' + str(current_frame) + self.images_extensions
                cv2.imwrite(path, frame)

                current_frame += 1            
            os.system(CLEAR_COMMAND)

    def extract_frames_consecutive(self, videos_path: str):
        """
        Separete the frames of a video in a consecutive form

        Args:
        - videos_path: the path containing the videos to separete the frames
        """
        
        videos = os.listdir(videos_path)

        # Select only videos on the directory
        for archive in videos:
            if not archive.endswith(self.videos_extension):
                videos.remove(archive)

        # Rename the videos 
        counter = 0       
        for video in videos:
            counter += 1
            os.rename(f"{videos_path}/{video}", f"{videos_path}/video-{counter}" + self.videos_extension)

        counter = 0
        videos = os.listdir(videos_path)

        # Select only videos on the directory
        for archive in videos:
            if not archive.endswith(self.videos_extension):
                videos.remove(archive)

        # Extract the frames
        for video in videos:
            # Create a folder for the extracted frames
            print(f"Extracting the frames of: {video}")
            current_video_frames_path = f"{videos_path}/{video.removesuffix(self.videos_extension)}-frames"

            if not os.path.exists(current_video_frames_path):
                os.mkdir(current_video_frames_path)

            counter += 1
            current_frame = 0
            cap = cv2.VideoCapture(f"{videos_path}/{video}")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if current_frame == 0:
                    print(f"Total frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
                    sleep(2)
                
                print(f"Current frame: {current_frame}")
                path = current_video_frames_path + '/' + f'video{counter}-' + 'frame' + str(current_frame) + self.images_extensions
                cv2.imwrite(path, frame) 

                current_frame += 1                        

        os.system(CLEAR_COMMAND)
    
    def move_images(self, images_path: Union[str, list[str]], path_to_save:str):
        images_path_type = type(images_path)

        if not os.path.exists(path_to_save):
            os.mkdir(path_to_save)

        if images_path_type == list:
            for path in images_path:
                print(f"Moving image: {path + '/' + image}")
                images = []
                archives = os.listdir(path)

                for archive in archives:
                    if archive.endswith(self.images_extensions):
                        images.append(archive)
                del(archives)

                for image in images:
                    shutil.move(path + '/' + image, path_to_save)

        else:
            archives = os.listdir(images_path)
            images = []

            for archive in archives:
                if archive.endswith(self.images_extensions):
                    images.append(archive)
            del(archives)

            for image in images:
                print(f"Moving image {images_path + '/' + image}")
                shutil.move(images_path + '/' + image, path_to_save)
        
        os.system(CLEAR_COMMAND)

    def generate_noise_in_images(self, images_path: StrOrBytesPath, path_to_save= ""):
        """
        Fill blank the path_to_save to save in the same path of the images
        """

        if path_to_save == "":
            path_to_save = images_path

        kernel1 = np.array([
                [0, -1,  0],
                [-1,  5, -1],
                [0, -1,  0]
            ])

        images = os.listdir(images_path)

        for image in images:
            print(f"Generating noise in inmage: {images_path + '/' + image}")
            img = cv2.imread(images_path + "/" + image)

            # Low quality images
            size1 = [
                int(img.shape[1] * 0.2),
                int(img.shape[0] * 0.2)
            ]

            low_quality1 = cv2.resize(img, size1, interpolation=cv2.INTER_AREA)

            # Sharpned images
            sharp_img_kernel1 = cv2.filter2D(src=low_quality1, ddepth=-1, kernel=kernel1)

            # Noisy image
            noise = np.random.randint(0, 100, low_quality1.shape, dtype="uint8")
            noisy_img = cv2.add(sharp_img_kernel1, noise)

            # Blurred image
            blurred = cv2.GaussianBlur(noisy_img, (5, 5), 0)

            cv2.imwrite(path_to_save + '/' + image.removesuffix(self.images_extensions) + '-blurred' + self.images_extensions, blurred)
            cv2.imwrite(path_to_save + '/' + image.removesuffix(self.images_extensions) + '-noisy' + self.images_extensions, noisy_img)

            cv2.imwrite(path_to_save + '/' + image.removesuffix(self.images_extensions) + '-sharp-img1' + self.images_extensions, sharp_img_kernel1)
            cv2.imwrite(path_to_save + '/' + image.removesuffix(self.images_extensions) + '-low-quality1' + self.images_extensions, low_quality1, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        os.system(CLEAR_COMMAND)

    def extract_labels(self, images_path: StrOrBytesPath, identified_images_path:str,
                       labels_path: StrOrBytesPath, min_conf: float = 0.6, view_detection: bool = False):
        IMAGES = os.listdir(images_path)
        RECTANGLE_COLOR = (0, 255, 0)
        RECTANGLE_THICKNESS = 4

        if not os.path.exists(labels_path):
            os.mkdir(labels_path)

        if not os.path.exists(identified_images_path):
            os.mkdir(identified_images_path)

        for image in IMAGES:
            img = cv2.imread(images_path + "/" + image)
            h, w = img.shape[:2]
            result = self.model(img, verbose=False, show=False)


            print(f"Extracting labels of: {images_path + '/' + image}")

            for r in result:
                boxes = r.boxes

                if boxes is None or len(result[0].boxes) == 0:
                    print("No detections")
                    continue

                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    if conf <= min_conf:
                        print("No detections")
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h

                    label_name = labels_path + "/" + image.removesuffix(self.images_extensions) + ".txt"
                    label_content = f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

                    if view_detection:
                        cv2.rectangle(
                            img,
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)), 
                            RECTANGLE_COLOR, 
                            RECTANGLE_THICKNESS
                        )
                        
                        cv2.imshow("resultado", img)
                        key = cv2.waitKey(0) & 0xFF

                        if key == ord("y"):
                            print("Label extracted")
                            shutil.copy(images_path + "/" + image, identified_images_path)
                            with open(label_name, "w+") as file:
                                file.write(label_content)
                            cv2.destroyAllWindows()
                    else:
                        shutil.copy(images_path + "/" + image, identified_images_path)
                        with open(label_name, "w+") as file:
                            file.write(label_content)


        print("All labels extracted")
        sleep(.5)
        os.system(CLEAR_COMMAND)
