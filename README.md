# YOLO Object Detection Training & Inference

This repository provides tools for training, converting, and deploying YOLO-based object detection models, with utilities for dataset preparation, model conversion, and real-time inference. It is designed for FRC (FIRST Robotics Competition) vision applications, but can be adapted for other use cases.

## Features

- **Dataset Generation:** Download, convert, and process videos/images for training.
- **Label Extraction:** Automatically label images using a YOLO model.
- **Model Training:** Train YOLO models with configurable parameters.
- **Model Conversion:** Convert PyTorch models to ONNX and quantize to INT8.
- **Real-Time Inference:** Run object detection and communicate results via NetworkTables.
- **Noise Augmentation:** Generate noisy/sharpened/blurred images for robust training.

## Directory Structure

```
.
├── data.yaml                # Dataset configuration for YOLO
├── models/                  # Trained and converted models (.pt, .onnx)
├── utils/                   # Utility scripts for dataset and model management
├── vision.py                # Main inference script
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── ...
```

## Getting Started

### 1. Requirements
Python version 3.11.9
```sh
pyenv install 3.11.9
pyenv local 3.11.9
```

(optional, but extremely recommended) download FFMPEG 
<details>
  <summary>Windows</summary>
  Step 1: Download the FFMPEG build from the official website https://www.ffmpeg.org/download.html#build-windows

  Step 2: Move the folder to your user directory

  ```sh
  mv C:/Users/%USERNAME%/Downloads/ffmpeg/ C:/users/%USERNAME%/
  ```

  Step 3: Add this to your environment variable
  1. Press the Windows Logo button and type "environment variables", and then press enter
  2. Click at the "Environment Variables..." button
  3. Double click at the "Path" variable, click "New" and put the path to your FFPMEG binary directory (C:/users/%USERNAME%/ffmpeg/bin)
</details>

<details>
  <summary>Linux</summary>
  
  #### Ubuntu & Debian:
  ```sh
  sudo apt install ffmpeg
  ```

  #### Arch:
  ```sh
  sudo pacman -S ffmpeg
  ```


</details>

### 2. Install Dependencies
```sh
pip install -r requirements.txt
```

The PyMovie dependencie is required to install manually, because it's have problems intalling via PyPi (pip)

1. Clone the repository of the MoviePy project
```sh
git clone https://github.com/Zulko/moviepy
cd moviepy
```
2. Move the moviepy folder inside of the cloned repository to 
```
./.venv/Lib/site-packages/ 
```


### 3. Prepare Dataset

Use the utilities in `utils/generate_datasets.py` to:
- Download and move videos
- Convert video/image formats
- Extract frames
- Generate noise/augmentations
- Extract YOLO labels

### 3. Configure Dataset

Edit `data.yaml` to point to your dataset folders and class names.

### 4. Train a Model

Use `utils/train.py`:

```python
from utils.train import train_model

train_model(
    base_model="yolov8n.pt",
    data_path="data.yaml",
    device="cpu",            # or "cuda" for GPU
    val=True,
    workers=True,
    pretrained=True,
    single_cls=False,
    epochs=50,
    patience=8,
    batch=16,
    imgsz=640
)
```

### 5. Convert & Quantize Model

Convert to ONNX and quantize to INT8 using `utils/convert_model.py`:

```python
from utils.convert_model import convert_to_ONNX, quantize_to_INT8

convert_to_ONNX("models/train3.pt")
quantize_to_INT8("models/train3.onnx", "models/train3-INT8.onnx")
```

### 6. Run Real-Time Inference

Edit parameters as needed and run `vision.py`:

```sh
python vision.py
```

## Customization

- **Camera and Model Settings:** Adjust in `vision.py` constructor.
- **NetworkTables:** Set `ROBORIO_IP` and `RASPBERRY_NAME` as needed environment variables for your FRC setup.

## Example:
See examples at the folder [examples](examples) file for a full example of how to use this

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

**Contributions and issues are welcome!**