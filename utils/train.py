import os
from typing import Union
from ultralytics import YOLO

StrOrBytesPath = Union[str, bytes, os.PathLike[str], os.PathLike[bytes]]

def train_model(
    base_model: Union[StrOrBytesPath, str], data_path: StrOrBytesPath, device: str, val: bool, 
    workers: Union[int, bool], pretrained: bool = False, single_cls: bool = False, epochs: int = 50, 
    patience: int = 8, batch: int = -1, imgsz: int = 640):
    """
    Train a YOLO model using Ultralytics and PyTorch.

    :param base_model: Path to the base YOLO model that will be trained.
    :param data_path: Path to the "data.yaml" file containing dataset information.
    :param device: Device to run the training on ("cpu", "gpu", or "ams").
    :param val: Whether to run validation after each epoch.
    :param workers: Number of worker threads for data loading. 
                    If True, uses all available CPU threads.
    :param pretrained: Whether to initialize the model with pretrained weights from `base_model`.
    :param single_cls: Treat the dataset as having a single class (all objects labeled as "item").
    :param epochs: Total number of training epochs.
    :param patience: Number of epochs without improvement before early stopping.
    :param batch: Batch size (-1 to let the trainer select automatically).
    :param imgsz: Input image size for training and inference.
    """
    
    WORKERS_NUMBER = int

    if type(workers) == bool:
        WORKERS_NUMBER = os.cpu_count()
    else:
        WORKERS_NUMBER = workers

    model = YOLO(base_model)
    model.train(
        imgsz=imgsz,
        epochs=epochs, 
        data=data_path,
        patience=patience,

        val=val,
        dfl=5, 
        box=7.5,
        cls=0.5,
        
        batch=batch,
        device=device,
        workers=WORKERS_NUMBER,
        pretrained=pretrained,
        single_cls=single_cls,

        degrees=0.3,
        hsv_v=0.3,
        scale=0.5,
        fliplr=0.5
    )