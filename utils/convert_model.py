import os
from typing import Union
from ultralytics import YOLO
from onnxruntime.quantization import quantize_dynamic, QuantType

StrOrBytesPath = Union[str, bytes, os.PathLike[str], os.PathLike[bytes]]

def convert_to_ONNX(pytorch_model_path: StrOrBytesPath):
    """
    Convert a PyTorch (.pt) model to a ONNX (.onnx) model

    :param pytorch_model_path: The path to the model that be converted
    """

    model = YOLO(pytorch_model_path)
    model.export(
        format="onnx",
        opset=12
    )

def quantize_to_INT8(onnx_model_path: StrOrBytesPath, path_to_save: StrOrBytesPath):
    """
    Quantize a ONNX model that have the weigths in FLOAT32 to INT8

    :param onnx_model_path: The path to the FLOAT32 model
    :param path_to_save: The path to save the INT8 model
    """

    quantize_dynamic(
        model_input=onnx_model_path,
        model_output=path_to_save,
        weight_type=QuantType.QUInt8
    )
