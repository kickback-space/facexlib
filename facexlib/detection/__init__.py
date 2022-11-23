import torch

from facexlib.utils import load_file_from_url
from .retinaface import RetinaFace


def init_detection_model(model_name, half=False, device="cuda", model_rootpath=None):
    if model_name == "retinaface_resnet50":
        model = RetinaFace(model_rootpath, network_name="resnet50", half=half)
        model_url = "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth"
    elif model_name == "retinaface_mobile0.25":
        model = RetinaFace(model_rootpath, network_name="mobile0.25", half=half)
        model_url = "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_mobilenet0.25_Final.pth"
    else:
        raise NotImplementedError(f"{model_name} is not implemented.")

    return model
