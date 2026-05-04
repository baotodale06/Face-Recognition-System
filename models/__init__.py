from pathlib import Path
from typing import Protocol, Tuple

import numpy as np

from .arcface import ArcFace
from .scrfd import SCRFD
from .yolo_face import YOLOFace


__all__ = ["ArcFace", "SCRFD", "YOLOFace", "FaceDetector", "create_detector"]

class FaceDetector(Protocol):
    """
    Any object is valid if it has these attributes/methods 
    I dont care what class it is
    -> Attribute: conf_thres
    -> Method: detect
    """
    conf_thres: float

    def detect(self, image: np.ndarray, 
               max_num: int = 0, 
               metric: str = "max") -> Tuple[np.ndarray, np.ndarray]:
        ... # no implementation, just a placeholder

def create_detector(model_path: str,
                    input_size: Tuple[int, int] = (640, 640),
                    conf_thres: float = 0.5,
                    iou_thres: float = 0.4) -> FaceDetector:
    model_name = Path(model_path).name.lower()
    if 'yolo' in model_name and 'face' in model_name:
        return YOLOFace(model_path=model_path,
                        input_size=input_size,
                        conf_thres=conf_thres,
                        iou_thres=iou_thres)
    
    return SCRFD(model_path=model_path,
                 input_size=input_size,
                 conf_thresh=conf_thres,
                 iou_thresh=iou_thres,)
