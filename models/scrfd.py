import logging
from typing import Tuple

import cv2
import numpy as np
import onnxruntime

from utils.helpers import distance2bbox, distance2kps

__all__ = ["SCRFD"]

logger = logging.getLogger(__name__)

class SCRFD:
    """
    Model from a paper named: "Sample & Compuation Redistribution for Efficient Face Detection
    """
    def __init__(self, model_path: str, input_size: Tuple[int, int] = (640, 640),
                 conf_thresh: float = 0.5, iou_thresh: float = 0.4) -> None:
        """
        SCRFD model initialization

        Args:
            model_path: path to a .onnx model file
            input_size: input image size as (width, height), default (640, 640)
            conf_thresh: confidence threshold, default 0.5
            iou_thresh: Non-max Suppression threshold, default 0.4
        """

        self.input_size = input_size
        self.conf_thres = conf_thresh
        self.iou_thres = iou_thresh

        # SCRFD model params --------
        self.fmc = 3 # 3 output feature maps
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.use_kps = True

        self.mean = 127.5
        self.std = 128.0

        self.center_cache = {}
        # ---------------------------

    def _initialize_model(self, model_path: str, providers: list = None) -> None:
        """
        Initialize ONNX inference session

        Args:
            model_path: path to a .onnx model
            providers: list of execution providers, default to None.
        """
        import onnxruntime
        if providers is None:
            available = onnxruntime.get_available_providers()
            logger.info(f"SCRFD onnxruntime module: {onnxruntime.__file__}")
            logger.info(f"SCRFD available providers: {available}")
            providers = []
            if 'CUDAExecutionProvider' in available:
                providers.append(('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kSameAsRequested',
                    'cudnn_conv_algo_search': 'EXHAUSTIVE'
                }))
            providers.append('CPUExecutionProvider')
            if 'CUDAExecutionProvider' not in available:
                logger.warning(f"CUDAExecutionProvider is not available for SCRFD, inference will use CPU instead")

        try:
            options = onnxruntime.SessionOptions() 
            options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = onnxruntime.InferenceSession(
                model_path,
                sess_options=options,
                providers=providers
            )
            # extract model info
            self.output_names = [x.name for x in self.session.get_outputs()]
            self.input_names = [x.name for x in self.session.get_inputs()]
            logger.info(f"Successfully load SCRFD model from {model_path}")
            logger.info(f"SCRFD active providers: {self.session.get_providers()}")

        except Exception as e:
            logger.warning(f"Fail to load the model with optimal providers: {e}. Fall back to CPU")
            self.session = onnxruntime.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider'],
            )
            self.output_names = [x.name for x in self.session.get_outputs()]
            self.input_names = [x.name for x in self.session.get_inputs()]
            logger.info(f"Successfully load SCRFD model from {model_path}")
            logger.info(f"SCRFD active providers: {self.session.get_providers()}")

    def forward(self, image: np.ndarray, threshold: float) -> Tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """
        Run a forward pass through the model

        Args:
            image: already preprocessed input image
            threshold: score threshold for FILTERING detections

        Returns:
            Tuple of (scores_list, bboxes_list, kpss_list) per FPN stride
        """
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(image.shape[0:2][::-1]) # (H, W, C) --[0:2]--> (H, W) --[::-1]-> (W, H)

        # (1, 3, H, W)
        blob = cv2.dnn.blobFromImage(
            image,
            1.0 / self.std,
            input_size,
            (self.mean, self.mean, self.mean),
            swapRB=True,
        ) 

        outputs = self.session.run(self.output_names, {self.input_names[0]: blob})
        # output is a list of tensors, containing [3 scores, 3 bboxes, 3 kpss]
        # 3 because feature_map_count = 3

        input_height = blob.shape[2]
        input_width = blob.shape[3]

        fmc = self.fmc
        for id, stride in enumerate(self._feat_stride_fpn):
            scores = outputs[id] # [0, 1, 2]
            bbox_preds = outputs[id + fmc] # [3, 4, 5] = [0+3, 1+3, 2+ 3]
            bbox_preds = bbox_preds * stride # model pred relative distance -> scale to real size
            if self.use_kps:
                kps_preds = outputs[id + fmc*2] * stride # [6, 7, 8]; scale to real size
            
            # size of the feature map
            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)

            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                # create a grid (x, y) positions for every cell
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                
                # convert from a grid to actual pixel coordinate of the image by multiply stride
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                
                # each location has multiple anchors -> duplicate
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers
            

            pos_inds = np.where(scores >= threshold)[0]
            # shape (N, 4)
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss) 
        
        # Each is a list of 3 arrays (one per stride)
        return scores_list, bboxes_list, kpss_list
    
    def detect(self, image: np.ndarray, max_num: int = 0, metric: str = "max") -> Tuple[np.ndarray, np.ndarray | None]:
        """
        Detect faces in an image

        Args:
            image: input BGR image
            max_num: maximum detections to return (0 is no limit)
            metric: selection metric when capping detections
        
        Returns:
            Tuple of (detections, keypoints) where detections has shape (N,5)
            with columns [x1, y1, x2, y2, score] and keypoints has shape (N, 5, 2)
        """
        width, height = self.input_size

        im_ratio = float(image.shape[0]) / image.shape[1]
        model_ratio = height / width
        
        # scale the image depends on the larger side
        if im_ratio > model_ratio:
            new_height = height
            new_width = int(new_height / im_ratio)
        else:
            new_width = width
            new_height = int(new_width * im_ratio)
        

        det_scale = float(new_height) / image.shape[0]
        resized_image = cv2.resize(image, (new_width, new_height))

        # pad image to model's input size
        det_image = np.zeros(shape=(height, width, 3), dtype=np.uint8)
        det_image[:new_height, :new_width, :] = resized_image

        # run forward
        scores_list, bboxex_list, kpss_list = self.forward(det_image, self.conf_thres)
        
        # merge outputs altogether
        scores = np.vstack(scores_list)
        # flatten
        scores_ravel = scores.ravel()
        # indices sorted desc
        order = scores_ravel.argsort()[::-1]

        # Combine + SCALE back to original image
        bboxes = np.vstack(bboxex_list) / det_scale

        # apply to keypoints as well
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale

        # shape (N, 5) ~ [x1, y1, x2, y2, score]
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        # sort based on score
        pre_det = pre_det[order, :]

        # rm overlapping boxes
        keep = self.nms(pre_det, iou_thres=self.iou_thres)
        det = pre_det[keep, :]

        # apply same indices for keypoints
        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None

        # if exceeds max, select the best ones
        if 0 < max_num < det.shape[0]:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            image_center = image.shape[0] // 2, image.shape[1] // 2
            
            # distance from image center
            offsets = np.vstack(
                [
                    (det[:, 0] + det[:, 2]) / 2 - image_center[1],
                    (det[:, 1] + det[:, 3]) / 2 -image_center[0],
                ]
            )   
            offset_dist_squared = np.sum(np.power(offsets, 2.0), axis=0)

            # you can decide what is a GOOD bbos: 1. larger, or 2. closer to the center
            if metric == 'max':
                values = area
            else:
                values = (area - offset_dist_squared * 2.0) # more extra weight on the centering

            # sort desc based on that value
            bindex = np.argsort(values)[::-1]
            
            # select top
            bindex = bindex[0:max_num]
            det = det[bindex, :]

            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss

    def nms(self, dets: np.ndarray, iou_thres: float) -> list[int]:
        """
        Greedy Non-Maximum Suppression

        Args:
            dets: Detections array of shape (N, 5) with columns [x1, y1, x2, y2, score]
            iou_thres: IoU threshold above which overlapping bboxes are suppressed
        
        Returns:
            List of kept detection indices
        """
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 +1)
        # order is sorted scores desc
        order = scores.argsort()[::-1]

        keep = []
        # loop until no boxes left
        while order.size > 0:
            # index of the highest-score box
            i = order[0]
            # pick/keep it
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            # compute intersection
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w*h
            # compute IoU
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            indices = np.where(ovr <= iou_thres)[0]
            order = order[indices+1]
        return keep
if __name__ == "__main__":
    from utils.helpers import draw_bbox

    detector = SCRFD(model_path="./weights/det_10.onnx")
    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            bboxes, kpss = detector.detect(frame)

            for bbox in bboxes:
                draw_bbox(frame, bbox[:4].astype(np.int32))
            
            cv2.imshow("FaceDetection", frame)
            # is press 'q' -> exit loop
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()