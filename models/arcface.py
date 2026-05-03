from logging import getLogger

import cv2
import numpy as np
from onnxruntime import InferenceSession

from utils.helpers import face_alignment

__all__ = ['ArcFace']

logger = getLogger(__name__)

class ArcFace:
    """
    ArcFace model for Face Recognition

    This class implements a face encoder using the ArcFace Architecture.
    Load a pretrained model from an ONNX file.
    """
    def __init__(self, model_path: str) -> None:
        """
        Initializes the ArcFace face encoder model.

        Args:
            model_path (str): Path to ONNX model file.

        Raises:
            RuntimeError: If model initialization fails.
        """
        self.model_path = model_path
        self.input_size = (112, 112)
        self.normalization_mean = 127.5
        self.normalization_scale = 127.5

        logger.info(f"Initializing ArcFace model from {self.model_path}")

        # ONNX show timeee
        import onnxruntime
        
        available = onnxruntime.get_available_providers()
        logger.info(f"ArcFace onnxruntime module: {onnxruntime.__file__}")
        logger.info(f"ArcFace available providers: {available}")
        providers = []
        if 'CUDAExecutionProvider' in available:
            providers.append(('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kSameAsRequested', # for Memory Allocation, Extends the memory exactly by the requested amount.
                'cudnn_conv_algo_search': 'EXHAUSTIVE', # for convolution performance, find the fastest algorithm. It results in a slow first run but faster subsequent inferences.
            }))
        providers.append('CPUExecutionProvider')
        if 'CUDAExecutionProvider' not in available:
            logger.warning("CUDAExecutionProvider is not available for ArcFace. Inference will use CPU.")

        try:
            opts = onnxruntime.SessionOptions()
            opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = InferenceSession(
                self.model_path,
                sess_options=opts,
                providers=providers
            )

            input_config = self.session.get_inputs()[0]
            self.input_name = input_config.name

            input_shape = input_config.shape
            model_input_size = tuple(input_shape[2:4][::-1])
            if model_input_size != self.input_size:
                logger.warning(
                    f"Model input size {model_input_size} differs from configured size {self.input_size}"
                )

            self.output_names = [o.name for o in self.session.get_outputs()]
            self.output_shape = self.session.get_outputs()[0].shape
            self.embedding_size = self.output_shape[1]

            if len(self.output_names) != 1:
                raise ValueError(
                    f"Expected exactly one output node, got {len(self.output_names)}."
                )
            logger.info(
                f"Successfully initialized face encoder from {self.model_path} "
                f"(embedding size: {self.embedding_size})"
            )
            logger.info(f"ArcFace active providers: {self.session.get_providers()}")

        except Exception as e:
            logger.warning(f"Failed to load with optimal providers: {e}. Falling back to CPU...")
            try:
                self.session = InferenceSession(
                    self.model_path,
                    providers=["CPUExecutionProvider"]
                )
                input_config = self.session.get_inputs()[0]
                self.input_name = input_config.name
                self.output_names = [o.name for o in self.session.get_outputs()]
                self.output_shape = self.session.get_outputs()[0].shape
                self.embedding_size = self.output_shape[1]
                logger.info(f"ArcFace active providers: {self.session.get_providers()}")
            except Exception as e2:
                logger.error(f"Failed to load face encoder model from '{self.model_path}'", exc_info=True)
                raise RuntimeError(f"Failed to initialize model session for '{self.model_path}'") from e2

    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess: resize, normalize, cvt to required format

        Args:
            face_image: input in BGR format
        
        Returns:
            np.ndarray: preprocessed image, ready for inference
        """
        resized_face = cv2.resize(face_image, self.input_size)

        if isinstance(self.normalization_scale, (list, tuple)):
            # handle per-channel normalization
            rgb_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB).astype(np.float32)

            mean_array = np.array(self.normalization_mean, np.float32)
            scale_array = np.array(self.normalization_scale, np.float32)
            normalized_face = (rgb_face - mean_array) / scale_array

            # (H, W, C) -> (batch, C, H, W)
            transposed_face = np.transpose(normalized_face, (2, 0, 1)) # (C, H, W)
            face_blob = np.expand_dims(transposed_face, axis=0)
        else:
            # single-vale normalization using cv2.dnn
            # blob is a 4D tensor
            face_blob = cv2.dnn.blobFromImage(
                resized_face,
                scalefactor=1.0 / self.normalization_scale,
                size=self.input_size,
                mean=(self.normalization_mean,) * 3,
                swapRB=True
            )
            return face_blob
        
    def get_embedding(self, image: np.ndarray, landmarks: np.ndarray, normalized: bool = False) -> np.ndarray:
        """
        Extract face embed from image using facial landmarks for alignment
        
        Args:
            image: input image (BGR)
            landmarks: 5-point facial landmarks for alignment
            normalized: whether normalize output vector embedding. defaults to False
        
        Returns:
            np.ndarray: face embedding vector
        
        Raises:
            ValueError: if inputs ar invalid
        """
        if image is None or landmarks is None:
            raise ValueError("Image and Landmarks must not be None")
        
        try:
            aligned_face, _ = face_alignment(image, landmarks)
            face_blob = self.preprocess(aligned_face)
            embedding = self.session.run(self.output_names, {self.input_name: face_blob})[0]

            if normalized:
                # L2 Norm
                norm = np.linalg.norm(embedding, axis=1, keepdims=True)
                normalized_embedding = embedding / norm
                return embedding.flatten()
        except Exception as e:
            logger.error(f"Error extracting face embedding: {e}")
            raise
