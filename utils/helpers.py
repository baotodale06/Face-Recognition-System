from typing import Optional, Tuple

import cv2
import numpy as np

from skimage.transform import SimilarityTransform

# reference alignment for facial landmarks
reference_alignment: np.ndarray = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ],
    dtype=np.float32
)

def estimate_norm(landmark: np.ndarray, image_size: int = 112) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the normalization transformation matrix for facial landmarks.

    Args:
        landmark (np.ndarray): Array of shape (5, 2) representing the coordinates of the facial landmarks.
        image_size (int, optional): The size of the output image. Default is 112.

    Returns:
        np.ndarray: The 2x3 transformation matrix for aligning the landmarks.
        np.ndarray: The 2x3 inverse transformation matrix for aligning the landmarks.

    Raises:
        ValueError: If the input landmark array does not have the shape (5, 2)
                    or if image_size is not a multiple of 112 or 128.
    """
    if landmark.shape != (5, 2):
        raise ValueError(f"Landmark array must have shape (5, 2), got {landmark.shape}!")

    if image_size % 112 != 0 and image_size % 128 != 0:
        raise ValueError(f"Image size must be a multiple of 112 or 128, got {image_size}!")
    
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0.0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    
    # adjust ref alignment based on ratio and diff_x
    alignment = reference_alignment *  ratio
    alignment[:,0] += diff_x

    if hasattr(SimilarityTransform, "from_estimate"):
        transform = SimilarityTransform.from_estimate(landmark, alignment)
        if not transform:
            raise ValueError("Failed to estimate similarity transform from landmarks!")
    else:
        transform = SimilarityTransform()
        transform.estimate(landmark, alignment)

    matrix = transform.params[0:2, :]
    inverse_matrix = np.linalg.inv(transform.params)[0:2, :]

    return matrix, inverse_matrix

def face_alignment(image: np.ndarray, landmark: np.ndarray, image_size: int = 112) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align the face in the input image based on the given facial landmarks.

    Arg:
        image (np.ndarray): input image
        landmark (np.ndarray): array of shape (5, 2) representing the coordinates of the facial landmarks
        image_size (int): size of the aligned output image, default is 112
    Returns:
        warped (np.ndarray): The aligned face as a np array
        M_inv (np.ndarray): the 2x3 transformation matrix used for alignment
    """
    # get the transformation matix
    M, M_inv = estimate_norm(landmark, image_size)

    # warp the input image to align the face
    warped = cv2.warpAffine(image, M, (image_size, image_size), borderValue=0.0)

    return warped, M_inv

def distance2bbox(points: np.ndarray,
                  distance: np.ndarray,
                  max_shape: Optional[Tuple[int, int]]) -> np.array:
    """
    Decode dinstance prediciton to bbox

    Args:
        points (np.ndarray): shape (n, 2), [x, y]
        distance: distance from the given point to 4 boundaries (l, t, r, b)
        max_shape: shape of the image as (height, width)
    
    Returns:
        np.ndarray: decoded bbox with shape (n, 4)
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    
    return np.stack([x1, y1, x2, y2], axis = -1)

def distance2kps(points: np.ndarray,
                 distance: np.ndarray,
                 max_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Decode dinstance prediction to keypoints

    Args:
        points: shape (n, 2) [x, y]
        distance: distance fromt the given point to keypoint offsets
        max_shape: shape of the image as (height, width)
    
    Returns:
        decoded keypoints with shape (n, 2k)
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = np.clip(px, 0, max_shape[1])
            py = np.clip(py, 0, max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

def compute_similarity(feat1: np.ndarray, feat2: np.ndarray) -> np.float32:
    """
    Compute similarity between 2 faces

    Args:
        feat1: 1st face feats
        feat2: 2nd face feats

    Returns:
        np.float32: Cosine Sim between 2 face feats
    """
    feat1 = feat1.ravel() #flatten a multi-dimensional array into a single 1D array
    feat2 = feat2.ravel()
    
    cosine_sim = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))

def draw_bbox(image: np.ndarray, bbox: list[int], color: Tuple[int, int, int] = (0,255,0), 
              thickness: int = 3, proportion: float = 0.2) -> None:
    """
    Draw a bbx with corner accents on the image

    Args:
        image: frame to draw on
        bbox: bbox coordinates [x1, y1, x2, y2]
        color: BGR color
        thickness: corner line thickness
        proportion: corner accent length as fraction of the shorter bbox side
    """
    x1, y1, x2, y2 = map(int, bbox)
    width = abs(x2 - x1)
    height = abs(y2 - y1)

    corner_length = int(proportion * min(width, height))

    # draw rectangle
    cv2.rectangle(image, (x1,y1), (x2,y2), color, 1)

    # top-left corner
    cv2.line(image, (x1, y1), (x1 + corner_length, y1), color, thickness)
    cv2.line(image, (x1, y1), (x1, corner_length + y1), color, thickness)

    # top-right
    cv2.line(image, (x2, y1), (x2 - corner_length, y1), color, thickness)
    cv2.line(image, (x2, y1), (x1, corner_length + y1), color, thickness)

    # bot-left
    cv2.line(image, (x1, y2), (x1, y2 -  corner_length), color, thickness)
    cv2.line(image, (x1, y2), (x1 + corner_length, y2), color, thickness)

    # bot-right
    cv2.line(image, (x2, y2), (x2, y2 - corner_length), color, thickness)
    cv2.line(image, (x2, y2), (x2 - corner_length, y2), color, thickness)

def draw_bbox_info(frame: np.ndarray, bbox: list[int], similarity: float, name: str, color: Tuple[int, int, int]) -> None:
    """
    Draw bbox with identity label and similarity bar

    Args:
        frame: frame to draw on
        bbox: bbox coordinates [x1, y1, x2, y2]
        similarity: consine similarity score
        name: Identity label of the person
        color: BGR color
    """
    x1, y1, x2, y2 = map(int, bbox)

    # keep text label within frame bounds
    text_y = max(y1 - 10, 15) # at least 15 units down from the upper side
    cv2.putText(
        frame,
        f"{name}: {similarity:.2f}",
        org=(x1, text_y),
        fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL,
        fontScale=1,
        color=color,
        thickness=1
    )

    # draw bbox
    draw_bbox(frame, bbox, color)

    # draw similarity bar (clampt to [0, 1] to avoid negative height)
    clamped_sim = float(np.clip(similarity, 0.0, 1.0))
    rect_x_start = x2+10
    rect_x_end = rect_x_start + 10
    rect_y_end = y2
    rect_height = int(clamped_sim * (y2-y1))
    rect_y_start = rect_y_end - rect_height

    # draw the filled rectangle
    cv2.rectangle(frame, (rect_x_start, rect_y_start), (rect_x_end, rect_y_end), color, cv2.FILLED)