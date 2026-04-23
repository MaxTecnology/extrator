import cv2
import numpy as np
from PIL import Image


def estimate_skew_angle(gray_image: np.ndarray) -> float:
    if gray_image.ndim == 3:
        gray = cv2.cvtColor(gray_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = gray_image

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    min_line_length = max(30, gray.shape[1] // 6)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=70,
        minLineLength=min_line_length,
        maxLineGap=20,
    )
    if lines is None:
        return 0.0

    angles: list[float] = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = [int(v) for v in line]
        if x1 == x2 and y1 == y2:
            continue
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        while angle <= -90.0:
            angle += 180.0
        while angle > 90.0:
            angle -= 180.0
        if -45.0 <= angle <= 45.0:
            angles.append(angle)

    if len(angles) < 3:
        return 0.0
    return float(np.median(np.asarray(angles, dtype=np.float32)))


def _deskew(gray_image: np.ndarray) -> np.ndarray:
    angle = estimate_skew_angle(gray_image)
    abs_angle = abs(angle)
    if abs_angle <= 0.5 or abs_angle >= 15.0:
        return gray_image

    height, width = gray_image.shape[:2]
    center = (width / 2.0, height / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray_image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def preprocess_stamp(crop_image: Image.Image, min_width: int = 600) -> Image.Image:
    rgb = crop_image.convert("RGB")
    np_rgb = np.asarray(rgb)

    height, width = np_rgb.shape[:2]
    if width < min_width:
        scale = min_width / max(1.0, float(width))
        new_height = max(1, int(round(height * scale)))
        np_rgb = cv2.resize(
            np_rgb,
            (min_width, new_height),
            interpolation=cv2.INTER_CUBIC,
        )

    gray = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    deskewed = _deskew(enhanced)
    denoised = cv2.fastNlMeansDenoising(deskewed, None, h=10)
    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return Image.fromarray(otsu)
