from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image


BBoxTuple = tuple[int, int, int, int]


@dataclass(slots=True)
class DetectionResult:
    bbox: Optional[BBoxTuple]
    confidence: float
    found: bool
    reason: str
    message: str
    fallback_bbox: BBoxTuple


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return float(max(minimum, min(maximum, value)))


def _compute_lower_left_roi(
    width: int,
    height: int,
    y_start_ratio: float,
    x_end_ratio: float,
) -> tuple[int, int, int, int]:
    x_start = 0
    x_end = max(1, min(width, int(round(width * x_end_ratio))))
    y_start = max(0, min(height - 1, int(round(height * y_start_ratio))))
    y_end = height
    return x_start, y_start, x_end, y_end


def _pad_bbox(bbox: BBoxTuple, width: int, height: int, padding: int) -> BBoxTuple:
    x, y, w, h = bbox
    x_min = max(0, x - padding)
    y_min = max(0, y - padding)
    x_max = min(width, x + w + padding)
    y_max = min(height, y + h + padding)
    return x_min, y_min, x_max - x_min, y_max - y_min


def _score_candidate(
    contour: np.ndarray,
    roi_gray: np.ndarray,
    roi_width: int,
    roi_height: int,
) -> tuple[float, float, BBoxTuple]:
    x, y, w, h = cv2.boundingRect(contour)
    area = float(cv2.contourArea(contour))
    bbox_area = float(max(1, w * h))
    roi_area = float(max(1, roi_width * roi_height))

    crop_gray = roi_gray[y : y + h, x : x + w]
    dark_density = float(np.mean(crop_gray < 170)) if crop_gray.size else 0.0
    rectangularity = _clamp(area / bbox_area)
    area_norm = _clamp(area / (roi_area * 0.20))

    center_x = (x + (w / 2.0)) / max(1.0, float(roi_width))
    center_y = (y + (h / 2.0)) / max(1.0, float(roi_height))
    position_score = _clamp(((1.0 - center_x) + center_y) / 2.0)

    score = (
        (0.35 * area_norm)
        + (0.25 * dark_density)
        + (0.20 * position_score)
        + (0.20 * rectangularity)
    )
    confidence = _clamp(0.15 + (0.85 * score))
    return score, confidence, (x, y, w, h)


def _find_best_candidate_in_roi(
    roi_binary: np.ndarray,
    roi_gray: np.ndarray,
    min_contour_area: int,
) -> tuple[Optional[BBoxTuple], float]:
    roi_height, roi_width = roi_binary.shape[:2]
    contours, _ = cv2.findContours(
        roi_binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    best_score = -1.0
    best_confidence = 0.0
    best_bbox_local: Optional[BBoxTuple] = None

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area <= min_contour_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if h <= 0:
            continue

        aspect_ratio = w / h
        if aspect_ratio < 0.5 or aspect_ratio > 5.0:
            continue

        score, confidence, bbox_local = _score_candidate(
            contour=contour,
            roi_gray=roi_gray,
            roi_width=roi_width,
            roi_height=roi_height,
        )
        if score > best_score:
            best_score = score
            best_confidence = confidence
            best_bbox_local = bbox_local

    return best_bbox_local, best_confidence


def detect_stamp_region(
    image: Image.Image,
    min_contour_area: int = 4000,
    padding_px: int = 20,
    fallback_roi_y_start: float = 0.55,
    fallback_roi_x_end: float = 0.55,
) -> DetectionResult:
    rgb = image.convert("RGB")
    np_rgb = np.asarray(rgb)

    gray = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    binary_inv = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )

    height, width = gray.shape[:2]
    base_x0, base_y0, base_x1, base_y1 = _compute_lower_left_roi(
        width,
        height,
        fallback_roi_y_start,
        fallback_roi_x_end,
    )
    fallback_bbox = (base_x0, base_y0, base_x1 - base_x0, base_y1 - base_y0)
    base_roi_gray = gray[base_y0:base_y1, base_x0:base_x1]

    # Mantém a ROI principal (55%+) e expande para faixas superiores
    # quando necessário, sem alterar o fallback oficial da API.
    roi_y_starts = [fallback_roi_y_start]
    for extra_y_start in (0.45, 0.40):
        if extra_y_start < fallback_roi_y_start:
            roi_y_starts.append(extra_y_start)

    for roi_y_start in roi_y_starts:
        roi_x0, roi_y0, roi_x1, roi_y1 = _compute_lower_left_roi(
            width=width,
            height=height,
            y_start_ratio=roi_y_start,
            x_end_ratio=fallback_roi_x_end,
        )
        roi_binary = binary_inv[roi_y0:roi_y1, roi_x0:roi_x1]
        roi_gray = gray[roi_y0:roi_y1, roi_x0:roi_x1]
        if roi_binary.size == 0 or roi_gray.size == 0:
            continue

        best_bbox_local, best_confidence = _find_best_candidate_in_roi(
            roi_binary=roi_binary,
            roi_gray=roi_gray,
            min_contour_area=min_contour_area,
        )
        if best_bbox_local is None:
            continue

        local_x, local_y, local_w, local_h = best_bbox_local
        global_bbox = (
            roi_x0 + local_x,
            roi_y0 + local_y,
            local_w,
            local_h,
        )
        padded_bbox = _pad_bbox(global_bbox, width=width, height=height, padding=padding_px)

        return DetectionResult(
            bbox=padded_bbox,
            confidence=_clamp(best_confidence),
            found=True,
            reason="detectado_por_contorno",
            message="Carimbo detectado por contorno",
            fallback_bbox=fallback_bbox,
        )

    fallback_dark_density = float(np.mean(base_roi_gray < 170)) if base_roi_gray.size else 0.0
    fallback_confidence = min(0.45, max(0.10, 0.18 + (0.25 * fallback_dark_density)))
    return DetectionResult(
        bbox=None,
        confidence=_clamp(fallback_confidence),
        found=False,
        reason="fallback_regiao_inferior",
        message=(
            "Carimbo não detectado. Retornando região de fallback "
            "(quadrante inferior esquerdo)"
        ),
        fallback_bbox=fallback_bbox,
    )
