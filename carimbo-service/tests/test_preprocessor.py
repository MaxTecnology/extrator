import numpy as np
from PIL import Image, ImageDraw

from app.services.preprocessor import estimate_skew_angle, preprocess_stamp


def _make_small_sample(width: int = 320, height: int = 160) -> Image.Image:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 50, width - 20, 100), outline=(20, 20, 20), width=3)
    draw.line((25, 75, width - 25, 75), fill=(10, 10, 10), width=2)
    return image


def _make_skewed_sample(angle: float = 5.0) -> Image.Image:
    image = Image.new("RGB", (900, 360), "white")
    draw = ImageDraw.Draw(image)

    for y in range(80, 300, 40):
        draw.line((90, y, 810, y), fill=(20, 20, 20), width=4)

    draw.rectangle((120, 105, 360, 245), outline=(20, 20, 20), width=5)
    draw.rectangle((420, 115, 760, 255), outline=(20, 20, 20), width=5)

    return image.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor="white")


def test_upscale_when_image_width_is_less_than_600() -> None:
    image = _make_small_sample()

    processed = preprocess_stamp(image)

    assert processed.width >= 600
    assert processed.mode == "L"


def test_deskew_corrects_image_to_near_zero() -> None:
    image = _make_skewed_sample(angle=5.0)
    before_angle = estimate_skew_angle(np.array(image.convert("L")))
    assert abs(before_angle) >= 3.0

    processed = preprocess_stamp(image)
    after_angle = estimate_skew_angle(np.array(processed))

    assert abs(after_angle) < 1.0


def test_white_image_does_not_raise_error() -> None:
    white_image = Image.new("RGB", (300, 140), "white")

    processed = preprocess_stamp(white_image)

    assert processed.mode == "L"
    assert processed.width >= 600

