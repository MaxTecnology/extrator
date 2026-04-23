from PIL import Image, ImageDraw

from app.services.detector import detect_stamp_region


def _blank_canvas() -> Image.Image:
    return Image.new("RGB", (1200, 1600), "white")


def test_detects_rectangle_in_lower_left_roi() -> None:
    image = _blank_canvas()
    draw = ImageDraw.Draw(image)
    draw.rectangle((120, 1080, 520, 1270), fill=(20, 20, 20))

    result = detect_stamp_region(image, min_contour_area=2000, padding_px=20)

    assert result.found is True
    assert result.bbox is not None
    assert result.confidence > 0.5
    assert result.reason == "detectado_por_contorno"


def test_blank_image_falls_back() -> None:
    image = _blank_canvas()

    result = detect_stamp_region(image, min_contour_area=2000, padding_px=20)

    assert result.found is False
    assert result.bbox is None
    assert result.reason == "fallback_regiao_inferior"
    assert result.confidence <= 0.45


def test_upper_right_rectangle_should_fallback() -> None:
    image = _blank_canvas()
    draw = ImageDraw.Draw(image)
    draw.rectangle((760, 140, 1110, 360), fill=(20, 20, 20))

    result = detect_stamp_region(image, min_contour_area=2000, padding_px=20)

    assert result.found is False
    assert result.bbox is None
    assert result.reason == "fallback_regiao_inferior"


def test_stamp_with_blue_signature_lines_still_detects() -> None:
    image = _blank_canvas()
    draw = ImageDraw.Draw(image)
    draw.rectangle((110, 1060, 560, 1290), fill=(35, 35, 35))
    draw.line((130, 1200, 520, 1125), fill=(40, 90, 220), width=6)
    draw.line((150, 1235, 540, 1165), fill=(30, 80, 220), width=5)
    draw.line((180, 1265, 500, 1190), fill=(20, 70, 220), width=4)

    result = detect_stamp_region(image, min_contour_area=2000, padding_px=20)

    assert result.found is True
    assert result.bbox is not None
    assert result.confidence > 0.5


def test_detects_stamp_slightly_above_55_percent_band() -> None:
    image = _blank_canvas()
    draw = ImageDraw.Draw(image)
    # Carimbo posicionado entre 44% e 54% da altura.
    draw.rectangle((180, 720, 620, 870), fill=(20, 20, 20))
    draw.line((210, 845, 590, 760), fill=(0, 0, 0), width=7)

    result = detect_stamp_region(image, min_contour_area=2000, padding_px=20)

    assert result.found is True
    assert result.bbox is not None
