from PIL import Image

from app.services.orientation import apply_orientation, map_bbox_to_original_orientation


def _oriented_bbox_from_original(
    original_size: tuple[int, int],
    original_bbox: tuple[int, int, int, int],
    orientation_mode: str,
) -> tuple[int, int, int, int]:
    mask = Image.new("L", original_size, 0)
    x, y, w, h = original_bbox
    block = Image.new("L", (w, h), 255)
    mask.paste(block, (x, y))
    oriented_mask = apply_orientation(mask, orientation_mode)
    bbox = oriented_mask.getbbox()
    assert bbox is not None
    left, top, right, bottom = bbox
    return left, top, right - left, bottom - top


def test_map_bbox_back_to_original_for_180_rotation() -> None:
    original_size = (1000, 1400)
    original_bbox = (160, 980, 260, 220)
    oriented_bbox = _oriented_bbox_from_original(original_size, original_bbox, "rot_180")
    mapped = map_bbox_to_original_orientation(
        bbox=oriented_bbox,
        orientation_mode="rot_180",
        oriented_size=original_size,
        original_size=original_size,
    )
    assert mapped == original_bbox


def test_map_bbox_back_to_original_for_90_rotation() -> None:
    original_size = (1100, 1700)
    original_bbox = (210, 1180, 300, 250)
    oriented_bbox = _oriented_bbox_from_original(original_size, original_bbox, "rot_90")
    mapped = map_bbox_to_original_orientation(
        bbox=oriented_bbox,
        orientation_mode="rot_90",
        oriented_size=(1700, 1100),
        original_size=original_size,
    )
    assert mapped == original_bbox


def test_map_bbox_back_to_original_for_270_rotation() -> None:
    original_size = (1200, 1800)
    original_bbox = (220, 1210, 280, 260)
    oriented_bbox = _oriented_bbox_from_original(original_size, original_bbox, "rot_270")
    mapped = map_bbox_to_original_orientation(
        bbox=oriented_bbox,
        orientation_mode="rot_270",
        oriented_size=(1800, 1200),
        original_size=original_size,
    )
    assert mapped == original_bbox
