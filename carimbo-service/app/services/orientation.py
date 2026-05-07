from PIL import Image

from app.services.detector import BBoxTuple

try:
    _ROTATE_90 = Image.Transpose.ROTATE_90
    _ROTATE_180 = Image.Transpose.ROTATE_180
    _ROTATE_270 = Image.Transpose.ROTATE_270
except AttributeError:  # pragma: no cover - compat com Pillow antigo.
    _ROTATE_90 = Image.ROTATE_90
    _ROTATE_180 = Image.ROTATE_180
    _ROTATE_270 = Image.ROTATE_270

ORIENTATION_MODE_TO_DEGREES = {
    "rot_0": 0,
    "rot_180": 180,
    "rot_90": 90,
    "rot_270": 270,
}


def apply_orientation(image: Image.Image, mode: str) -> Image.Image:
    if mode == "rot_0":
        return image
    if mode == "rot_180":
        return image.transpose(_ROTATE_180)
    if mode == "rot_90":
        return image.transpose(_ROTATE_90)
    if mode == "rot_270":
        return image.transpose(_ROTATE_270)
    return image


def inverse_orientation_mode(mode: str) -> str:
    if mode == "rot_90":
        return "rot_270"
    if mode == "rot_270":
        return "rot_90"
    return mode


def map_bbox_to_original_orientation(
    *,
    bbox: BBoxTuple,
    orientation_mode: str,
    oriented_size: tuple[int, int],
    original_size: tuple[int, int],
) -> BBoxTuple:
    if orientation_mode == "rot_0":
        return bbox

    x, y, w, h = bbox
    oriented_width, oriented_height = oriented_size
    x = max(0, min(oriented_width - 1, int(x)))
    y = max(0, min(oriented_height - 1, int(y)))
    w = max(1, min(int(w), oriented_width - x))
    h = max(1, min(int(h), oriented_height - y))

    mask = Image.new("L", oriented_size, 0)
    stamp = Image.new("L", (w, h), 255)
    mask.paste(stamp, (x, y))
    inv_mask = apply_orientation(mask, inverse_orientation_mode(orientation_mode))
    if inv_mask.size != original_size:
        inv_mask = inv_mask.resize(original_size)

    bb = inv_mask.getbbox()
    if not bb:
        return 0, 0, max(1, original_size[0]), max(1, original_size[1])
    left, top, right, bottom = bb
    return left, top, max(1, right - left), max(1, bottom - top)
