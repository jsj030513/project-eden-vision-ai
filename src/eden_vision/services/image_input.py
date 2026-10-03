"""Bounded decoding into RGB8 domain bytes, without HTTP or model logic."""

from enum import StrEnum
from io import BytesIO
from threading import Lock
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener

from eden_vision.vision.types import ImageInput


class ImageErrorCode(StrEnum):
    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    UNSUPPORTED_IMAGE_TYPE = "UNSUPPORTED_IMAGE_TYPE"
    INVALID_IMAGE = "INVALID_IMAGE"


class ImageInputError(ValueError):
    def __init__(self, code: ImageErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


# pillow-heif exposes the primary still image to Pillow as format HEIF. Both
# standard upload MIME types are accepted for that decoded format.
register_heif_opener(thumbnails=False)
FORMAT_MIMES = {"JPEG": ("image/jpeg",), "PNG": ("image/png",),
                "WEBP": ("image/webp",), "HEIF": ("image/heic", "image/heif")}
ALPHA_BACKGROUND = (128, 128, 128)  # Neutral gray, explicitly composited before RGB output.
# Python 3.12 warning filters are process-global. Serialize our catch_warnings
# sections rather than racing concurrent decoder calls. Pillow limits stay on.
_DECODE_LOCK = Lock()


class ImageInputService:
    def __init__(self, *, max_upload_bytes: int, max_image_pixels: int) -> None:
        if any(type(v) is not int or v <= 0 for v in (max_upload_bytes, max_image_pixels)):
            raise ValueError("image limits must be positive integers")
        self.max_upload_bytes = max_upload_bytes
        self.max_image_pixels = max_image_pixels

    def decode(self, data: bytes, declared_mime: str | None) -> ImageInput:
        """Decoder format is authoritative, but a conflicting declared MIME is rejected.

        Missing/application/octet-stream MIME is accepted after full validation.
        Unrecognizable bytes (including truncated headers) are INVALID_IMAGE;
        identified unsupported formats and MIME mismatches are UNSUPPORTED_IMAGE_TYPE.
        Pillow-detected MPO uses only primary frame 0, with JPEG MIME/source
        semantics. Other frames are never decoded; input bytes are not rewritten.
        """
        if len(data) > self.max_upload_bytes:
            raise ImageInputError(ImageErrorCode.IMAGE_TOO_LARGE, "Upload exceeds byte limit")
        mime = (declared_mime or "").split(";", 1)[0].strip().lower()
        try:
            with _DECODE_LOCK, warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(data)) as image:
                    detected_format = image.format
                    source_format = "JPEG" if detected_format == "MPO" else detected_format
                    if source_format not in FORMAT_MIMES:
                        raise ImageInputError(ImageErrorCode.UNSUPPORTED_IMAGE_TYPE, "Only JPEG (including MPO primary still), PNG, static WebP, HEIC and HEIF are supported")
                    if mime not in ("", "application/octet-stream", *FORMAT_MIMES[source_format]):
                        raise ImageInputError(ImageErrorCode.UNSUPPORTED_IMAGE_TYPE, "Declared MIME does not match decoded format")
                    if detected_format == "MPO":
                        image.seek(0)
                    if image.width <= 0 or image.height <= 0 or image.width * image.height > self.max_image_pixels:
                        raise ImageInputError(ImageErrorCode.IMAGE_TOO_LARGE, "Image exceeds pixel limit")
                    if detected_format != "MPO" and getattr(image, "n_frames", 1) != 1:
                        raise ImageInputError(ImageErrorCode.UNSUPPORTED_IMAGE_TYPE, "Multi-frame images are not supported")
                    image.verify()
                # verify checks file integrity; reopen and load to force actual decoding.
                with Image.open(BytesIO(data)) as image:
                    if detected_format == "MPO":
                        image.seek(0)  # Only the primary JPEG still, never subsequent MPO frames.
                    image.load()
                    with ImageOps.exif_transpose(image) as oriented:
                        if "A" in oriented.getbands() or "transparency" in oriented.info:
                            with oriented.convert("RGBA") as rgba:
                                with Image.new("RGBA", rgba.size, (*ALPHA_BACKGROUND, 255)) as background:
                                    with Image.alpha_composite(background, rgba) as composite:
                                        with composite.convert("RGB") as rgb:
                                            return ImageInput(rgb.width, rgb.height, rgb.tobytes(), source_format)
                        with oriented.convert("RGB") as rgb:
                            return ImageInput(rgb.width, rgb.height, rgb.tobytes(), source_format)
        except ImageInputError:
            raise
        except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
            raise ImageInputError(ImageErrorCode.IMAGE_TOO_LARGE, "Image exceeds decoder safety limit") from exc
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError, EOFError) as exc:
            raise ImageInputError(ImageErrorCode.INVALID_IMAGE, "Image is empty, damaged or undecodable") from exc
