from io import BytesIO
from uuid import UUID

from fastapi.testclient import TestClient
from PIL import Image, features
import pytest

from eden_vision.config import Settings
from eden_vision.main import create_app
from eden_vision.schemas.classification import ClassificationResponse
from eden_vision.services.image_input import ImageErrorCode, ImageInputError, ImageInputService
from eden_vision.vision.adapters.dummy import DummyVisionModel
from eden_vision.vision.types import ImageInput


def encoded(fmt="PNG", mode="RGB", size=(4, 3), color=None, **options):
    with Image.new(mode, size, color) as image:
        output = BytesIO()
        image.save(output, format=fmt, **options)
        return output.getvalue()


def encoded_heif(size=(8, 6), color=(220, 30, 30)):
    # pillow-heif writes the same HEIF container used by ordinary HEIC stills.
    return encoded("HEIF", size=size, color=color)


@pytest.fixture
def client():
    with TestClient(create_app(Settings(_env_file=None))) as client:
        yield client


@pytest.fixture
def decoder():
    return ImageInputService(max_upload_bytes=10485760, max_image_pixels=25000000)


def upload(client, data, mime="image/png"):
    return client.post("/v1/vision/classify", files={"image": ("image.bin", data, mime)})


def assert_error(response, status, code):
    assert response.status_code == status, response.text
    data = response.json()
    assert UUID(data["request_id"]).version == 4
    assert response.headers["x-request-id"] == data["request_id"]
    assert data["error"]["code"] == code
    assert data["error"]["retryable"] is False


@pytest.mark.parametrize(("fmt", "mime"), [("JPEG", "image/jpeg"), ("PNG", "image/png"), ("WEBP", "image/webp")])
def test_supported_upload(client, fmt, mime):
    if fmt == "WEBP" and not features.check("webp"):
        pytest.skip("Pillow build lacks WebP")
    response = upload(client, encoded(fmt), mime)
    assert response.status_code == 200, response.text
    value = ClassificationResponse.model_validate(response.json())
    assert (value.image.width, value.image.height) == (4, 3)
    assert UUID(value.request_id).version == 4
    assert response.headers["x-request-id"] == value.request_id
    assert value.meta.model_id == "dummy-injected-candidates"
    assert value.predictions[0].label == "plant"
    assert value.predictions[0].confidence is None


@pytest.mark.parametrize(("data", "mime"), [(b"hello", "text/plain"), (b"hello", "image/jpeg"),
                                            (b"\xff\xd8\xff\xe0", "image/jpeg"), (b"", "image/png")])
def test_non_image_corrupt_and_spoofed_bytes(client, data, mime):
    assert_error(upload(client, data, mime), 422, "INVALID_IMAGE")


def test_mime_mismatch(client):
    assert_error(upload(client, encoded("PNG"), "image/jpeg"), 415, "UNSUPPORTED_IMAGE_TYPE")


def test_generic_mime_is_decoded(client):
    assert upload(client, encoded(), "application/octet-stream").status_code == 200


def test_actual_format_and_missing_mime(decoder):
    assert decoder.decode(encoded(), None).source_format == "PNG"


def test_upload_limit():
    with TestClient(create_app(Settings(_env_file=None, max_upload_bytes=20))) as client:
        assert_error(upload(client, encoded()), 413, "IMAGE_TOO_LARGE")


def test_body_limit_before_multipart_parsing():
    with TestClient(create_app(Settings(_env_file=None, max_upload_bytes=20))) as client:
        response = client.post("/v1/vision/classify", content=b"x" * (65536 + 21),
                               headers={"Content-Type": "multipart/form-data; boundary=x"})
        assert_error(response, 413, "IMAGE_TOO_LARGE")


def test_chunked_body_without_content_length_is_bounded():
    with TestClient(create_app(Settings(_env_file=None, max_upload_bytes=20))) as client:
        response = client.post("/v1/vision/classify", content=iter([b"x" * 32768] * 3),
                               headers={"Content-Type": "multipart/form-data; boundary=x"})
        assert_error(response, 413, "IMAGE_TOO_LARGE")


def test_pixel_limit():
    with TestClient(create_app(Settings(_env_file=None, max_image_pixels=10))) as client:
        assert_error(upload(client, encoded(size=(4, 3))), 413, "IMAGE_TOO_LARGE")


def test_byte_and_pixel_limits_inclusive():
    data = encoded(size=(4, 3))
    assert ImageInputService(max_upload_bytes=len(data), max_image_pixels=12).decode(data, "image/png").width == 4


@pytest.mark.parametrize("pillow_limit", [8, 4])
def test_bomb_warning_and_error_are_rejected(monkeypatch, decoder, pillow_limit):
    data = encoded(size=(4, 3))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", pillow_limit)
    with pytest.raises(ImageInputError) as exc:
        decoder.decode(data, "image/png")
    assert exc.value.code == ImageErrorCode.IMAGE_TOO_LARGE
    assert Image.MAX_IMAGE_PIXELS == pillow_limit


def test_exif_orientation(client, decoder):
    exif = Image.Exif()
    exif[274] = 6
    data = encoded("JPEG", size=(4, 2), exif=exif)
    decoded = decoder.decode(data, "image/jpeg")
    assert (decoded.width, decoded.height) == (2, 4)
    assert decoded.source_format == "JPEG"
    assert upload(client, data, "image/jpeg").json()["image"] == {"width": 2, "height": 4}


@pytest.mark.parametrize(("alpha", "expected"), [(0, (128, 128, 128)), (128, (192, 64, 64)), (255, (255, 0, 0))])
def test_alpha_composited_onto_gray(decoder, alpha, expected):
    value = decoder.decode(encoded(mode="RGBA", size=(1, 1), color=(255, 0, 0, alpha)), "image/png")
    assert value.pixels == bytes(expected)


@pytest.mark.parametrize(("mode", "fmt"), [("L", "PNG"), ("P", "PNG"), ("CMYK", "JPEG"), ("RGB", "PNG")])
def test_modes_normalized_to_rgb(decoder, mode, fmt):
    value = decoder.decode(encoded(fmt, mode), None)
    assert len(value.pixels) == value.width * value.height * 3


def test_palette_transparency(decoder):
    value = decoder.decode(encoded(mode="P", size=(1, 1), color=0, transparency=0), None)
    assert value.pixels == bytes((128, 128, 128))


@pytest.mark.parametrize("fmt", ["GIF", "BMP", "TIFF"])
def test_unsupported_formats(client, fmt):
    assert_error(upload(client, encoded(fmt), "application/octet-stream"), 415, "UNSUPPORTED_IMAGE_TYPE")


@pytest.mark.parametrize("fmt", ["GIF", "PNG", "WEBP"])
def test_multiframe(client, fmt):
    if fmt == "WEBP" and not features.check("webp"):
        pytest.skip("Pillow build lacks WebP")
    with Image.new("RGB", (4, 3), "red") as first, Image.new("RGB", (4, 3), "blue") as second:
        data = BytesIO()
        first.save(data, format=fmt, save_all=True, append_images=[second], duration=100, loop=0)
    assert_error(upload(client, data.getvalue(), "application/octet-stream"), 415, "UNSUPPORTED_IMAGE_TYPE")


def test_truncated_png(client):
    assert_error(upload(client, encoded()[:45]), 422, "INVALID_IMAGE")


def test_corrupt_heif_container_is_invalid(client):
    data = b"\x00\x00\x00\x18ftypheic" + b"\x00" * 12
    assert_error(upload(client, data, "image/heic"), 422, "INVALID_IMAGE")


@pytest.mark.parametrize(("suffix", "mime"), [(".heic", "image/heic"), (".heif", "image/heif")])
def test_heif_upload_and_rgb_normalization(client, decoder, suffix, mime):
    data = encoded_heif()
    value = decoder.decode(data, mime)
    assert value.source_format == "HEIF"
    assert (value.width, value.height, len(value.pixels)) == (8, 6, 8 * 6 * 3)
    response = client.post("/v1/vision/classify", files={"image": ("photo" + suffix, data, mime)})
    assert response.status_code == 200, response.text
    assert response.json()["image"] == {"width": 8, "height": 6}


def test_heif_mime_mismatch_and_corrupt(client):
    data = encoded_heif()
    assert_error(upload(client, data, "image/jpeg"), 415, "UNSUPPORTED_IMAGE_TYPE")
    assert_error(upload(client, data[:30], "image/heic"), 422, "INVALID_IMAGE")


def test_heif_pixel_limit_and_size_limit(client):
    data = encoded_heif(size=(8, 6))
    with TestClient(create_app(Settings(_env_file=None, max_image_pixels=10))) as limited:
        assert_error(upload(limited, data, "image/heic"), 413, "IMAGE_TOO_LARGE")
    with TestClient(create_app(Settings(_env_file=None, max_upload_bytes=len(data) - 1))) as limited:
        assert_error(upload(limited, data, "image/heic"), 413, "IMAGE_TOO_LARGE")


def test_missing_image(client):
    assert_error(client.post("/v1/vision/classify"), 422, "INVALID_IMAGE")


def test_malformed_multipart(client):
    response = client.post("/v1/vision/classify", content=b"broken", headers={"Content-Type": "multipart/form-data"})
    assert_error(response, 400, "INVALID_REQUEST")


def test_request_ids_are_distinct(client):
    ids = {upload(client, encoded()).json()["request_id"] for _ in range(2)}
    assert len(ids) == 2


def test_lifespan_and_replacement_model():
    model = DummyVisionModel(())
    image = ImageInput(1, 1, b"\x00\x00\x00")
    with pytest.raises(RuntimeError):
        model.predict(image)
    with TestClient(create_app(model_factory=lambda: model)) as client:
        assert model.predict(image) == ()
        assert upload(client, encoded()).json()["status"] == "unknown"
        assert client.get("/health").json() == {"status": "ok", "service": "eden-vision-ai"}
    with pytest.raises(RuntimeError):
        model.predict(image)


def test_inference_error_is_not_unknown(client, monkeypatch):
    def fail(image):
        raise RuntimeError("private failure details")
    monkeypatch.setattr(client.app.state.classification, "classify", fail)
    response = upload(client, encoded())
    assert_error(response, 500, "INFERENCE_FAILED")
    assert "private" not in response.text


def test_source_format_validation_and_backwards_compatibility():
    assert ImageInput(1, 1, b"\x00\x00\x00").source_format is None
    with pytest.raises(ValueError, match="source_format"):
        ImageInput(1, 1, b"\x00\x00\x00", "GIF")
