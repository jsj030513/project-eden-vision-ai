import hashlib
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image, ImageOps, MpoImagePlugin
import pytest

from eden_vision.config import Settings
from eden_vision.main import create_app
from eden_vision.services.image_input import ImageErrorCode, ImageInputError, ImageInputService
from experiments.eval_dataset import _read_jsonl, import_inbox, init_dataset
from experiments.model_runtime import load_evaluation_taxonomy


def mpo_bytes(*, mode="RGB", orientation=None):
    # Real local MPO container: distinct primary/secondary pixels and dimensions.
    options = {}
    if orientation is not None:
        exif = Image.Exif()
        exif[274] = orientation
        options["exif"] = exif
    with Image.new(mode, (8, 6), "red") as first, Image.new(mode, (10, 12), "blue") as second:
        stream = BytesIO()
        first.save(stream, format="MPO", save_all=True, append_images=[second], **options)
    return stream.getvalue()


@pytest.fixture
def decoder():
    return ImageInputService(max_upload_bytes=10 * 1024 * 1024, max_image_pixels=25_000_000)


@pytest.mark.parametrize("mime", [None, "", "image/jpeg", "application/octet-stream"])
def test_real_mpo_uses_only_primary_frame_and_jpeg_semantics(decoder, monkeypatch, mime):
    data = mpo_bytes()
    with Image.open(BytesIO(data)) as image:
        assert image.format == "MPO" and image.n_frames == 2
        expected = image.convert("RGB").tobytes()
        image.seek(1)
        assert image.size == (10, 12) and image.convert("RGB").tobytes() != expected

    seeks, loads = [], []
    original_seek = MpoImagePlugin.MpoImageFile.seek
    original_load = MpoImagePlugin.MpoImageFile.load

    def primary_seek(self, frame):
        assert frame == 0, "secondary MPO frame was selected"
        seeks.append(frame)
        return original_seek(self, frame)

    def primary_load(self):
        assert self.tell() == 0, "secondary MPO frame was decoded"
        loads.append(self.tell())
        return original_load(self)

    monkeypatch.setattr(MpoImagePlugin.MpoImageFile, "seek", primary_seek)
    monkeypatch.setattr(MpoImagePlugin.MpoImageFile, "load", primary_load)
    result = decoder.decode(data, mime)
    assert result.source_format == "JPEG"
    assert (result.width, result.height) == (8, 6)
    assert result.pixels == expected
    assert len(seeks) >= 2 and loads


@pytest.mark.parametrize("mime", ["image/png", "image/webp", "image/heic", "image/heif", "image/mpo"])
def test_mpo_conflicting_mime_rejected(decoder, mime):
    with pytest.raises(ImageInputError) as exc:
        decoder.decode(mpo_bytes(), mime)
    assert exc.value.code == ImageErrorCode.UNSUPPORTED_IMAGE_TYPE


def test_mpo_primary_exif_orientation_and_rgb_conversion(decoder):
    data = mpo_bytes(mode="L", orientation=6)
    with Image.open(BytesIO(data)) as image:
        assert image.format == "MPO" and image.mode == "L"
        expected = ImageOps.exif_transpose(image).convert("RGB").tobytes()
    result = decoder.decode(data, "image/jpeg")
    assert result.source_format == "JPEG"
    assert (result.width, result.height) == (6, 8)
    assert result.pixels == expected and len(result.pixels) == 6 * 8 * 3


def test_mpo_byte_and_primary_pixel_limits():
    data = mpo_bytes()
    assert ImageInputService(max_upload_bytes=len(data), max_image_pixels=48).decode(data, None).width == 8
    for byte_limit, pixel_limit in [(len(data) - 1, 48), (len(data), 47)]:
        with pytest.raises(ImageInputError) as exc:
            ImageInputService(max_upload_bytes=byte_limit, max_image_pixels=pixel_limit).decode(data, None)
        assert exc.value.code == ImageErrorCode.IMAGE_TOO_LARGE


@pytest.mark.parametrize("pillow_limit", [30, 20])
def test_mpo_decompression_warning_and_error_still_rejected(decoder, monkeypatch, pillow_limit):
    data = mpo_bytes()
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", pillow_limit)
    with pytest.raises(ImageInputError) as exc:
        decoder.decode(data, "image/jpeg")
    assert exc.value.code == ImageErrorCode.IMAGE_TOO_LARGE


@pytest.mark.parametrize("fmt", ["GIF", "BMP", "TIFF"])
def test_unsupported_input_cannot_claim_jpeg_mime(decoder, fmt):
    with Image.new("RGB", (8, 6)) as image:
        stream = BytesIO()
        image.save(stream, format=fmt)
    with pytest.raises(ImageInputError) as exc:
        decoder.decode(stream.getvalue(), "image/jpeg")
    assert exc.value.code == ImageErrorCode.UNSUPPORTED_IMAGE_TYPE


def test_plain_jpeg_source_format_unchanged(decoder):
    with Image.new("RGB", (8, 6), "green") as image:
        stream = BytesIO()
        image.save(stream, format="JPEG")
    result = decoder.decode(stream.getvalue(), "image/jpeg")
    assert result.source_format == "JPEG" and (result.width, result.height) == (8, 6)


def test_mpo_upload_through_shared_service():
    with TestClient(create_app(Settings(_env_file=None))) as client:
        response = client.post("/v1/vision/classify", files={"image": ("phone.JPG", mpo_bytes(), "image/jpeg")})
    assert response.status_code == 200
    assert response.json()["image"] == {"width": 8, "height": 6}


def test_import_mpo_preserves_original_container_and_existing_draft(tmp_path):
    root = tmp_path / "dataset"
    init_dataset(root, "test-mpo", load_evaluation_taxonomy("eden-v2"))
    with Image.new("RGB", (3, 2), "green") as image:
        image.save(root / "inbox/old.png")
    assert import_inbox(root)["imported"] == 1
    previous = _read_jsonl(root / "manifest.draft.jsonl")
    previous_bytes = (root / previous[0]["file"]).read_bytes()
    source = root / "inbox/phone.JPG"
    original = mpo_bytes(orientation=6)
    source.write_bytes(original)
    result = import_inbox(root)
    assert result == {"scanned": 2, "imported": 1, "duplicates": 1, "rejected": 0}
    rows = _read_jsonl(root / "manifest.draft.jsonl")
    assert rows[:1] == previous and (root / previous[0]["file"]).read_bytes() == previous_bytes
    row = rows[1]
    target = root / row["file"]
    assert target.suffix == ".jpg" and row["format"] == "JPEG"
    assert (row["width"], row["height"]) == (6, 8)
    assert row["sha256"] == hashlib.sha256(original).hexdigest()
    assert target.read_bytes() == source.read_bytes() == original
    with Image.open(target) as image:
        assert image.format == "MPO" and image.n_frames == 2  # Container was not re-encoded.
