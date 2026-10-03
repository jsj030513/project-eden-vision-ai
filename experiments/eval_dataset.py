"""Human-controlled dataset registration; no image understanding or AI imports."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from eden_vision.classification.taxonomy import Taxonomy
from eden_vision.services.image_input import ImageInputError, ImageInputService
from experiments.evaluation_core import (
    HASH_CHUNK_BYTES, ManifestValidationError, _json_object, file_sha256,
    load_dataset, validate_dataset_rows,
)
from experiments.model_runtime import load_evaluation_taxonomy

FORMAT_SUFFIX = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "HEIF": ".heif"}
CANONICAL_FIELDS = ("id", "file", "sha256", "expected", "tags", "unknown", "notes")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ManifestValidationError(f"invalid JSON at {path}:{line_no}") from exc
        if not isinstance(value, dict):
            raise ManifestValidationError(f"expected object at {path}:{line_no}")
        rows.append(value)
    return rows


def _jsonl(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def _atomic_text(path: Path, text: str) -> None:
    """Same-directory staging + fsync + replace; never expose truncated JSON."""
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_text(path, _jsonl(rows))


def _commit_manifests(changes: dict[Path, str]) -> None:
    """Rollback ordinary write failures. Not a multi-file power-loss transaction.

    Canonical is written before metadata and draft, preserving labels if killed.
    After an abrupt interruption, inspect the files before retrying; never
    silently discard a duplicate draft or repair metadata automatically.
    Run dataset mutations from one process at a time.
    """
    previous = {path: path.read_text(encoding="utf-8") if path.exists() else None for path in changes}
    written = []
    try:
        for path, text in changes.items():
            _atomic_text(path, text)
            written.append(path)
    except OSError:
        for path in reversed(written):
            if previous[path] is None:
                path.unlink(missing_ok=True)
            else:
                _atomic_text(path, previous[path])
        raise


_sha256 = file_sha256


def dataset_taxonomy(dataset_dir: Path, taxonomy: Taxonomy | None = None) -> tuple[dict[str, Any], Taxonomy]:
    metadata = _json_object(dataset_dir / "dataset.json")
    selected = taxonomy or load_evaluation_taxonomy(metadata.get("taxonomy_version"))
    if metadata.get("taxonomy_version") != selected.taxonomy_version:
        raise ManifestValidationError("dataset taxonomy_version does not match selected Ground Truth taxonomy")
    return metadata, selected


def init_dataset(dataset_dir: Path, dataset_id: str, taxonomy: Taxonomy) -> dict[str, Any]:
    """Create the existing canonical format in a new root; never reuse a root.

    A failed initialization may leave a partial new directory for inspection.
    Retrying never truncates it or any pre-existing user dataset.
    """
    dataset_dir = Path(dataset_dir).expanduser().absolute()
    metadata = {"dataset_id": dataset_id, "taxonomy_version": taxonomy.taxonomy_version, "image_count": 0}
    validate_dataset_rows(dataset_dir, metadata, [], taxonomy)
    dataset_dir.mkdir(exist_ok=False)
    for name in ("inbox", "images", "rejected", "reports", "backups"):
        (dataset_dir / name).mkdir()
    _atomic_text(dataset_dir / "dataset.json", json.dumps(metadata, indent=2) + "\n")
    _write_jsonl(dataset_dir / "manifest.jsonl", [])
    _write_jsonl(dataset_dir / "manifest.draft.jsonl", [])
    load_dataset(dataset_dir, taxonomy)
    return metadata


def set_dataset_taxonomy(dataset_dir: Path, taxonomy: Taxonomy) -> str:
    """Explicit change only before any draft/canonical annotations exist."""
    metadata = _json_object(dataset_dir / "dataset.json")
    if (_read_jsonl(dataset_dir / "manifest.jsonl") or
            _read_jsonl(dataset_dir / "manifest.draft.jsonl") or metadata.get("image_count") != 0):
        raise ManifestValidationError("set-taxonomy requires empty canonical and draft manifests")
    metadata["taxonomy_version"] = taxonomy.taxonomy_version
    validate_dataset_rows(dataset_dir, metadata, [], taxonomy)
    _atomic_text(dataset_dir / "dataset.json", json.dumps(metadata, indent=2) + "\n")
    return taxonomy.taxonomy_version


def _next_id(used: set[str]) -> str:
    number = 1
    while f"eval-{number:03d}" in used:
        number += 1
    return f"eval-{number:03d}"


def _read_import(source: Path, limit: int) -> tuple[bytes, str]:
    # Reject oversized regular files without opening/reading their contents.
    if source.stat().st_size > limit:
        raise ValueError("file exceeds upload byte limit")
    digest = hashlib.sha256()
    data = bytearray()
    with source.open("rb") as stream:
        while block := stream.read(min(HASH_CHUNK_BYTES, limit + 1 - len(data))):
            data.extend(block)
            if len(data) > limit:
                raise ValueError("file grew beyond upload byte limit")
            digest.update(block)
    return bytes(data), digest.hexdigest()


def import_inbox(dataset_dir: Path, *, max_upload_bytes: int = 10 * 1024 * 1024,
                 max_image_pixels: int = 25_000_000, taxonomy: Taxonomy | None = None) -> dict[str, int]:
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    metadata, selected = dataset_taxonomy(dataset_dir, taxonomy)
    inbox, images = dataset_dir / "inbox", dataset_dir / "images"
    inbox.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    draft_path = dataset_dir / "manifest.draft.jsonl"
    drafts = _read_jsonl(draft_path)
    rows = [*_read_jsonl(dataset_dir / "manifest.jsonl"), *drafts]
    used_ids = {row["id"] for row in rows}
    used_ids.update(path.stem for path in images.iterdir())
    known_hashes = {row.get("sha256") for row in rows}
    known_hashes.update(file_sha256(path) for path in images.iterdir() if path.is_file())
    service = ImageInputService(max_upload_bytes=max_upload_bytes, max_image_pixels=max_image_pixels)
    scanned = imported = duplicates = rejected = 0
    for source in sorted(path for path in inbox.iterdir() if path.is_file()):
        scanned += 1
        try:
            data, digest = _read_import(source, max_upload_bytes)
            image = service.decode(data, None)
        except (OSError, ValueError, ImageInputError):
            rejected += 1
            continue
        if digest in known_hashes:
            duplicates += 1
            continue
        item_id = _next_id(used_ids)
        destination = images / f"{item_id}{FORMAT_SUFFIX[image.source_format]}"
        # Exclusive creation; copy exactly the bytes decoded/hashed, not a
        # second read of a potentially changed source. Inbox remains untouched.
        with destination.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        drafts.append({
            "id": item_id, "file": destination.relative_to(dataset_dir).as_posix(),
            "sha256": digest, "format": image.source_format,
            "width": image.width, "height": image.height, "file_size_bytes": len(data),
            "taxonomy_version": selected.taxonomy_version, "source_filename": source.name,
            "expected": None, "tags": [], "unknown": None, "notes": None,
        })
        used_ids.add(item_id)
        known_hashes.add(digest)
        imported += 1
    _write_jsonl(draft_path, drafts)
    return {"scanned": scanned, "imported": imported, "duplicates": duplicates, "rejected": rejected}


def check_draft_taxonomy(row: dict[str, Any], taxonomy: Taxonomy) -> None:
    # Legacy drafts were v1-only. Missing version must never silently mean v2.
    version = row.get("taxonomy_version", "eden-v1")
    if version != taxonomy.taxonomy_version:
        raise ManifestValidationError(f"draft taxonomy_version mismatch: {row.get('id')} ({version})")


def draft_incomplete(row: dict[str, Any]) -> bool:
    return row.get("expected") is None or row.get("unknown") is None or row.get("tags") == []


def canonical_row(row: dict[str, Any]) -> dict[str, Any]:
    required = set(CANONICAL_FIELDS) | {"format", "width", "height", "file_size_bytes"}
    allowed = required | {"taxonomy_version", "source_filename"}
    if not required <= row.keys() or not row.keys() <= allowed:
        raise ManifestValidationError(f"draft row {row.get('id')} has unexpected or missing fields")
    return {key: row[key] for key in CANONICAL_FIELDS}


class PromotionError(ManifestValidationError):
    def __init__(self, message: str, summary: dict[str, Any]):
        super().__init__(message)
        self.summary = summary


def promote_draft(dataset_dir: Path, taxonomy: Taxonomy) -> dict[str, Any]:
    """Skip unfinished rows; reject invalid/conflicting completed rows before writes."""
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    drafts = _read_jsonl(dataset_dir / "manifest.draft.jsonl")
    summary = {"promoted": 0, "remaining_drafts": len(drafts), "skipped_incomplete": 0,
               "duplicates": 0, "errors": []}
    try:
        metadata, _ = dataset_taxonomy(dataset_dir, taxonomy)
        load_dataset(dataset_dir, taxonomy)  # Protect existing canonical content.
        canonical = _read_jsonl(dataset_dir / "manifest.jsonl")
        remaining, completed = [], []
        for row in drafts:
            check_draft_taxonomy(row, taxonomy)
            if draft_incomplete(row):
                remaining.append(row)
                summary["skipped_incomplete"] += 1
            else:
                completed.append(canonical_row(row))
        if not completed:
            return summary
        proposed = [*canonical, *completed]
        new_metadata = {**metadata, "image_count": len(proposed)}
        validate_dataset_rows(dataset_dir, new_metadata, proposed, taxonomy)
    except ManifestValidationError as exc:
        summary["errors"].append(str(exc))
        summary["duplicates"] = int("duplicate" in str(exc))
        raise PromotionError(f"canonical promotion rejected: {exc}", summary) from exc
    _commit_manifests({
        dataset_dir / "manifest.jsonl": _jsonl(proposed),
        dataset_dir / "dataset.json": json.dumps(new_metadata, indent=2) + "\n",
        dataset_dir / "manifest.draft.jsonl": _jsonl(remaining),
    })
    load_dataset(dataset_dir, taxonomy)  # Verify persisted data with evaluation validator.
    summary.update(promoted=len(completed), remaining_drafts=len(remaining))
    return summary


def update_dataset_count(dataset_dir: Path, count: int | None = None) -> int:
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    metadata_path = dataset_dir / "dataset.json"
    metadata = _json_object(metadata_path)
    actual = len(_read_jsonl(dataset_dir / "manifest.jsonl"))
    if count is not None and count != actual:
        raise ManifestValidationError("image_count must equal canonical count")
    metadata["image_count"] = actual
    _atomic_text(metadata_path, json.dumps(metadata, indent=2) + "\n")
    return actual


def dataset_status(dataset_dir: Path) -> dict[str, int]:
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    inbox_count = sum(path.is_file() for path in (dataset_dir / "inbox").glob("*") ) if (dataset_dir / "inbox").exists() else 0
    imported_count = sum(path.is_file() for path in (dataset_dir / "images").glob("*") ) if (dataset_dir / "images").exists() else 0
    draft_count = len(_read_jsonl(dataset_dir / "manifest.draft.jsonl"))
    canonical_count = len(_read_jsonl(dataset_dir / "manifest.jsonl"))
    return {"inbox": inbox_count, "imported": imported_count, "draft": draft_count, "canonical": canonical_count, "ready": canonical_count}
