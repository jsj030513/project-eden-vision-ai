"""Offline, human-only Ground Truth editor, separate from the production API."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
from threading import RLock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict
from starlette.middleware.trustedhost import TrustedHostMiddleware

from eden_vision.classification.hierarchy import HierarchyResolver
from experiments.eval_dataset import (
    CANONICAL_FIELDS, _read_jsonl, _write_jsonl, check_draft_taxonomy, dataset_taxonomy, draft_incomplete,
)
from experiments.evaluation_core import ALLOWED_TAGS, ManifestValidationError, validate_dataset_rows
from experiments.model_runtime import load_evaluation_taxonomy


class LabelInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    taxonomy_version: str
    revision: str
    concept_ids: list[str]
    unknown: bool
    tags: list[str]
    notes: str | None = None


def _revision(row: dict) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def create_app(dataset: Path | str) -> FastAPI:
    dataset = Path(dataset).expanduser().resolve()
    draft = dataset / "manifest.draft.jsonl"
    taxonomy = load_evaluation_taxonomy("eden-v2")
    resolver = HierarchyResolver(taxonomy)
    mutex = RLock()

    def read_rows() -> list[dict]:
        dataset_taxonomy(dataset, taxonomy)
        if not draft.is_file():
            raise ManifestValidationError("manifest.draft.jsonl is missing; import the dataset first")
        rows = _read_jsonl(draft)
        ids = set()
        for row in rows:
            check_draft_taxonomy(row, taxonomy)
            item_id = row.get("id")
            if not isinstance(item_id, str) or not item_id or item_id in ids:
                raise ManifestValidationError("draft has missing or duplicate IDs")
            ids.add(item_id)
        return rows

    read_rows()  # Read-only startup: never create, migrate, or initialize manifests.
    concepts = [
        {"concept_id": c.concept_id, "display_name": c.display_name,
         "path": list(resolver.ancestor_path(c.concept_id)),
         "broad_category": c.broad_category_id}
        for c in taxonomy.concepts
    ]
    known = {c.concept_id for c in taxonomy.concepts}

    @contextmanager
    def write_lock():
        # Stable sidecar survives atomic manifest replacement. Serializes UI
        # instances as well as worker threads; existing CLI writers do not lock.
        with mutex, (dataset / ".labeling.lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def find_row(rows: list[dict], item_id: str) -> dict:
        for row in rows:
            if row["id"] == item_id:
                return row
        raise HTTPException(404, "Unknown eval ID")

    def public_row(row: dict) -> dict:
        return {**row, "completed": not draft_incomplete(row), "revision": _revision(row)}

    app = FastAPI(title="Eden Vision Ground Truth", docs_url=None, redoc_url=None,
                  openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]"])

    @app.middleware("http")
    async def local_browser_guard(request: Request, call_next):
        # Cross-origin websites cannot send this header without a CORS preflight.
        # No CORS permissions are granted by this app.
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if (request.headers.get("x-labeling-request") != "1"
                    or (origin and origin != f"{request.url.scheme}://{request.url.netloc}")):
                return JSONResponse({"detail": "Same-origin labeling request required"}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; img-src 'self'; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
        )
        return response

    @app.exception_handler(ManifestValidationError)
    async def manifest_error(request: Request, exc: ManifestValidationError):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.get("/health")
    def health():
        return {"status": "ok", "taxonomy_version": taxonomy.taxonomy_version}

    @app.get("/", response_class=HTMLResponse)
    def page():
        return Path(__file__).with_name("labeling_app.html").read_text(encoding="utf-8")

    @app.get("/api/dataset")
    def items():
        rows = read_rows()
        completed = sum(not draft_incomplete(row) for row in rows)
        return {"taxonomy_version": taxonomy.taxonomy_version,
                "allowed_tags": sorted(ALLOWED_TAGS),
                "total": len(rows), "completed": completed, "remaining": len(rows) - completed,
                "initial_index": next((i for i, r in enumerate(rows) if draft_incomplete(r)), 0),
                "items": [public_row(row) for row in rows]}

    @app.get("/api/concepts")
    def search_concepts(q: str = ""):
        query = q.strip().casefold()
        return {"taxonomy_version": taxonomy.taxonomy_version, "concepts": [
            c for c in concepts if query in (c["concept_id"] + " " + c["display_name"]).casefold()
        ]}

    @app.get("/api/images/{item_id}")
    def image(item_id: str):
        row = find_row(read_rows(), item_id)
        raw = row.get("file")
        if not isinstance(raw, str) or "\\" in raw:
            raise HTTPException(404, "Invalid registered image")
        relative = PurePosixPath(raw)
        if (relative.is_absolute() or ".." in relative.parts
                or len(relative.parts) < 2 or relative.parts[0] != "images"):
            raise HTTPException(404, "Invalid registered image")
        try:
            images = (dataset / "images").resolve(strict=True)
            target = (dataset / relative).resolve(strict=True)
            if (not images.is_relative_to(dataset) or not target.is_relative_to(images)
                    or not target.is_file()):
                raise HTTPException(404, "Invalid registered image")
        except (OSError, RuntimeError):
            raise HTTPException(404, "Registered image not found") from None
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                ".webp": "image/webp", ".heif": "image/heif", ".heic": "image/heic"}
        if target.suffix.lower() not in mime:
            raise HTTPException(404, "Unsupported image format")
        return FileResponse(target, media_type=mime[target.suffix.lower()])

    @app.put("/api/items/{item_id}")
    def save(item_id: str, label: LabelInput):
        if label.taxonomy_version != taxonomy.taxonomy_version:
            raise HTTPException(422, "taxonomy_version must be eden-v2")
        if any(cid not in known for cid in label.concept_ids):
            raise HTTPException(422, "Unknown concept ID")
        if not label.unknown and not label.concept_ids:
            raise HTTPException(422, "Known items require at least one expected concept")
        if any(tag not in ALLOWED_TAGS for tag in label.tags):
            raise HTTPException(422, "Tags must belong to canonical ALLOWED_TAGS")
        tags = list(dict.fromkeys(label.tags))
        if "single_object" in tags and "multi_object" in tags:
            raise HTTPException(422, "single_object and multi_object are mutually exclusive")
        if not label.unknown and "unknown" in tags:
            raise HTTPException(422, "unknown tag must match unknown field")
        if label.unknown and "unknown" not in tags:
            tags.append("unknown")
        if not tags:
            raise HTTPException(422, "At least one tag is required")
        expected = [] if label.unknown else [
            {"concept_id": cid, "acceptable_ancestors": list(resolver.ancestor_path(cid)[:-1])}
            for cid in dict.fromkeys(label.concept_ids)
        ]
        try:
            with write_lock():
                rows = read_rows()  # Never write a cached copy of other human labels.
                row = find_row(rows, item_id)
                if label.revision != _revision(row):
                    raise HTTPException(409, "This item changed. Reload before saving to preserve newer labels.")
                proposed = {**row, "expected": expected, "unknown": label.unknown,
                            "tags": tags, "notes": label.notes}
                # Validate this completed row using the canonical contract. The
                # one-row metadata is in-memory only; unfinished drafts remain untouched.
                metadata, _ = dataset_taxonomy(dataset, taxonomy)
                try:
                    validate_dataset_rows(dataset, {**metadata, "image_count": 1},
                                          [{key: proposed[key] for key in CANONICAL_FIELDS}], taxonomy)
                except ManifestValidationError as exc:
                    raise HTTPException(422, str(exc)) from exc
                row.update(proposed)
                _write_jsonl(draft, rows)
                return {"item": public_row(row)}
        except OSError as exc:
            raise HTTPException(500, "Save failed; previous manifest remains intact. Check disk permissions/space.") from exc

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("/Users/dev/Datasets/eden-vision-eval"))
    parser.add_argument("--host", choices=["127.0.0.1", "::1"], default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(create_app(args.dataset), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
