"""End-to-end evaluation preflight with isolated synthetic fixtures only.

This runner never touches the production dataset and never imports torch or
transformers. Fixtures are intentionally geometric and are not accuracy data.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eden_vision.classification.hierarchy import HierarchyResolver
from eden_vision.classification.taxonomy import load_taxonomy
from experiments.eval_dataset import import_inbox, promote_draft
from experiments.evaluation_core import aggregate_metrics, evaluate_image, load_dataset


def _fixture(path: Path, index: int) -> None:
    colors = [(220, 50, 50), (50, 180, 70), (60, 90, 220), (220, 190, 50)]
    with Image.new("RGB", (32, 24), colors[index % len(colors)]) as image:
        draw = ImageDraw.Draw(image)
        draw.rectangle((4, 4, 14, 14), fill=(255, 255, 255))
        draw.ellipse((18, 8, 28, 18), fill=(30, 30, 30))
        draw.line((index % 16, 0, index % 16, 23), fill=(index * 17 % 255, 40, 120), width=1)
        image.save(path, format="PNG")


def _rank(*concepts: str) -> list[dict[str, Any]]:
    return [{"concept_id": concept, "raw_score": float(len(concepts) - index), "score_kind": "fixture_raw_score"} for index, concept in enumerate(concepts)]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _make_dataset(root: Path) -> None:
    for name in ("inbox", "images", "reports"):
        (root / name).mkdir(parents=True, exist_ok=True)
    _write_json(root / "dataset.json", {"dataset_id": "eden-vision-preflight", "taxonomy_version": "eden-v1", "image_count": 0})
    (root / "manifest.jsonl").write_text("", encoding="utf-8")
    for index in range(10):
        _fixture(root / "inbox" / f"fixture-{index}.png", index)


def _label_fixture(root: Path) -> None:
    rows = [
        ("rose", ["fine_grained"], False),
        ("rose", ["fine_grained"], False),
        ("rose", ["fine_grained"], False),
        ("rose", ["fine_grained"], False),
        ("flower", ["broad_only"], False),
        (["person", "dog", "tree"], ["multi_object"], False),
        (["person", "dog", "tree"], ["multi_object"], False),
        ([], ["unknown"], True),
        ("plant", ["broad_only"], False),
        ("plant", ["broad_only", "ambiguous"], False),
    ]
    draft_path = root / "manifest.draft.jsonl"
    draft_rows = [json.loads(line) for line in draft_path.read_text().splitlines() if line.strip()]
    taxonomy = load_taxonomy()
    resolver = HierarchyResolver(taxonomy)
    for row, (expected, tags, unknown) in zip(draft_rows, rows):
        concepts = expected if isinstance(expected, list) else [expected]
        row["expected"] = [{"concept_id": concept, "acceptable_ancestors": list(resolver.ancestor_path(concept)[:-1])} for concept in concepts]
        row["tags"], row["unknown"] = tags, unknown
        row["notes"] = "human-authored preflight fixture"
    draft_path.write_text("".join(json.dumps(row) + "\n" for row in draft_rows), encoding="utf-8")


def _fake_predictions(model: str, image_id: str) -> list[dict[str, Any]]:
    if model == "fake-model-a":
        candidates = {
            "eval-001": _rank("rose", "flower", "plant", "living_thing", "tulip"),
            "eval-002": _rank("flower", "plant", "rose", "living_thing", "tulip"),
            "eval-003": _rank("plant", "flower", "rose", "living_thing", "tree"),
            "eval-004": _rank("car", "vehicle", "phone", "artifact", "person"),
            "eval-005": _rank("rose", "flower", "plant", "living_thing", "tulip"),
            "eval-006": _rank("person", "dog", "tree", "car", "flower"),
            "eval-007": _rank("person", "dog", "car", "phone", "flower"),
            "eval-008": _rank("phone", "car", "chair", "flower", "dog"),
            "eval-009": _rank("flower", "rose", "tulip", "plant", "living_thing"),
            "eval-010": _rank("artifact", "flower", "phone", "person", "tree"),
        }
    else:
        candidates = {
            "eval-001": _rank("flower", "rose", "plant", "car", "phone"),
            "eval-002": _rank("rose", "car", "phone", "flower", "plant"),
            "eval-003": _rank("tree", "car", "phone", "flower", "rose"),
            "eval-004": _rank("phone", "car", "artifact", "person", "tree"),
            "eval-005": _rank("rose", "tulip", "plant", "flower", "car"),
            "eval-006": _rank("person", "car", "dog", "phone", "tree"),
            "eval-007": _rank("car", "phone", "person", "dog", "tree"),
            "eval-008": _rank("flower", "plant", "rose", "tree", "dog"),
            "eval-009": _rank("phone", "car", "person", "artifact", "flower"),
            "eval-010": _rank("artifact", "phone", "car", "person", "tree"),
        }
    return candidates[image_id]


def _model_report(dataset, model: str, resolver: HierarchyResolver) -> dict[str, Any]:
    rows = []
    for index, entry in enumerate(dataset.entries, 1):
        rows.append(evaluate_image(entry, _fake_predictions(model, entry.id), float(index), resolver))
    return {
        "dataset_id": dataset.metadata.dataset_id,
        "taxonomy_version": dataset.metadata.taxonomy_version,
        "model_id": model,
        "model_revision": "preflight-fixture-v1",
        "image_count": len(rows),
        "score_kind": "fixture_raw_score",
        "metrics": aggregate_metrics(rows, resolver),
        "latency_summary": aggregate_metrics(rows, resolver)["latency_ms"],
        "images": rows,
    }


def run(root: Path) -> dict[str, Any]:
    if root.exists():
        shutil.rmtree(root)
    _make_dataset(root)
    import_result = import_inbox(root)
    _label_fixture(root)
    promoted = promote_draft(root, load_taxonomy())
    dataset = load_dataset(root, load_taxonomy())
    resolver = HierarchyResolver(load_taxonomy())
    reports = {model: _model_report(dataset, model, resolver) for model in ("fake-model-a", "fake-model-b")}
    comparison = {
        "dataset_id": dataset.metadata.dataset_id,
        "taxonomy_version": dataset.metadata.taxonomy_version,
        "models": {
            model: {"model_id": report["model_id"], "model_revision": report["model_revision"], "metrics": report["metrics"], "latency_summary": report["latency_summary"]}
            for model, report in reports.items()
        },
        "winner": None,
    }
    for model, report in reports.items():
        _write_json(root / "reports" / f"{model}.json", report)
    _write_json(root / "reports" / "comparison-report.json", comparison)
    return {"root": str(root), "import": import_result, "promoted": promoted, "reports": list(reports), "comparison": comparison}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="isolated output directory; defaults to a temporary directory")
    args = parser.parse_args()
    root = args.output or Path(tempfile.mkdtemp(prefix="eden-vision-eval-preflight-"))
    result = run(root)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
