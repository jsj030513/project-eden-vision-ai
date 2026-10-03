import hashlib
import json
from pathlib import Path
import subprocess
import sys

from fastapi.testclient import TestClient
from PIL import Image
import pytest

import experiments.collect_open_set_signals as collector
from experiments.eval_dataset import _read_jsonl, _write_jsonl, import_inbox, init_dataset, promote_draft
from experiments.evaluation_core import ManifestValidationError, load_dataset
from experiments.labeling_app import create_app
from experiments.model_runtime import MODELS, RankedImage, load_evaluation_taxonomy


@pytest.fixture
def root(tmp_path):
    root = tmp_path / "open-set"
    init_dataset(root, "eden-vision-open-set-eval-v1", load_evaluation_taxonomy("eden-v2"))
    return root


def file_bytes(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def populate(root):
    annotations = [(["cat"], ["single_object"], False),
                   (["animal"], ["broad_only"], False),
                   (["cat"], ["ambiguous"], False),
                   ([], ["unknown"], True),
                   (["electronic_device"], ["broad_only", "ambiguous"], False)]
    rows = []
    for i, (labels, tags, unknown) in enumerate(annotations):
        path = root / "images" / f"{i}.png"
        Image.new("RGB", (4, 3), (i, 10, 20)).save(path)
        rows.append({"id": f"eval-{i + 1:03}", "file": f"images/{i}.png",
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "expected": [{"concept_id": cid} for cid in labels],
                     "tags": tags, "unknown": unknown, "notes": f"human note {i}"})
    set_rows(root, rows)
    return rows


def set_rows(root, rows):
    _write_jsonl(root / "manifest.jsonl", rows)
    metadata = json.loads((root / "dataset.json").read_text())
    (root / "dataset.json").write_text(json.dumps({**metadata, "image_count": len(rows)}))


class FakeRuntime:
    def __init__(self, model):
        self.name = model
        self.metadata = {"model_id": MODELS[model].model_id, "model_revision": MODELS[model].revision}
        self.loaded = self.closed = False
        self.rank_calls = 0

    def load(self):
        self.loaded = True

    def prepare_taxonomy(self, taxonomy):
        assert self.loaded
        self.taxonomy = taxonomy
        return {"candidate_count": len(taxonomy.concepts)}

    def memory(self):
        return {}

    def rank(self, image):
        self.rank_calls += 1
        first = ["cat", "animal", "living_thing", "dog", "car"]
        ids = first + [c.concept_id for c in self.taxonomy.concepts if c.concept_id not in first]
        scores = [10, 7, 4, 2, -1] + [-2 - i for i in range(len(ids) - 5)]
        offset = 100 if self.name == "siglip2" else 0
        return RankedImage([{"concept_id": cid, "raw_score": score + offset,
                             "score_kind": "raw_image_text_logit"}
                            for cid, score in zip(ids, scores, strict=True)], {"total_ms": 1.0})

    def close(self):
        self.closed = True


@pytest.mark.parametrize("model", ["clip", "siglip2"])
def test_collector_schema_signals_groups_and_runtime_reuse(root, tmp_path, monkeypatch, model):
    populate(root)
    before = file_bytes(root)
    runtime = FakeRuntime(model)
    calls = []

    def factory(name):
        calls.append(name)
        return runtime

    monkeypatch.setattr(collector, "create_runtime", factory)
    output = tmp_path / f"{model}.json"
    collector.collect_signals(root, model, "eden-v2", output)
    result = json.loads(output.read_text())
    assert calls == [model] and runtime.loaded and runtime.closed and runtime.rank_calls == 5
    assert result["model"] == model and result["model_id"] == MODELS[model].model_id
    assert result["model_revision"] == MODELS[model].revision
    assert result["annotation_taxonomy_version"] == result["taxonomy_version"] == "eden-v2"
    assert result["schema_version"] == "1.0" and result["candidate_count"] == 78
    summary = result["summary"]
    assert [summary[k] for k in ["image_count", "known_count", "unknown_count", "broad_only_count", "ambiguous_count"]] == [5, 4, 1, 2, 2]
    assert summary["groups"]["known"]["top1_raw_score"]["count"] == 4
    assert summary["groups"]["unknown"]["top1_top2_raw_margin"] == {"count": 1, "min": 3, "median": 3, "p95": 3, "max": 3}
    assert summary["groups_overlap"] is True
    row = result["images"][0]
    assert row["provenance"]["dataset_id"] == "eden-vision-open-set-eval-v1"
    assert row["provenance"]["image_id"] == "eval-001"
    assert row["provenance"]["image_sha256"] == hashlib.sha256((root / "images/0.png").read_bytes()).hexdigest()
    assert row["provenance"]["model"] == model and row["provenance"]["candidate_count"] == 78
    assert row["ground_truth"] == {"expected": [{"concept_id": "cat", "acceptable_ancestors": []}],
                                   "unknown": False, "tags": ["single_object"], "notes": "human note 0"}
    assert row["ranking"]["top_1"] == {"concept_id": "cat", "raw_score": 110 if model == "siglip2" else 10}
    assert row["ranking"]["top_2"]["concept_id"] == "animal"
    assert row["ranking"]["top1_top2_raw_margin"] == 3
    assert len(row["ranking"]["top_5"]) == 5
    assert row["diagnostics"]["top1_top5_raw_spread"] == 11
    assert row["diagnostics"]["top_k_hierarchy_coherence"] == {"k": 5, "related_pair_fraction": 0.6, "unique_broad_categories": 3}
    assert result["thresholds_applied"] is False
    assert "confidence" not in output.read_text().lower()
    assert file_bytes(root) == before


def test_distribution_uses_existing_percentile_rule():
    assert collector.distribution([0, 10, 20]) == {"count": 3, "min": 0, "median": 10, "p95": 19, "max": 20}
    assert collector.distribution([]) == {"count": 0, "min": None, "median": None, "p95": None, "max": None}


def test_unknown_group_empty_is_explicit(root, tmp_path):
    rows = populate(root)
    set_rows(root, [r for r in rows if not r["unknown"]])
    result = collector.collect_signals(root, "clip", "eden-v2", tmp_path / "known.json", runtime_factory=FakeRuntime)
    assert result["summary"]["unknown_count"] == 0
    assert result["summary"]["unknown_status"] == "no_unknown_samples"
    assert result["summary"]["groups"]["unknown"]["status"] == "no_samples"
    assert result["summary"]["groups"]["unknown"]["top1_raw_score"]["p95"] is None


@pytest.mark.parametrize("changes", [
    {"unknown": True}, {"expected": []}, {"tags": ["unknown"]},
    {"unknown": True, "expected": [], "tags": ["ambiguous"]},
    {"tags": ["outdoor"]}, {"expected": [{"concept_id": "not_a_concept"}]},
])
def test_invalid_canonical_rejected_before_runtime_or_output(root, tmp_path, changes):
    rows = populate(root)
    rows[0].update(changes)
    set_rows(root, rows)
    output = tmp_path / "invalid.json"
    with pytest.raises(ManifestValidationError):
        collector.collect_signals(root, "clip", "eden-v2", output,
                                  runtime_factory=lambda name: pytest.fail("must not load"))
    assert not output.exists()


def test_output_overwrite_refused_before_runtime(root, tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("historical report")
    with pytest.raises(FileExistsError):
        collector.collect_signals(root, "clip", "eden-v2", output,
                                  runtime_factory=lambda name: pytest.fail("must not load"))
    assert output.read_text() == "historical report"
    with pytest.raises(FileExistsError):
        collector._publish_new(output, {"new": True})
    assert output.read_text() == "historical report"
    assert not list(tmp_path.glob(".existing.json.*"))


def test_load_failure_closes_runtime_and_publishes_nothing(root, tmp_path):
    populate(root)
    runtime = FakeRuntime("clip")
    def fail():
        raise RuntimeError("MPS unavailable")
    runtime.load = fail
    output = tmp_path / "failed.json"
    with pytest.raises(RuntimeError, match="MPS"):
        collector.collect_signals(root, "clip", "eden-v2", output, runtime_factory=lambda name: runtime)
    assert runtime.closed and not output.exists()


def test_dataset_change_during_collection_refuses_output(root, tmp_path):
    populate(root)
    runtime = FakeRuntime("clip")
    def close():
        runtime.closed = True
        with (root / "manifest.jsonl").open("a") as stream:
            stream.write("\n")
    runtime.close = close
    output = tmp_path / "changed.json"
    with pytest.raises(ManifestValidationError, match="changed"):
        collector.collect_signals(root, "clip", "eden-v2", output, runtime_factory=lambda name: runtime)
    assert not output.exists()


def test_empty_cli_no_ai_imports_and_valid_summary(root, tmp_path):
    script = '''
import sys, runpy
class NoAI:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'transformers', 'huggingface_hub'}:
            raise AssertionError('AI import: ' + fullname)
sys.meta_path.insert(0, NoAI())
path = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(path, run_name='__main__')
'''
    output = tmp_path / "empty.json"
    result = subprocess.run([sys.executable, "-c", script,
        str(Path(collector.__file__).resolve()), "--dataset", str(root), "--model", "siglip2",
        "--taxonomy", "eden-v2", "--output", str(output)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    data = json.loads(output.read_text())
    assert data["status"] == "empty_dataset" and data["model_loaded"] is False
    assert data["images"] == [] and data["summary"]["image_count"] == 0
    assert data["summary"]["unknown_status"] == "no_unknown_samples"
    assert all(group["status"] == "no_samples" for group in data["summary"]["groups"].values())


def test_init_is_empty_valid_and_never_reuses_existing_root(root):
    assert {p.name for p in root.iterdir()} == {"dataset.json", "manifest.jsonl", "manifest.draft.jsonl",
                                               "inbox", "images", "reports", "backups", "rejected"}
    assert load_dataset(root, load_evaluation_taxonomy("eden-v2")).entries == ()
    before = file_bytes(root)
    with pytest.raises(FileExistsError):
        init_dataset(root, "different", load_evaluation_taxonomy("eden-v2"))
    assert before == file_bytes(root)


def test_empty_labeling_ui_and_import_label_promote_reuse(root):
    before = file_bytes(root)
    with TestClient(create_app(root), base_url="http://127.0.0.1") as client:
        assert client.get("/health").status_code == 200
        page = client.get("/")
        assert page.status_code == 200 and "draft 항목이 없습니다." in page.text
        data = client.get("/api/dataset").json()
        assert data["items"] == [] and data["total"] == data["completed"] == data["remaining"] == 0
        assert before == file_bytes(root)
        Image.new("RGB", (4, 3), (11, 22, 33)).save(root / "inbox/fixture.png")
        assert import_inbox(root)["imported"] == 1
        row = client.get("/api/dataset").json()["items"][0]
        response = client.put(f"/api/items/{row['id']}", headers={"X-Labeling-Request": "1"}, json={
            "taxonomy_version": "eden-v2", "revision": row["revision"], "concept_ids": [],
            "unknown": True, "tags": ["ambiguous"], "notes": "synthetic test only"})
        assert response.status_code == 200
        assert response.json()["item"]["tags"] == ["ambiguous", "unknown"]
    taxonomy = load_evaluation_taxonomy("eden-v2")
    assert promote_draft(root, taxonomy)["promoted"] == 1
    entry = load_dataset(root, taxonomy).entries[0]
    assert entry.unknown is True and entry.expected == () and "unknown" in entry.tags
    assert _read_jsonl(root / "manifest.draft.jsonl") == []
