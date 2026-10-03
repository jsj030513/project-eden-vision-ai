"""Runner tests use temporary fixtures and fakes; no AI packages required."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from experiments.evaluate_models import (
    build_parser, evaluation_matrix, make_plan, report_filename, run_evaluations,
)
from experiments.model_runtime import (
    MODELS, RankedImage, cache_identity, create_runtime, load_evaluation_taxonomy,
)


def dataset(root, *, version="eden-v1", labels=("rose",)):
    root.mkdir()
    (root / "images").mkdir()
    entries = []
    for index, label in enumerate(labels):
        with Image.new("RGB", (3, 2), (index, 40, 70)) as image:
            image.save(root / "images" / f"{index}.png")
        entries.append({"id": str(index), "file": f"images/{index}.png",
                        "expected": [] if label is None else [{"concept_id": label}],
                        "unknown": label is None,
                        "tags": ["unknown"] if label is None else ["single_object"], "notes": None})
    (root / "dataset.json").write_text(json.dumps({"dataset_id": "runner-test", "taxonomy_version": version,
                                                 "image_count": len(entries)}))
    (root / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in entries))
    return root


@pytest.mark.parametrize("model,taxonomy,expected", [
    ("clip", "eden-v1", 1), ("siglip2", "eden-v2", 1), ("all", "eden-v1", 2),
    ("clip", "all", 2), ("all", "all", 4), ("both", "all", 4),
])
def test_selection_and_report_uniqueness(model, taxonomy, expected):
    args = build_parser().parse_args(["--model", model, "--taxonomy", taxonomy])
    matrix = evaluation_matrix(args.model, args.taxonomy)
    assert len(matrix) == expected
    assert len({report_filename(*item) for item in matrix}) == expected
    if model == "all" and taxonomy == "all":
        assert matrix == (("clip", "eden-v1"), ("clip", "eden-v2"),
                          ("siglip2", "eden-v1"), ("siglip2", "eden-v2"))


@pytest.mark.parametrize("flag,value", [("--model", "llm"), ("--taxonomy", "eden-v3")])
def test_invalid_selection_rejected(flag, value):
    with pytest.raises(SystemExit):
        build_parser().parse_args([flag, value])


@pytest.mark.parametrize("flags,labels", [
    (["--validate-only"], ("rose",)), (["--dry-run"], ("rose",)), ([], ()),
])
def test_cli_never_imports_ai_or_writes_reports_on_non_inference_paths(tmp_path, flags, labels):
    root = dataset(tmp_path / "dataset", labels=labels)
    script = '''
import importlib.abc, runpy, sys
class NoAI(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'transformers', 'huggingface_hub'}:
            raise AssertionError('AI import attempted: ' + fullname)
sys.meta_path.insert(0, NoAI())
path = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(path, run_name='__main__')
'''
    result = subprocess.run([sys.executable, "-c", script,
                             str(Path("experiments/evaluate_models.py").resolve()),
                             "--dataset", str(root), "--model", "all", "--taxonomy", "all", *flags],
                            cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "model not loaded" in result.stdout
    assert "siglip2 / eden-v2" in result.stdout
    assert not (root / "reports").exists()


def test_cache_identity_has_all_inputs_and_preserves_order():
    v1 = load_evaluation_taxonomy("eden-v1")
    identity = cache_identity(MODELS["clip"], v1)
    assert identity == cache_identity(MODELS["clip"], load_evaluation_taxonomy("eden-v1"))
    assert identity.candidate_ids == tuple(c.concept_id for c in v1.concepts)
    assert identity.prompts[0] == "a photo of a living thing"
    assert identity != cache_identity(MODELS["siglip2"], v1)
    assert identity != cache_identity(replace(MODELS["clip"], revision="changed"), v1)
    assert identity != cache_identity(MODELS["clip"], replace(v1, concepts=tuple(reversed(v1.concepts))))
    renamed = replace(v1.concepts[0], display_name="Changed name")
    assert identity != cache_identity(MODELS["clip"], replace(v1, concepts=(renamed, *v1.concepts[1:])))


@pytest.mark.parametrize("model", ["clip", "siglip2"])
def test_prepare_reuses_identity_and_replaces_cache_without_ai(monkeypatch, model):
    runtime = create_runtime(model)
    runtime.model = object()  # stand-in for loaded weights
    calls = []
    def encode(prompts):
        calls.append(prompts)
        return object(), {"bytes": 0}
    monkeypatch.setattr(runtime, "_encode_taxonomy", encode)
    v1, v2 = load_evaluation_taxonomy("eden-v1"), load_evaluation_taxonomy("eden-v2")
    runtime.prepare_taxonomy(v1)
    feature = runtime._features
    runtime.prepare_taxonomy(v1)
    assert len(calls) == 1 and runtime._features is feature
    runtime.prepare_taxonomy(v2)
    assert len(calls) == 2 and runtime._features is not feature
    runtime.spec = replace(runtime.spec, revision="different")
    runtime.prepare_taxonomy(v2)
    assert len(calls) == 3
    def fail(prompts):
        raise RuntimeError("text encoding failed")
    monkeypatch.setattr(runtime, "_encode_taxonomy", fail)
    with pytest.raises(RuntimeError):
        runtime.prepare_taxonomy(v1)
    assert runtime.identity is None and runtime._features is None
    runtime.close()
    with pytest.raises(RuntimeError, match="load"):
        runtime.rank(None)


def test_rank_requires_prepared_taxonomy():
    runtime = create_runtime("siglip2")
    with pytest.raises(RuntimeError, match="load"):
        runtime.prepare_taxonomy(load_evaluation_taxonomy("eden-v1"))
    runtime.model = object()
    with pytest.raises(RuntimeError, match="prepare_taxonomy"):
        runtime.rank(None)
    runtime.close()


class FakeRuntime:
    def __init__(self, name, events, failure=None):
        self.name, self.events, self.failure = name, events, failure
        self.metadata = {"model_id": MODELS[name].model_id, "model_revision": MODELS[name].revision}

    def load(self):
        self.events.append((self.name, "load"))
        if self.failure == "load":
            raise RuntimeError("load failure")

    def prepare_taxonomy(self, taxonomy):
        self.taxonomy = taxonomy
        self.events.append((self.name, taxonomy.taxonomy_version))
        return {"identity": {"taxonomy_version": taxonomy.taxonomy_version}}

    def rank(self, image):
        if self.failure == "rank":
            raise RuntimeError("rank failure")
        rows = [{"concept_id": c.concept_id, "raw_score": 1.0 if c.concept_id == "plant" else 0.0,
                 "score_kind": "raw_image_text_logit"} for c in self.taxonomy.concepts]
        rows.sort(key=lambda row: (-row["raw_score"], row["concept_id"]))
        return RankedImage(rows, {"total_ms": 1.0})

    def memory(self):
        return {"rss_mib": 1.0, "mps_allocated_mib": 1.0, "mps_driver_mib": 1.0}

    def close(self):
        self.events.append((self.name, "close"))


def test_matrix_execution_reports_gt_and_sequential_cleanup(tmp_path):
    root = dataset(tmp_path / "dataset", version="eden-v2", labels=("shrub", None))
    plan = make_plan(root, "all", "all")
    events = []
    reports = run_evaluations(plan, runtime_factory=lambda name: FakeRuntime(name, events))
    assert events == [(m, action) for m in ("clip", "siglip2") for action in ("load", "eden-v1", "eden-v2", "close")]
    assert len(reports) == 4
    for report in reports:
        assert report["image_count"] == 2
        assert report["images"][0]["expected"] == ["shrub"]
        assert report["images"][0]["hierarchy_relations"]["shrub"] == "ancestor_distance_1"
        assert report["images"][1]["unknown_ground_truth"] is True
        assert not report["unknown_detection"]["evaluated"]
        assert report["metrics"]["top_k_exact"]["top_1"]["total"] == 1
        assert report["truth_concepts_outside_candidates"] == (["shrub"] if report["taxonomy_version"] == "eden-v1" else [])
    comparison = json.loads((root / "reports/model-taxonomy-comparison.json").read_text())
    assert len(comparison["conditions"]) == 4 and "winner" not in comparison
    assert len(list((root / "reports").glob("*.json"))) == 5
    before = list(events)
    with pytest.raises(FileExistsError):
        run_evaluations(plan, runtime_factory=lambda name: FakeRuntime(name, events))
    assert events == before


@pytest.mark.parametrize("failure", ["load", "rank"])
def test_failure_closes_runtime_and_never_loads_second_model(tmp_path, failure):
    plan = make_plan(dataset(tmp_path / "dataset"), "all", "all")
    events = []
    with pytest.raises(RuntimeError, match="failure"):
        run_evaluations(plan, runtime_factory=lambda name: FakeRuntime(name, events, failure))
    assert events[-1] == ("clip", "close")
    assert all(name == "clip" for name, _ in events)


def test_empty_plan_has_no_runtime_creation(tmp_path):
    plan = make_plan(dataset(tmp_path / "dataset", labels=()), "all", "all")
    def forbidden(name):
        pytest.fail("runtime factory called for empty dataset")
    assert run_evaluations(plan, runtime_factory=forbidden) == []


def test_annotation_version_remains_strict(tmp_path):
    # Selecting v2 candidates must not allow v2-only labels under declared v1 GT.
    root = dataset(tmp_path / "dataset", version="eden-v1", labels=("shrub",))
    with pytest.raises(ValueError, match="unknown expected concept"):
        make_plan(root, "clip", "eden-v2")
