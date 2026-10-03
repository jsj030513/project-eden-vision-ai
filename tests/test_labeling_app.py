from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import shutil
import subprocess
import sys

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from experiments.eval_dataset import CANONICAL_FIELDS, _read_jsonl, _write_jsonl
from experiments.evaluation_core import ALLOWED_TAGS, ManifestValidationError, validate_dataset_rows
from experiments.labeling_app import create_app
from experiments.model_runtime import load_evaluation_taxonomy


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "images").mkdir()
    (root / "dataset.json").write_text(json.dumps({
        "dataset_id": "test", "taxonomy_version": "eden-v2", "image_count": 0,
    }))
    (root / "manifest.jsonl").write_text("\n")
    rows = []
    for i in range(1, 4):
        item_id = f"eval-{i:03}"
        Image.new("RGB", (8, 6), (i, 2, 3)).save(root / "images" / f"{item_id}.jpg")
        rows.append({"id": item_id, "file": f"images/{item_id}.jpg",
                     "source_filename": f"원본 {i}.jpg", "taxonomy_version": "eden-v2",
                     "expected": None, "unknown": None, "tags": [], "notes": None,
                     "sha256": hashlib.sha256((root / "images" / f"{item_id}.jpg").read_bytes()).hexdigest(),
                     "width": 8, "height": 6,
                     "format": "JPEG", "file_size_bytes": 0})
    rows[0].update(expected=[{"concept_id": "cat", "acceptable_ancestors": ["living_thing", "animal"]}],
                   unknown=False, tags=["single_object", "broad_only"], notes="Human's original note")
    _write_jsonl(root / "manifest.draft.jsonl", rows)
    return root


@pytest.fixture
def client(dataset):
    with TestClient(create_app(dataset), base_url="http://127.0.0.1") as client:
        yield client


def payload(client, item_id="eval-002", **changes):
    row = next(r for r in client.get("/api/dataset").json()["items"] if r["id"] == item_id)
    return {"taxonomy_version": "eden-v2", "revision": row["revision"],
            "concept_ids": ["cat"], "unknown": False, "tags": ["single_object"],
            "notes": "human label", **changes}


def save(client, data=None, item_id="eval-002"):
    return client.put(f"/api/items/{item_id}", json=data if data is not None else payload(client, item_id),
                      headers={"X-Labeling-Request": "1"})


def test_health_page_and_read_only_startup(client, dataset):
    before = {p.name: p.read_bytes() for p in dataset.iterdir() if p.is_file()}
    assert client.get("/health").json() == {"status": "ok", "taxonomy_version": "eden-v2"}
    page = client.get("/")
    assert page.status_code == 200 and "Eden Vision Ground Truth" in page.text
    assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
    assert "cdn" not in page.text.lower()
    assert client.get("/api/dataset").status_code == 200
    assert client.get("/api/concepts").status_code == 200
    assert {p.name: p.read_bytes() for p in dataset.iterdir() if p.is_file()} == before


def test_dataset_listing_preserves_labels_and_starts_at_first_incomplete(client):
    result = client.get("/api/dataset").json()
    assert (result["total"], result["completed"], result["remaining"], result["initial_index"]) == (3, 1, 2, 1)
    assert result["items"][0]["notes"] == "Human's original note"
    assert result["items"][0]["tags"] == ["single_object", "broad_only"]
    assert set(result["allowed_tags"]) == ALLOWED_TAGS
    assert result["items"][1]["source_filename"] == "원본 2.jpg"


def test_concept_search_by_id_name_and_path(client):
    concepts = client.get("/api/concepts").json()["concepts"]
    assert len(concepts) == 78
    cat = next(c for c in concepts if c["concept_id"] == "cat")
    assert cat["path"] == ["living_thing", "animal", "cat"]
    for query in ["cat", cat["display_name"].upper()]:
        assert cat in client.get("/api/concepts", params={"q": query}).json()["concepts"]
    assert client.get("/api/concepts", params={"q": "missing-concept"}).json()["concepts"] == []


def test_save_ancestors_and_preserve_other_drafts(client, dataset):
    path = dataset / "manifest.draft.jsonl"
    original = _read_jsonl(path)
    canonical = (dataset / "manifest.jsonl").read_bytes()
    metadata = (dataset / "dataset.json").read_bytes()
    response = save(client)
    assert response.status_code == 200
    rows = _read_jsonl(path)
    assert len(rows) == 3 and rows[0] == original[0] and rows[2] == original[2]
    assert rows[1]["expected"] == [{"concept_id": "cat", "acceptable_ancestors": ["living_thing", "animal"]}]
    for field in ["id", "file", "source_filename", "sha256", "width", "height", "format", "file_size_bytes"]:
        assert rows[1][field] == original[1][field]
    assert rows[1]["notes"] == "human label"
    assert (dataset / "manifest.jsonl").read_bytes() == canonical
    assert (dataset / "dataset.json").read_bytes() == metadata
    assert client.get("/api/dataset").json()["completed"] == 2


def test_multi_label_and_edit_completed(client, dataset):
    result = save(client, payload(client, "eval-001", concept_ids=["car", "road", "bus"],
                                  tags=["multi_object", "scene"]), "eval-001")
    assert result.status_code == 200
    assert result.json()["item"]["tags"] == ["multi_object", "scene"]
    expected = _read_jsonl(dataset / "manifest.draft.jsonl")[0]["expected"]
    assert [e["concept_id"] for e in expected] == ["car", "road", "bus"]
    assert expected[0]["acceptable_ancestors"] == ["artifact", "vehicle"]
    assert_canonical(dataset, result.json()["item"])


def test_unknown_clears_expected_and_adds_tag(client, dataset):
    result = save(client, payload(client, unknown=True, tags=[]))
    assert result.status_code == 200
    row = result.json()["item"]
    assert row["expected"] == [] and row["unknown"] is True and row["tags"] == ["unknown"]
    assert_canonical(dataset, row)


@pytest.mark.parametrize("changes", [
    {"concept_ids": ["not_a_concept"]}, {"concept_ids": []}, {"tags": []},
    {"tags": [" ", ""]}, {"taxonomy_version": "eden-v1"}, {"unknown": "false"},
    {"tags": ["unknown"]}, {"tags": ["single_object", "unknown"]},
    {"tags": ["single_object", "multi_object"]},
    {"unknown": True, "tags": ["single_object", "multi_object"]},
])
def test_invalid_save_is_rejected_without_writes(client, dataset, changes):
    path = dataset / "manifest.draft.jsonl"
    before = path.read_bytes()
    assert save(client, payload(client, **changes)).status_code == 422
    assert path.read_bytes() == before


def assert_canonical(dataset, row):
    metadata = json.loads((dataset / "dataset.json").read_text())
    validated = validate_dataset_rows(
        dataset, {**metadata, "image_count": 1},
        [{key: row[key] for key in CANONICAL_FIELDS}], load_evaluation_taxonomy("eden-v2"),
    )
    assert validated.entries[0].tags == tuple(row["tags"])


@pytest.mark.parametrize("tag", [
    "multiple_objects", "outdoor", "animal", "vehicle", "food", "drink", "artifact",
    "indoor", "product_image", "urban", "wildlife", "difficult", "custom-tag", " scene ",
])
@pytest.mark.parametrize("unknown", [False, True])
def test_noncanonical_tags_rejected(client, dataset, tag, unknown):
    path = dataset / "manifest.draft.jsonl"
    before = path.read_bytes()
    assert save(client, payload(client, tags=[tag], unknown=unknown)).status_code == 422
    assert path.read_bytes() == before


@pytest.mark.parametrize("tag", sorted(ALLOWED_TAGS))
def test_every_canonical_tag_can_be_saved(client, dataset, tag):
    response = save(client, payload(client, tags=[tag], unknown=tag == "unknown"))
    assert response.status_code == 200
    row = response.json()["item"]
    assert row["tags"] == [tag]
    assert_canonical(dataset, row)


@pytest.mark.parametrize("tags", [["single_object", "broad_only"], ["multi_object", "scene"]])
def test_existing_allowed_tags_preserved_when_reopened_and_saved(client, dataset, tags):
    path = dataset / "manifest.draft.jsonl"
    rows = _read_jsonl(path)
    rows[0]["tags"] = tags
    _write_jsonl(path, rows)
    before = path.read_bytes()
    reopened = client.get("/api/dataset").json()["items"][0]
    assert reopened["tags"] == tags
    assert path.read_bytes() == before
    data = payload(client, "eval-001", tags=reopened["tags"], notes=reopened["notes"])
    assert save(client, data, "eval-001").status_code == 200
    assert _read_jsonl(path) == rows


def test_canonical_validation_failure_does_not_write(client, dataset):
    # Canonical validation also protects against a changed registered image.
    before = (dataset / "manifest.draft.jsonl").read_bytes()
    (dataset / "images/eval-002.jpg").write_bytes(b"changed after import")
    response = save(client)
    assert response.status_code == 422
    assert "SHA-256" in response.json()["detail"]
    assert (dataset / "manifest.draft.jsonl").read_bytes() == before


def test_ui_tag_selection_and_unknown_sync(client, tmp_path):
    """Run the shipped JS against a minimal DOM; no browser or npm dependency."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed for the optional JavaScript interaction check")
    html = client.get("/").text
    assert 'id="tags"' not in html  # No free-text tags control.
    script = html.split("<script>", 1)[1].split("</script>", 1)[0]
    data = client.get("/api/dataset").json()
    taxonomy = client.get("/api/concepts").json()
    harness = r'''
const assert = require('node:assert/strict');
class Element {
  constructor() {
    this.children = []; this.events = {}; this.attributes = {};
    this.value = ''; this.checked = false; this.disabled = false;
    this.classList = {toggle() {}};
  }
  addEventListener(name, fn) { this.events[name] = fn; }
  setAttribute(key, value) { this.attributes[key] = value; }
  append(child) { this.children.push(child); }
  replaceChildren() { this.children = []; }
  get childElementCount() { return this.children.length; }
}
const elements = new Map();
const document = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  },
  createElement() { return new Element(); },
  addEventListener() {}
};
const window = {addEventListener() {}, confirm() { return true; }};
const fetch = async url => ({ok: true, json: async () => responses[url]});
'''
    checks = r'''
setImmediate(() => {
  try {
    assert.equal($('message').textContent || '', '');
    const choices = () => $('tagChoices').children;
    const choice = tag => choices().find(b => b.textContent === tag);
    const pressed = () => choices().filter(b => b.attributes['aria-pressed'] === 'true').map(b => b.textContent).sort();
    const click = tag => choice(tag).events.click();
    assert.deepEqual(choices().map(b => b.textContent).sort(), responses['/api/dataset'].allowed_tags);
    assert.equal(choices().length, 7);
    assert(choices().every(b => b.title && b.attributes['aria-description']));
    show(0);
    assert.deepEqual(pressed(), ['broad_only', 'single_object']);
    assert.equal($('save').disabled, false);
    click('multi_object');
    assert.deepEqual(pressed(), ['broad_only', 'multi_object']);
    click('single_object');
    assert.deepEqual(pressed(), ['broad_only', 'single_object']);
    click('unknown');
    assert.equal($('unknown').checked, true);
    assert.equal(selected.size, 0);
    assert.equal($('search').disabled, true);
    assert(pressed().includes('unknown'));
    $('unknown').checked = false;
    $('unknown').events.change();
    assert(!pressed().includes('unknown'));
    assert.equal($('save').disabled, true); // No expected concepts after Unknown.
    $('unknown').checked = true;
    $('unknown').events.change();
    click('unknown');
    assert.equal($('unknown').checked, false);
    assert(!pressed().includes('unknown'));
    items[0].tags = ['multi_object', 'scene'];
    show(0);
    assert.deepEqual(pressed(), ['multi_object', 'scene']);
    click('multi_object'); click('scene');
    assert.equal($('save').disabled, true); // Known sample requires a tag.
    click('scene');
    assert.equal($('save').disabled, false);
  } catch (error) { console.error(error); process.exitCode = 1; }
});
'''
    path = tmp_path / "tag-ui-check.cjs"
    path.write_text("const responses = " + json.dumps({"/api/dataset": data, "/api/concepts": taxonomy})
                    + ";\n" + harness + script + checks)
    result = subprocess.run([node, str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_stale_tab_rejected_and_other_rows_reloaded(client, dataset):
    stale = payload(client)
    assert save(client).status_code == 200
    path = dataset / "manifest.draft.jsonl"
    before = path.read_bytes()
    assert save(client, stale).status_code == 409
    assert path.read_bytes() == before
    rows = _read_jsonl(path)
    rows[0]["notes"] = "Edited by another human after startup"
    _write_jsonl(path, rows)
    assert save(client).status_code == 200
    assert _read_jsonl(path)[0] == rows[0]


def test_concurrent_instances_preserve_both_saves(client, dataset):
    with TestClient(create_app(dataset), base_url="http://127.0.0.1") as other:
        first = payload(client, "eval-002")
        second = payload(other, "eval-003")
        with ThreadPoolExecutor(max_workers=2) as pool:
            a = pool.submit(save, client, first, "eval-002")
            b = pool.submit(save, other, second, "eval-003")
            assert a.result().status_code == b.result().status_code == 200
    assert all(r["expected"] for r in _read_jsonl(dataset / "manifest.draft.jsonl"))


def test_atomic_replace_failure_preserves_entire_manifest(client, dataset, monkeypatch):
    import experiments.eval_dataset as module
    path = dataset / "manifest.draft.jsonl"
    before = path.read_bytes()

    def fail_replace(*args):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(module.os, "replace", fail_replace)
    assert save(client).status_code == 500
    assert path.read_bytes() == before
    assert not list(dataset.glob(".manifest.draft.jsonl.*"))


def test_image_only_serves_registered_id(client, dataset):
    response = client.get("/api/images/eval-001")
    assert response.status_code == 200
    assert response.content == (dataset / "images/eval-001.jpg").read_bytes()
    for url in ["/api/images/missing", "/api/images/%2e%2e%2fdataset.json",
                "/api/images/%252e%252e%252fsecret", "/images/eval-001.jpg",
                "/api/images/Users/dev/secret"]:
        assert client.get(url).status_code == 404


@pytest.mark.parametrize("filename", ["../secret.jpg", "images/../../secret.jpg", "/etc/passwd", "images\\secret.jpg"])
def test_malicious_manifest_paths_rejected(client, dataset, filename):
    path = dataset / "manifest.draft.jsonl"
    rows = _read_jsonl(path)
    rows[0]["file"] = filename
    _write_jsonl(path, rows)
    assert client.get("/api/images/eval-001").status_code == 404


def test_symlink_escape_rejected(client, dataset):
    secret = dataset.parent / "secret.jpg"
    secret.write_bytes(b"private data")
    target = dataset / "images/eval-001.jpg"
    target.unlink()
    target.symlink_to(secret)
    assert client.get("/api/images/eval-001").status_code == 404


def test_taxonomy_change_after_startup_and_bad_startup(client, dataset):
    metadata = dataset / "dataset.json"
    request = payload(client)
    metadata.write_text(json.dumps({"taxonomy_version": "eden-v1"}))
    before = (dataset / "manifest.draft.jsonl").read_bytes()
    assert save(client, request).status_code == 409
    assert (dataset / "manifest.draft.jsonl").read_bytes() == before
    with pytest.raises(ManifestValidationError, match="taxonomy"):
        create_app(dataset)


def test_row_taxonomy_mismatch_rejected(client, dataset):
    request = payload(client)
    path = dataset / "manifest.draft.jsonl"
    rows = _read_jsonl(path)
    rows[0]["taxonomy_version"] = "eden-v1"
    _write_jsonl(path, rows)
    before = path.read_bytes()
    assert save(client, request).status_code == 409
    assert path.read_bytes() == before


def test_cross_origin_and_untrusted_host_rejected(client):
    data = payload(client)
    assert client.put("/api/items/eval-002", json=data).status_code == 403
    assert client.put("/api/items/eval-002", json=data, headers={
        "X-Labeling-Request": "1", "Origin": "https://external.example",
    }).status_code == 403
    assert client.get("/api/dataset", headers={"Host": "external.example"}).status_code == 400


def test_import_and_app_do_not_load_ml_packages(dataset):
    code = '''
import sys
class NoML:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'transformers', 'clip', 'open_clip'}:
            raise AssertionError('forbidden ML import: ' + fullname)
sys.meta_path.insert(0, NoML())
from experiments.labeling_app import create_app
from fastapi.testclient import TestClient
with TestClient(create_app(sys.argv[1]), base_url='http://127.0.0.1') as client:
    assert client.get('/api/concepts').status_code == 200
    assert client.get('/api/images/eval-001').status_code == 200
'''
    subprocess.run([sys.executable, "-c", code, str(dataset)], check=True, capture_output=True, text=True)
