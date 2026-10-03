"""No inference: freeze integrity, synthetic score plumbing and source reservations."""
import json
from pathlib import Path

import pytest
from experiments import semantic_identifiability_v3 as v
from experiments.model_runtime import MODELS as EXISTING_MODELS

EXPECTED_PROMPTS = [
    'a blurry photograph with no identifiable subject',
    'an overexposed photograph with no identifiable subject',
    'an underexposed photograph with no identifiable subject',
    'an abstract texture with no identifiable object',
    'an extreme close-up with no identifiable subject',
    'an unrecognizable photograph',
    'a photograph with no clear object or scene',
]


def is_ephemeral(path):
    """Runtime caches/OS metadata explicitly ignored by this repository."""
    path = Path(path)
    return (bool({'__pycache__', '.pytest_cache'} & set(path.parts))
            or path.suffix in {'.pyc', '.pyo'} or path.name == '.DS_Store')


def canonical_fingerprints(fingerprints):
    return {str(Path(path)): digest for path, digest in fingerprints.items()
            if not is_ephemeral(path)}


def assert_protected_fingerprints(expected):
    expected = canonical_fingerprints(expected)
    # Filter before reading: historical cache entries may no longer exist.
    actual = canonical_fingerprints({path: v.sha(path) for path in expected})
    assert actual == expected


@pytest.fixture(scope='module')
def artifacts():
    if not v.OUTPUT.exists():
        pytest.skip('external v3 planning artifacts absent')
    return {p.stem: json.loads(p.read_text()) for p in v.OUTPUT.glob('*.json')}


def test_v2_failed_conclusion_exact_and_spent_policy():
    root = v.DATASETS/'eden-vision-open-set-quality-holdout-v2/reports'
    path = root/'quality-gate-v2-conclusion.json'
    if not path.exists():
        pytest.skip('external conclusion absent')
    c = json.loads(path.read_text())
    e = json.loads((root/'holdout-v2-frozen-quality-evaluation.json').read_text())
    p = json.loads((v.DATASETS/'eden-vision-open-set-quality-calibration-v2/reports/quality-gate-preregistration.json').read_text())
    assert c['status'] == 'failed_independent_validation'
    for name, holdout, calibration in [('primary', [2,5,5,12], [11,3,0,18]), ('challenger', [3,4,5,12], [12,2,1,17])]:
        assert c[name]['passed'] is False
        assert c[name]['holdout_v2_metrics'] == e['population'][name.upper()]
        assert c[name]['calibration_metrics'] == p[name+'_rule']['tuning_metrics']
        assert [c[name]['holdout_v2_metrics'][k] for k in ('TP','FN','FP','TN')] == holdout
        assert [c[name]['calibration_metrics'][k] for k in ('tp','fn','fp','tn')] == calibration
    assert all(c['policy'].values())
    assert c['provenance']['evaluation_sha256'] == v.sha(root/'holdout-v2-frozen-quality-evaluation.json')


def test_prompt_freeze_exact_models_taxonomy_source(artifacts):
    p = artifacts['prompt-preregistration']
    assert p['reject_prompts'] == EXPECTED_PROMPTS and len(set(EXPECTED_PROMPTS)) == 7
    assert p == v.prompt_registration()
    assert v.canonical(p) == v.canonical(v.prompt_registration())
    path = v.OUTPUT/'prompt-preregistration.json'
    assert v.sha(path) == path.with_suffix('.sha256').read_text().strip()
    assert p['taxonomy_sha256'] == v.sha(v.TAXONOMY)
    assert p['source_code_sha256'] == v.sha(v.__file__)
    assert len(p['taxonomy_prompts']) == 78 and p['prompt_set_frozen'] is True
    for key, spec in EXISTING_MODELS.items():
        assert p['models'][key] == {'model_id': spec.model_id, 'revision': spec.revision}
    assert p['spent_holdout_v1_v2_usage'].startswith('historical failure analysis only')


def test_no_inference_no_cutoffs_no_production_change(artifacts):
    def walk(value):
        if isinstance(value, dict):
            for k, x in value.items():
                assert 'threshold' not in k.lower() and 'cutoff' not in k.lower()
                walk(x)
        elif isinstance(value, list):
            for x in value:
                walk(x)
    for key in ('prompt-preregistration', 'source-partition', 'generation-plan'):
        walk(artifacts[key])
    assert artifacts['prompt-preregistration']['inference_run'] is False
    assert artifacts['prompt-preregistration']['production_integration'] is False
    assert artifacts['generation-plan']['images_generated'] is False
    assert artifacts['source-partition']['holdout_images_viewed'] is False
    assert artifacts['source-partition']['holdout_inference_run'] is False
    # The v3 snapshot records a historical no-production-change run, not a
    # permanent freeze of API development. The subsequent authorized schema
    # 1.1 contracts and P2 opt-in lifecycle change only the listed source files.
    # P2 tests separately protect default dummy composition. Keep every
    # dataset artifact, taxonomy and decision policy protected.
    contract_extensions = {v.REPO / name for name in (
        'src/eden_vision/vision/types.py',
        'src/eden_vision/vision/contracts.py',
        'src/eden_vision/schemas/classification.py',
        'src/eden_vision/api/vision.py',
        'src/eden_vision/main.py',
    )}
    assert artifacts['artifact-protection']['production_modified'] is False
    assert_protected_fingerprints({
        path: digest for path, digest in artifacts['artifact-protection']['sha256_before'].items()
        if Path(path) not in contract_extensions
    })
    source = Path(v.__file__).read_text()
    assert 'import torch' not in source and 'import transformers' not in source


@pytest.mark.parametrize('cache_name', [
    '__pycache__/module.cpython-312.pyc', '__pycache__/nested/file',
    'module.pyc', 'module.pyo', '.pytest_cache/v/cache/nodeids', '.DS_Store',
])
def test_historical_cache_changes_and_missing_entries_are_ignored(tmp_path, cache_name):
    source = tmp_path/'protected.py'
    source.write_text('protected = True\n')
    cache = tmp_path/cache_name
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(b'old cache')
    snapshot = {str(path): v.sha(path) for path in (source, cache)}
    original = dict(snapshot)
    cache.write_bytes(b'new cache')
    assert_protected_fingerprints(snapshot)
    actual = {str(path): v.sha(path) for path in (source, cache)}
    actual[str(tmp_path/'new.pyc')] = 'new ephemeral entry'
    assert canonical_fingerprints(actual) == canonical_fingerprints(snapshot)
    cache.unlink()
    assert_protected_fingerprints(snapshot)
    assert snapshot == original


@pytest.mark.parametrize('name', [
    'protected.py', 'dataset/image.png', 'reports/report.json',
    'prompt-preregistration.json', 'generation-plan.json',
])
def test_historical_protected_content_changes_still_fail(tmp_path, name):
    path = tmp_path/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'original protected content')
    snapshot = {str(path): v.sha(path)}
    assert_protected_fingerprints(snapshot)
    path.write_bytes(b'changed protected content')
    with pytest.raises(AssertionError):
        assert_protected_fingerprints(snapshot)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        assert_protected_fingerprints(snapshot)


def test_raw_score_reducer_per_model_only():
    tax = {r['concept_id']: float(i) for i,r in enumerate(v.prompt_registration()['taxonomy_prompts'])}
    reject = {p: float(i) for i,p in enumerate(v.REJECT_PROMPTS)}
    for model in v.MODELS:
        r = v.summarize_raw_scores(model, v.MODELS[model]['revision'], tax, reject)
        assert r['best_taxonomy_raw_score'] == 77
        assert r['best_reject_raw_score'] == 6
        assert r['taxonomy_reject_gap'] == 71
        assert r['taxonomy_top1_top2_margin'] == 1
        assert len(r['taxonomy_top5']) == 5
        assert not {'pred_unknown', 'classification', 'threshold'} & r.keys()
        with pytest.raises(ValueError):
            v.summarize_raw_scores(model, 'main', tax, reject)
        with pytest.raises(ValueError):
            v.summarize_raw_scores(model, v.MODELS[model]['revision'], tax, {})
        invalid = {**tax, next(iter(tax)): float('nan')}
        with pytest.raises(ValueError):
            v.summarize_raw_scores(model, v.MODELS[model]['revision'], invalid, reject)


def source_rows(tmp_path):
    rows = []
    for i in range(20):
        p = tmp_path/f'{i}.bin'
        p.write_bytes(f'synthetic source {i}'.encode())
        rows.append({'source_id': f'new:{i}', 'source_path': str(p), 'source_sha256': v.sha(p)})
    return rows[:10], rows[10:]


def test_source_partition_disjointness_and_historical_rejection(tmp_path):
    cal, hold = source_rows(tmp_path)
    inventory = {'excluded_ids': [], 'excluded_sha256': []}
    p = v.reserve_sources(cal, hold, inventory)
    assert p['historical_source_overlap'] == p['calibration_holdout_overlap'] == 0
    assert not {r['source_id'] for r in cal} & {r['source_id'] for r in hold}
    assert not {r['source_sha256'] for r in cal} & {r['source_sha256'] for r in hold}
    with pytest.raises(ValueError, match='overlap'):
        v.reserve_sources(cal, cal, inventory)
    for key, value in [('excluded_ids', cal[0]['source_id']), ('excluded_sha256', hold[0]['source_sha256'])]:
        with pytest.raises(ValueError, match='overlap'):
            v.reserve_sources(cal, hold, {**inventory, key: [value]})
    Path(cal[0]['source_path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA mismatch'):
        v.reserve_sources(cal, hold, inventory)


def test_actual_reservations_not_fabricated_and_balanced_plan(artifacts):
    partition = artifacts['source-partition']
    assert partition['status'] == 'blocked_missing_new_sources'
    assert partition['reservation_complete'] is False
    assert partition['calibration_v3'] == partition['future_holdout_v3'] == []
    assert partition['historical_source_overlap'] is None
    assert partition['calibration_holdout_overlap'] is None
    plan = artifacts['generation-plan']
    assert plan == v.generation_plan(partition)
    assert plan['planned_total'] == 40 and plan['target_known'] == plan['target_UNKNOWN'] == 20
    for f in v.FAMILIES:
        rows = [r for r in plan['candidates'] if r['transformation_family'] == f]
        assert len(rows) == 8
        assert sum(r['recruitment_target'] == 'known' for r in rows) == 4
        assert all(r['human_gt'] is None and r['source_id'] is None for r in rows)
    inventory = artifacts['historical-source-exclusion']
    assert all(f'baseline:eval-{i:03d}' in inventory['excluded_ids'] for i in range(1,57))
    for path,digest in inventory['evidence_files'].items():
        assert v.sha(path) == digest


def test_freeze_refuses_overwrite(tmp_path):
    path = tmp_path/'freeze.json'
    v.write_new(path, {'frozen': True})
    with pytest.raises(FileExistsError):
        v.write_new(path, {'frozen': False})
