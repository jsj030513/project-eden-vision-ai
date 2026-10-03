"""Verify the independent, immutable quality-gate preregistration artifact."""
import hashlib
import json
from pathlib import Path

import pytest

PATH = Path('/Users/dev/Datasets/eden-vision-open-set-quality-calibration-v2/reports/quality-gate-preregistration.json')


@pytest.fixture(scope='module')
def artifact():
    if not PATH.exists():
        pytest.skip('external quality preregistration not installed')
    return json.loads(PATH.read_bytes())


@pytest.mark.parametrize('key,name,expression,thresholds,counts,fp,fn', [
    ('primary_rule', 'Q2-00623-conservative-primary',
     'near_black_fraction >= 0.90 OR near_white_fraction >= 0.90 OR (detail_metric <= 2.0 AND entropy <= 6.0)',
     {'near_black_fraction': .9, 'near_white_fraction': .9, 'detail_metric': 2., 'entropy': 6.},
     (11, 3, 0, 18), [], ['quality-v2-003', 'quality-v2-005', 'quality-v2-011']),
    ('challenger_rule', 'Q3-00537-high-recall-challenger',
     'near_black_fraction >= 0.90 OR near_white_fraction >= 0.50 OR (std_luminance <= 30.0 AND detail_metric <= 2.0)',
     {'near_black_fraction': .9, 'near_white_fraction': .5, 'std_luminance': 30., 'detail_metric': 2.},
     (12, 2, 1, 17), ['quality-v2-032'], ['quality-v2-005', 'quality-v2-011']),
])
def test_exact_rules_and_metrics(artifact, key, name, expression, thresholds, counts, fp, fn):
    r = artifact[key]
    assert r['name'] == name
    assert r['expression'] == expression
    assert r['thresholds'] == thresholds
    assert r['boolean_expression'] == {'or': [
        {'gte': ['near_black_fraction', thresholds['near_black_fraction']]},
        {'gte': ['near_white_fraction', thresholds['near_white_fraction']]},
        {'and': [{'lte': [k, v]} for k, v in thresholds.items()
                 if k not in ('near_black_fraction', 'near_white_fraction')]},
    ]}
    m = r['tuning_metrics']
    assert tuple(m[k] for k in ('tp', 'fn', 'fp', 'tn')) == counts
    assert m['unknown_recall'] == counts[0] / 14
    assert m['known_fpr'] == counts[2] / 18
    assert m['specificity'] == counts[3] / 18
    assert r['fp_ids'] == fp and r['fn_ids'] == fn


def test_state_and_population(artifact):
    assert artifact['calibration_population'] == {'total': 32, 'known': 18, 'unknown': 14}
    assert artifact['status'] == 'frozen_before_holdout_v2'
    for key in ('rules_frozen', 'threshold_refit_after_holdout_forbidden',
                'metric_semantics_change_after_holdout_forbidden',
                'metric_implementation_frozen_through_holdout_v2_evaluation'):
        assert artifact[key] is True
    for key in ('holdout_v2_seen', 'holdout_v2_inference_run', 'semantic_rule_combined'):
        assert artifact[key] is False


def test_sources_and_metric_implementation_hashes(artifact):
    sources = (artifact['source_sweep_reports'] + artifact['human_review_artifacts']
               + [artifact['image_information_diagnostic_report']]
               + artifact['source_code'] + [artifact['decoder_source']])
    assert {Path(r['path']).name for r in artifact['source_code']} == {
        'image_information_diagnostics.py', 'sweep_quality_gate_rules.py'}
    assert len(artifact['source_sweep_reports']) == 2
    assert len(artifact['human_review_artifacts']) == 2
    for source in sources:
        assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() == source['sha256']
    from experiments.image_information_diagnostics import DEFINITIONS
    assert artifact['metric_definitions'] == DEFINITIONS
    assert DEFINITIONS['fixed_metric_constants']['dark_lte'] == 10
    assert DEFINITIONS['fixed_metric_constants']['bright_gte'] == 245
    assert DEFINITIONS['fixed_metric_constants']['analysis_max_side'] == 512


def test_detached_sha_and_deterministic_serialization(artifact):
    raw = PATH.read_bytes()
    canonical = (json.dumps(artifact, ensure_ascii=False, sort_keys=True,
                            indent=2, allow_nan=False) + '\n').encode()
    assert raw == canonical
    assert hashlib.sha256(raw).hexdigest() == PATH.with_suffix('.sha256').read_text().strip()
    assert hashlib.sha256(canonical).hexdigest() == hashlib.sha256(raw).hexdigest()
