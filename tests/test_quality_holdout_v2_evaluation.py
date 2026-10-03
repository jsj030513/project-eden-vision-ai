"""Frozen semantics tests; never recompute holdout image metrics or write artifacts."""
import copy
import json
import math
from io import BytesIO

import pytest
from PIL import Image
from experiments import evaluate_quality_holdout_v2 as ev
from experiments import image_information_diagnostics as info


@pytest.fixture(scope='module')
def artifacts():
    p = ev.ROOT / 'reports/holdout-v2-frozen-quality-evaluation.json'
    if not p.exists():
        pytest.skip('external one-shot evaluation absent')
    return json.loads(p.read_text()), json.loads((p.parent / 'holdout-v2-image-information-diagnostics.json').read_text())


def test_validation_population_and_exact_unknown(artifacts):
    pre, gt, meta, provenance = ev.validate()
    r, d = artifacts
    assert len(gt) == len(meta) == len(d['candidates']) == 24
    assert {k for k, g in gt.items() if g['unknown']} == ev.UNKNOWN
    assert r['population']['known'] == 17 and r['population']['UNKNOWN'] == 7
    assert all(r['provenance'][k] == v for k, v in provenance.items())
    assert r['rules']['PRIMARY'] == pre['primary_rule']
    assert r['rules']['CHALLENGER'] == pre['challenger_rule']
    assert d['metric_definitions'] == pre['metric_definitions']


@pytest.mark.parametrize('label', ['GT', 'preregistration', 'metric implementation', 'generated'])
def test_sha_validation_rejects_mismatch(tmp_path, label):
    p = tmp_path / label
    p.write_bytes(b'original')
    expected = ev.sha(p)
    assert ev.require_sha(p, expected) == expected
    p.write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA mismatch'):
        ev.require_sha(p, expected)


@pytest.mark.parametrize('rule', ev.NAMES)
def test_exact_boundaries_and_boolean_semantics(rule):
    base = dict(near_black_fraction=0., near_white_fraction=0., std_luminance=100., entropy=8., detail_metric=100.)
    white = .9 if rule == 'PRIMARY' else .5
    for field, cutoff in [('near_black_fraction', .9), ('near_white_fraction', white)]:
        assert any(ev.conditions({**base, field: cutoff}, rule).values())
        assert not any(ev.conditions({**base, field: math.nextafter(cutoff, -math.inf)}, rule).values())
    field, cutoff = ('entropy', 6.) if rule == 'PRIMARY' else ('std_luminance', 30.)
    low = {**base, field: cutoff, 'detail_metric': 2.}
    assert any(ev.conditions(low, rule).values())
    for key, bound in [(field, cutoff), ('detail_metric', 2.)]:
        assert not any(ev.conditions({**low, key: math.nextafter(bound, math.inf)}, rule).values())


def test_confusion_and_raw_rates():
    rows = [{'truth_unknown': t, 'PRIMARY': {'pred_unknown': p}} for t, p in [(True, True), (True, False), (False, True), (False, False), (False, False)]]
    c = ev.confusion(rows, 'PRIMARY')
    assert [c[k] for k in ('TP', 'FN', 'FP', 'TN')] == [1, 1, 1, 2]
    assert c['unknown_recall'] == ev.ratio(1, 2)
    assert c['unknown_precision'] == ev.ratio(1, 2)
    assert c['known_fpr'] == ev.ratio(1, 3)
    assert c['specificity'] == ev.ratio(2, 3)
    assert c['balanced_accuracy'] == (.5 + 2/3) / 2


def test_deterministic_evaluation_and_intended_type_not_gt(artifacts):
    r, d = artifacts
    pre, gt, _, _ = ev.validate()
    first = ev.evaluate(d['candidates'], gt, pre, r['provenance'])
    second = ev.evaluate(d['candidates'], gt, pre, r['provenance'])
    assert first == second
    assert first == {k: v for k, v in r.items() if k != 'artifact_protection'}
    altered = copy.deepcopy(d['candidates'])
    for m in altered:
        m['intended_type'] = 'not_a_truth_label'
    assert ev.evaluate(altered, gt, pre, r['provenance'])['population'] == first['population']
    assert ev.render(first) == ev.render(second)
    assert ev.render(r) == (ev.ROOT / 'reports/holdout-v2-frozen-quality-evaluation.txt').read_text()


def test_deterministic_diagnostics_synthetic_only():
    image = Image.new('RGB', (17, 13))
    image.putdata([(i % 256, (i * 17) % 256, (i * 31) % 256) for i in range(221)])
    b = BytesIO()
    image.save(b, format='PNG')
    assert info.measure_bytes(b.getvalue()) == info.measure_bytes(b.getvalue())


def test_report_counts_rates_and_artifact_preservation(artifacts):
    r, d = artifacts
    for s in [r['population'], *r['by_intended_type'].values(), *r['by_transformation'].values()]:
        for rule in ev.NAMES:
            c = s[rule]
            assert c['TP'] + c['FN'] == s['UNKNOWN']
            assert c['FP'] + c['TN'] == s['known']
            for key in ('unknown_recall', 'unknown_precision', 'known_fpr', 'specificity'):
                a = c[key]
                assert a == ev.ratio(a['numerator'], a['denominator'])
    assert r['provenance']['rules_applied_unchanged'] is True
    for k in ('thresholds_refit', 'metric_semantics_changed', 'semantic_models_used'):
        assert r['provenance'][k] is False
    for path, expected in r['artifact_protection']['sha256_before'].items():
        ev.require_sha(path, expected)
    assert len(d['candidates']) == 24
    with pytest.raises(FileExistsError):
        ev.main()
