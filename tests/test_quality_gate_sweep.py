from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiments import sweep_quality_gate_rules as q


@pytest.fixture
def population_root(tmp_path):
    (tmp_path / 'reports').mkdir()
    (tmp_path / 'generated').mkdir()
    reviews, images = [], []
    for n, key in enumerate(q.IDS, 1):
        raw = f'raw image fixture {n}'.encode()
        (tmp_path / 'generated' / f'{key}.jpg').write_bytes(raw)
        unknown = n <= 16 and n not in (4, 12)
        transform = q.UNKNOWN_TRANSFORMS[(n - 1) % 8] if n <= 16 else ('moderate_defocus' if n % 2 else 'moderate_underexposure')
        source = f'baseline:eval-{((n - 1) % 16) // 2 + 1:03d}'
        review = {'candidate_id': key, 'source_id': source, 'transformation': transform,
                  'generated_sha256': q.sha(raw), 'unknown': unknown,
                  'expected': [] if unknown else ['outdoor_scene'],
                  'tags': ['unknown', 'ambiguous'] if unknown else ['ambiguous'],
                  'notes': f'human notes {n}', 'review_method': 'human_visual_review_without_model_signals'}
        reviews.append(review)
        values = {'mean_luminance': 100., 'std_luminance': float(n),
                  'near_black_fraction': .99 if n == 1 else 0.,
                  'near_white_fraction': .99 if n == 2 else 0.,
                  'percentile_01': 50., 'percentile_99': 50. + n * 2,
                  'dynamic_range': n * 3., 'entropy': min(n / 4, 8.), 'detail_metric': n / 5,
                  'edge_density': .1, 'saturation_mean': .2, 'saturation_std': .1}
        images.append({'candidate_id': key, 'source_id': source, 'transform': transform,
                       'generated_sha256': q.sha(raw), 'origin': 'quality_calibration_v2_candidate',
                       'intended_type': 'deliberately_wrong_not_truth', **values})
    for name, rows in zip(q.INPUT_NAMES[:2], (reviews[:16], reviews[16:]), strict=True):
        (tmp_path / name).write_text(''.join(json.dumps(row) + '\n' for row in rows))
    (tmp_path / q.INPUT_NAMES[2]).write_text(json.dumps({'row_count': 32, 'images': images}))
    return tmp_path


def test_population_and_human_only_truth(population_root):
    rows, hashes = q.load_population(population_root)
    assert len(rows) == 32 and sum(r['truth_unknown'] for r in rows) == 14
    assert len(hashes) == 3
    assert rows[3]['truth_unknown'] is False and rows[11]['truth_unknown'] is False
    assert rows[0]['source_group'] == rows[1]['source_group'] == rows[16]['source_group'] == rows[17]['source_group']
    assert len({r['source_group'] for r in rows}) == 8


@pytest.mark.parametrize('kind', ['review_duplicate', 'review_id', 'diagnostic_duplicate', 'diagnostic_id', 'sha', 'origin', 'truth_count'])
def test_input_alignment_and_sha_validation(population_root, kind):
    root = population_root
    if kind.startswith('review') or kind == 'truth_count':
        path = root / q.INPUT_NAMES[0]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if kind == 'truth_count':
            rows[0].update(unknown=False, expected=['cat'], tags=['single_object'])
        else:
            rows[0]['candidate_id'] = rows[1]['candidate_id'] if kind.endswith('duplicate') else 'unknown-candidate-009'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    elif kind == 'sha':
        (root / 'generated/quality-v2-001.jpg').write_bytes(b'changed')
    else:
        path = root / q.INPUT_NAMES[2]
        data = json.loads(path.read_text())
        if kind == 'origin':
            data['images'][0]['origin'] = 'spent_holdout_v1'
        else:
            data['images'][0]['candidate_id'] = data['images'][1]['candidate_id'] if kind.endswith('duplicate') else 'unknown-candidate-009'
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        q.load_population(root)


def test_exact_fixed_grid_and_enumeration():
    assert q.FIXED_GRID == {
        'near_black_fraction': (.5, .75, .9, .95, .99), 'near_white_fraction': (.5, .75, .9, .95, .99),
        'std_luminance': (1, 2, 5, 10, 15, 20, 30), 'entropy': (.5, 1, 2, 3, 4, 5, 6),
        'detail_metric': (.1, .25, .5, 1, 2, 5, 10), 'dynamic_range': (5, 10, 20, 40, 60, 100),
        'robust_dynamic_range': (5, 10, 20, 40, 60, 100)}
    assert len(q.enumerate_rules()) == 13325
    assert q.enumerate_rules() == q.enumerate_rules()


@pytest.mark.parametrize('family,active', list(q.FAMILY_FEATURES.items()))
def test_q0_through_q5_and_inclusive_direction(family, active):
    p = {'near_black_fraction': .9, 'near_white_fraction': .95,
         'std_luminance': 10, 'entropy': 3, 'detail_metric': .5, 'robust_dynamic_range': 40}
    rule = {'family': family, 'parameters': p}
    values = {key: 999. for key in p}
    values.update(near_black_fraction=0., near_white_fraction=0.)
    assert q.rejects(rule, {'metrics': values}) is False
    for exposure in ('near_black_fraction', 'near_white_fraction'):
        m = dict(values, **{exposure: p[exposure]})
        assert q.rejects(rule, {'metrics': m}) is True
        m[exposure] -= 1e-6
        assert q.rejects(rule, {'metrics': m}) is False
    if active:
        m = dict(values, **{key: p[key] for key in active})
        assert q.rejects(rule, {'metrics': m}) is True
        for key in active:
            changed = dict(m, **{key: p[key] + 1e-6})
            assert q.rejects(rule, {'metrics': changed}) is False


def test_metrics_raw_counts_and_zero_denominator():
    rows = [{'candidate_id': str(i), 'truth_unknown': i < 2} for i in range(4)]
    m = q.metrics(rows, {'0', '2'})
    assert (m['tp'], m['fn'], m['fp'], m['tn']) == (1, 1, 1, 1)
    assert m['ratios']['known_fpr'] == {'numerator': 1, 'denominator': 2, 'value': .5}
    assert m['balanced_accuracy'] == .5
    assert q.metrics(rows, set())['unknown_precision'] is None
    assert q.metrics(rows[:2], {'0'})['known_fpr'] is None
    assert q.metrics(rows[2:], set())['unknown_recall'] is None
    assert q.metrics([], set())['balanced_accuracy'] is None


def test_pareto_dominance_preserves_ties():
    rules = [{'rule_id': str(i), 'metrics': {'tp': tp, 'fp': fp}}
             for i, (tp, fp) in enumerate([(5, 0), (5, 1), (7, 1), (7, 1), (6, 2), (9, 2)])]
    assert [r['rule_id'] for r in q.frontier(rules)] == ['0', '2', '3', '5']


def test_transformation_grouping_includes_empty_unknown_motion(population_root):
    rows, _ = q.load_population(population_root)
    grouped = q.transformation_performance(rows, set(q.IDS))
    assert grouped['unknown']['heavy_motion_blur'] == {'numerator': 0, 'denominator': 0, 'value': None}
    assert grouped['known']['heavy_motion_blur']['denominator'] == 2
    assert sum(r['denominator'] for r in grouped['known_hard_negative_review'].values()) == 16


def test_source_group_log_out_without_retuning_and_na(population_root):
    rows, _ = q.load_population(population_root)
    predictions = {r['candidate_id'] for r in rows if r['truth_unknown']}
    result = q.leave_one_source_group_out(rows, predictions)
    assert result['group_count'] == 8 and result['thresholds_retuned'] is False
    assert result['min_unknown_recall'] == 1. and result['max_known_fpr'] == 0.
    assert all(sum(fold['metrics'][k] for k in ('tp', 'fn', 'fp', 'tn')) == 28 for fold in result['folds'])
    tiny = [{'candidate_id': 'a', 'truth_unknown': True, 'source_group': 'a'},
            {'candidate_id': 'b', 'truth_unknown': False, 'source_group': 'b'}]
    result = q.leave_one_source_group_out(tiny, {'a'})
    assert result['recall_na_folds'] == result['fpr_na_folds'] == 1


def test_descriptive_percentiles(population_root):
    rows, _ = q.load_population(population_root)
    summary = q.summarize(rows)
    assert all(d['count'] == 18 for d in summary['by_truth']['known'].values())
    assert all(d['count'] == 14 for d in summary['by_truth']['unknown'].values())
    assert q.distribution([0, 10, 20, 30]) == pytest.approx({'count': 4, 'min': 0, 'median': 15., 'p25': 7.5, 'p75': 22.5, 'p95': 28.5, 'max': 30})


def test_deterministic_reports_and_no_forbidden_file_reads(population_root, monkeypatch):
    root = population_root
    forbidden = root / 'reports/image-information-diagnostics-spent-holdout-v1.json'
    forbidden.write_text('NEVER READ')
    before_stat = forbidden.stat()
    allowed = {root / name for name in q.INPUT_NAMES} | {root / 'generated' / f'{key}.jpg' for key in q.IDS}
    original = Path.open

    def allow_only_inputs(self, mode='r', *args, **kwargs):
        if 'r' in mode and self not in allowed:
            pytest.fail(f'forbidden file read: {self}')
        return original(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', allow_only_inputs)
    rows, hashes = q.load_population(root)
    original_rows = deepcopy(rows)
    report = q.build_report(rows, hashes)
    again = q.build_report(list(reversed(rows)), hashes)
    assert q.encode(report) == q.encode(again)
    assert q.render_text(report) == q.render_text(again)
    assert rows == original_rows
    assert len(report['shortlist_max_10']) <= 10
    assert report['spent_holdout_v1_used'] is False
    assert report['future_holdout_v2_used'] is False
    assert report['production_rule_selected'] is False
    q.publish(root, report)
    with pytest.raises(FileExistsError):
        q.publish(root, report)
    after_rows, after_hashes = q.load_population(root)
    assert rows == after_rows and hashes == after_hashes
    assert forbidden.stat().st_mtime_ns == before_stat.st_mtime_ns


def test_no_model_taxonomy_or_image_decoder_imports():
    code = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','transformers','PIL','eden_vision'} or fullname in {'experiments.model_runtime','experiments.calibration_diagnostics'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from experiments.sweep_quality_gate_rules import enumerate_rules
assert len(enumerate_rules()) == 13325
'''
    done = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
