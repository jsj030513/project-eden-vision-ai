from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from experiments import evaluate_frozen_holdout as h
from experiments.calibration_diagnostics import taxonomy_relation
from experiments.model_runtime import load_evaluation_taxonomy
from eden_vision.classification.hierarchy import HierarchyResolver


@pytest.fixture
def fixture_root(tmp_path, monkeypatch):
    root = tmp_path / 'candidates'
    (root / 'generated').mkdir(parents=True)
    (root / 'reports').mkdir()
    metadata = []
    for number in range(1, 17):
        key = f'unknown-candidate-{number:03d}'
        row = {'candidate_id': key, 'intended_split': 'holdout' if number >= 9 else 'calibration',
               'source_image_id': f'eval-{number:03d}', 'transformation': 'test', 'generated_sha256': 'unused'}
        if number >= 9:
            path = root / 'generated' / f'{key}.jpg'
            Image.new('RGB', (24, 24), (number, 20, 30)).save(path)
            row['generated_sha256'] = h.digest(path.read_bytes())
        metadata.append(row)
    (root / 'metadata.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in metadata))
    gate = {'eq': ['relation', 'unrelated']}
    prereg = {
        'rules_frozen': True, 'taxonomy_version': 'eden-v2',
        'models': {key: {'model_id': spec.model_id, 'revision': spec.revision,
                         'prompt_template': h.PROMPT_TEMPLATE} for key, spec in h.MODELS.items()},
        'source_code_sha256': {name: h.digest((h.REPO / 'experiments' / name).read_bytes())
                               for name in ('calibration_diagnostics.py', 'sweep_open_set_rules.py')},
        'source_tuning_reports': {},
        'relation_semantics': {'root_only_common_ancestor_is_unrelated': True},
        'primary_rule': {
            'expression': 'relation == "unrelated" AND (clip_margin <= 0.10 OR siglip2_margin <= 0.50)',
            'boolean_expression': {'and': [gate, {'or': [
                {'lte': ['clip_margin', .10]}, {'lte': ['siglip2_margin', .50]}]}]},
        },
        'challenger_rule': {
            'expression': 'relation == "unrelated" AND ((clip_margin <= 3.00 AND clip_top1_top5_spread <= 1.00) OR (siglip2_margin <= 0.50 AND siglip2_top1_top5_spread <= 4.00))',
            'boolean_expression': {'and': [gate, {'or': [
                {'and': [{'lte': ['clip_margin', 3.]}, {'lte': ['clip_top1_top5_spread', 1.]}]},
                {'and': [{'lte': ['siglip2_margin', .5]}, {'lte': ['siglip2_top1_top5_spread', 4.]}]},
            ]}]},
        },
    }
    raw = h.encoded_json(prereg)
    (root / 'reports/open-set-rule-preregistration.json').write_bytes(raw)
    (root / 'reports/open-set-rule-preregistration.sha256').write_text(h.digest(raw) + '\n')
    monkeypatch.setattr(h, 'PREREG_SHA', h.digest(raw))
    return root, prereg


def test_review_is_eight_unknown_excludes_calibration_and_matches_sha(fixture_root):
    root, _ = fixture_root
    sha = h.create_review(root)
    _, items, actual_sha = h.validate_inputs(root)
    assert sha == actual_sha == h.digest((root / 'holdout-review.jsonl').read_bytes())
    assert {r['candidate_id'] for r in items} == set(h.IDS)
    assert sum(r['unknown'] for r in items) == 8
    assert all(r['expected'] == r['acceptable_ancestors'] == [] for r in items)
    with pytest.raises(FileExistsError):
        h.create_review(root)


@pytest.mark.parametrize('failure', ['missing_gt', 'gt_hash', 'prereg_json', 'prereg_sha', 'calibration_row', 'image_sha', 'relation_source'])
def test_invalid_fingerprints_and_inputs_stop_before_runtime(fixture_root, failure):
    root, _ = fixture_root
    h.create_review(root)
    if failure == 'missing_gt':
        (root / 'holdout-review.sha256').unlink()
    elif failure == 'gt_hash':
        (root / 'holdout-review.sha256').write_text('0' * 64)
    elif failure == 'prereg_json':
        path = root / 'reports/open-set-rule-preregistration.json'
        path.write_bytes(path.read_bytes() + b' ')
    elif failure == 'prereg_sha':
        (root / 'reports/open-set-rule-preregistration.sha256').write_text('0' * 64)
    elif failure == 'image_sha':
        (root / 'generated/unknown-candidate-009.jpg').write_bytes(b'changed')
    elif failure == 'relation_source':
        # Test an unchanged preregistration against changed implementation bytes.
        original_repo = h.REPO
        fake_repo = root / 'fake_repo'
        (fake_repo / 'experiments').mkdir(parents=True)
        for name in ('calibration_diagnostics.py', 'sweep_open_set_rules.py'):
            (fake_repo / 'experiments' / name).write_bytes((original_repo / 'experiments' / name).read_bytes() + b'\n')
        from unittest.mock import patch
        with patch.object(h, 'REPO', fake_repo):
            with pytest.raises(ValueError, match='source code changed'):
                h.run(root, runtime_factory=lambda _: pytest.fail('runtime created before validation'))
        return
    else:
        path = root / 'holdout-review.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['candidate_id'] = 'unknown-candidate-001'
        raw = ''.join(json.dumps(row) + '\n' for row in rows).encode()
        path.write_bytes(raw)
        (root / 'holdout-review.sha256').write_text(h.digest(raw))
    with pytest.raises((ValueError, FileNotFoundError)):
        h.run(root, runtime_factory=lambda _: pytest.fail('runtime created before validation'))


def test_gt_fingerprint_workflow_before_model_load_and_existing_files_unchanged(fixture_root):
    root, _ = fixture_root
    h.create_review(root)
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    events = []

    class FakeRuntime:
        metadata = {'device': 'mps'}

        def load(self):
            h.validate_inputs(root)
            events.append('load_after_gt_and_prereg_validated')

        def prepare_taxonomy(self, taxonomy):
            return {}

        def rank(self, image):
            events.append('rank')
            return SimpleNamespace(candidates=[{'concept_id': key, 'raw_score': score}
                for key, score in zip(('cat', 'dog', 'animal', 'tree', 'flower'), (10., 9.9, 8., 7., 6.))])

        def close(self):
            pass

    def factory(model):
        h.validate_inputs(root)
        events.append('create_after_gt_and_prereg_validated')
        return FakeRuntime()

    result = h.run(root, runtime_factory=factory)
    assert events[0:2] == ['create_after_gt_and_prereg_validated', 'load_after_gt_and_prereg_validated']
    assert events.count('rank') == 16
    assert result['rules_applied_unchanged'] is True
    assert result['threshold_refit_performed'] is False
    assert result['tuning_rerun'] is False
    assert all(p.read_bytes() == raw for p, raw in before.items())
    assert all((root / 'reports' / name).exists() for name in h.OUTPUTS)
    with pytest.raises(FileExistsError):
        h.run(root, runtime_factory=lambda _: pytest.fail('reran inference'))


@pytest.mark.parametrize('cm,sm,cp,sp,relation,primary,challenger', [
    (.10, 1., 1., 5., 'unrelated', True, True),
    (.100001, .500001, 1.000001, 4.000001, 'unrelated', False, False),
    (3., .6, 1., 5., 'unrelated', False, True),
    (3.000001, .6, 1., 5., 'unrelated', False, False),
    (5., .5, 5., 4., 'unrelated', True, True),
    (5., .5, 5., 4.000001, 'unrelated', True, False),
    (0., 0., 0., 0., 'exact', False, False),
    (0., 0., 0., 0., 'ancestor_descendant', False, False),
    (0., 0., 0., 0., 'shared_ancestor', False, False),
])
def test_exact_frozen_rules_at_boundaries(fixture_root, cm, sm, cp, sp, relation, primary, challenger):
    _, prereg = fixture_root
    features = {'relation': relation, 'clip_margin': cm, 'siglip2_margin': sm,
                'clip_top1_top5_spread': cp, 'siglip2_top1_top5_spread': sp}
    assert h.apply_expression(prereg['primary_rule']['boolean_expression'], features) is primary
    assert h.apply_expression(prereg['challenger_rule']['boolean_expression'], features) is challenger


def test_zero_known_metrics_and_deterministic_evaluation_without_refitting(fixture_root):
    _, prereg = fixture_root
    original = deepcopy(prereg)
    rows = [{'candidate_id': key, 'truth_unknown': True, 'clip_margin': .1,
             'siglip2_margin': 2., 'clip_spread': 2., 'siglip2_spread': 5.,
             'cross_model_relation': {'relation': 'unrelated'}} for key in h.IDS]
    first = h.evaluate(prereg, rows, 'fixture_sha', 'fixed_time')
    second = h.evaluate(prereg, list(reversed(rows)), 'fixture_sha', 'fixed_time')
    assert h.encoded_json(first) == h.encoded_json(second)
    assert h.render_text(first) == h.render_text(second)
    assert prereg == original
    assert first['primary']['tp'] == 8 and first['primary']['fn'] == 0
    assert first['primary']['unknown_recall'] == 1.
    assert first['challenger']['unknown_recall'] == 0.
    for field in ('fp', 'tn', 'known_fpr', 'specificity', 'unknown_precision'):
        assert first['primary'][field] is None
    assert h.LIMITATION in h.render_text(first)


def test_relation_semantics_reused_unchanged():
    resolver = HierarchyResolver(load_evaluation_taxonomy('eden-v2'))
    assert taxonomy_relation('cat', 'cat', resolver)['relation'] == 'exact'
    assert taxonomy_relation('cat', 'animal', resolver)['relation'] == 'ancestor_descendant'
    assert taxonomy_relation('cat', 'living_thing', resolver)['relation'] == 'ancestor_descendant'
    assert taxonomy_relation('cat', 'dog', resolver)['relation'] == 'shared_ancestor'
    assert taxonomy_relation('natural_scene', 'outdoor_scene', resolver)['relation'] == 'unrelated'
    assert taxonomy_relation('cat', 'screen', resolver)['relation'] == 'unrelated'
