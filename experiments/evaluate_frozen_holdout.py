"""Evaluate the preregistered rules once on human-reviewed holdout inputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eden_vision.classification.hierarchy import HierarchyResolver
from eden_vision.services.image_input import ImageInputService
from experiments.calibration_diagnostics import _cross_row, _validate_ranking
from experiments.evaluation_core import _coherence
from experiments.model_runtime import MODELS, PROMPT_TEMPLATE, create_runtime, load_evaluation_taxonomy


ROOT = Path('/Users/dev/Datasets/eden-vision-open-set-candidates')
REPO = Path(__file__).resolve().parents[1]
PREREG_SHA = '1f936fb9ed3c132edb11c4665973a79e8688aaee62de5a91095322aef7c9a50b'
IDS = tuple(f'unknown-candidate-{n:03d}' for n in range(9, 17))
NOTES = dict(zip(IDS, (
    'heavily defocused image with no safely identifiable subject',
    'severely underexposed image with no safely identifiable subject',
    'severely overexposed image with no safely identifiable subject',
    'texture-only crop with no safely identifiable subject',
    'extreme crop with no safely identifiable subject',
    'texture-only crop with no safely identifiable subject',
    'severe motion blur with no safely identifiable subject',
    'severely overexposed image with no safely identifiable subject',
), strict=True))
REVIEW_METHOD = 'human_visual_review_before_model_inference'
OUTPUTS = (
    'holdout-clip-signals.json', 'holdout-siglip2-signals.json',
    'holdout-cross-model.json', 'holdout-frozen-rule-evaluation.json',
    'holdout-frozen-rule-evaluation.txt',
)
LIMITATION = ('This holdout contains 8 UNKNOWN and 0 known samples.\n'
              'It independently evaluates UNKNOWN recall only.\n'
              'It cannot independently estimate known false-positive rate.')


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded_json(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def write_new(path, data):
    """Publish a complete file atomically, refusing existing evidence."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    fd, name = tempfile.mkstemp(prefix='.holdout-stage-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def indexed(rows):
    result = {}
    for row in rows:
        key = row.get('candidate_id')
        if not isinstance(key, str) or key in result:
            raise ValueError('missing or duplicate candidate ID')
        result[key] = row
    return result


def holdout_metadata(root):
    rows = indexed([json.loads(line) for line in (root / 'metadata.jsonl').read_text().splitlines() if line.strip()])
    selected = {key: row for key, row in rows.items() if row.get('intended_split') == 'holdout'}
    if set(selected) != set(IDS):
        raise ValueError('holdout metadata must contain exactly candidates 009 through 016')
    for key, row in selected.items():
        if digest((root / 'generated' / f'{key}.jpg').read_bytes()) != row['generated_sha256']:
            raise ValueError(f'generated SHA mismatch: {key}')
    return selected


def review_row(metadata):
    return {
        **{key: metadata[key] for key in ('candidate_id', 'source_image_id', 'transformation', 'generated_sha256')},
        'expected': [], 'acceptable_ancestors': [], 'unknown': True,
        'tags': ['unknown', 'ambiguous'], 'notes': NOTES[metadata['candidate_id']],
        'review_method': REVIEW_METHOD,
    }


def create_review(root):
    """Persist the user's GT and its fingerprint without creating any runtime."""
    root = Path(root)
    paths = [root / 'holdout-review.jsonl', root / 'holdout-review.sha256']
    if any(path.exists() or path.is_symlink() for path in paths):
        raise FileExistsError('holdout review already exists; refusing to overwrite')
    metadata = holdout_metadata(root)
    data = ''.join(json.dumps(review_row(metadata[key]), sort_keys=True) + '\n' for key in IDS).encode()
    write_new(paths[0], data)
    write_new(paths[1], (digest(data) + '\n').encode())
    if digest(paths[0].read_bytes()) != paths[1].read_text().strip():
        raise ValueError('review read-back SHA mismatch')
    print(f'GT fingerprint established before model load: {digest(data)}', flush=True)
    return digest(data)


def verify_preregistration(root):
    path = root / 'reports/open-set-rule-preregistration.json'
    raw = path.read_bytes()
    if digest(raw) != PREREG_SHA or path.with_suffix('.sha256').read_text().strip() != PREREG_SHA:
        raise ValueError('preregistration SHA mismatch; inference prohibited')
    prereg = json.loads(raw)
    if not prereg['rules_frozen'] or prereg['taxonomy_version'] != 'eden-v2':
        raise ValueError('preregistration is not frozen eden-v2')
    for name in ('calibration_diagnostics.py', 'sweep_open_set_rules.py'):
        if digest((REPO / 'experiments' / name).read_bytes()) != prereg['source_code_sha256'][name]:
            raise ValueError(f'frozen source code changed: {name}')
    for source in prereg['source_tuning_reports'].values():
        if digest(Path(source['path']).read_bytes()) != source['sha256']:
            raise ValueError(f'frozen source report changed: {source["path"]}')
    for model, spec in MODELS.items():
        frozen = prereg['models'][model]
        if (spec.model_id, spec.revision, PROMPT_TEMPLATE) != (frozen['model_id'], frozen['revision'], frozen['prompt_template']):
            raise ValueError('runtime identity differs from preregistration')
    return prereg


def validate_inputs(root):
    root = Path(root)
    prereg = verify_preregistration(root)
    raw = (root / 'holdout-review.jsonl').read_bytes()
    review_sha = digest(raw)
    if review_sha != (root / 'holdout-review.sha256').read_text().strip():
        raise ValueError('holdout GT SHA mismatch; inference prohibited')
    reviews = indexed([json.loads(line) for line in raw.splitlines() if line.strip()])
    metadata = holdout_metadata(root)
    if set(reviews) != set(IDS):
        raise ValueError('review must contain only the eight holdout IDs')
    items = []
    for key in IDS:
        if reviews[key] != review_row(metadata[key]):
            raise ValueError(f'human review/provenance mismatch: {key}')
        items.append({**reviews[key], 'generated_path': root / 'generated' / f'{key}.jpg'})
    return prereg, items, review_sha


def collect_signals(items, model, taxonomy, resolver, runtime_factory):
    if len(items) != 8 or {row['candidate_id'] for row in items} != set(IDS):
        raise ValueError('inference requires exactly eight holdout IDs')
    runtime = runtime_factory(model)
    results = []
    decoder = ImageInputService(max_upload_bytes=10 * 1024 * 1024, max_image_pixels=25_000_000)
    try:
        runtime.load()
        if runtime.metadata.get('device') not in {'mps', 'mps:0'}:
            raise ValueError('MPS required; CPU fallback prohibited')
        cache = runtime.prepare_taxonomy(taxonomy)
        for item in items:
            raw = item['generated_path'].read_bytes()
            if digest(raw) != item['generated_sha256']:
                raise ValueError('holdout image changed before inference')
            ranked = _validate_ranking(runtime.rank(decoder.decode(raw, 'image/jpeg')).candidates,
                                       {c.concept_id for c in taxonomy.concepts})
            top5 = ranked[:5]
            results.append({
                **{key: item[key] for key in ('candidate_id', 'source_image_id', 'transformation', 'generated_sha256')},
                'ground_truth': {key: item[key] for key in ('expected', 'unknown', 'tags', 'notes')},
                'top1': top5[0], 'top2': top5[1], 'top5': top5,
                'top1_top2_raw_margin': top5[0]['raw_score'] - top5[1]['raw_score'],
                'top1_top5_raw_spread': top5[0]['raw_score'] - top5[4]['raw_score'],
                'hierarchy_coherence': _coherence([r['concept_id'] for r in top5], resolver),
            })
        runtime_metadata = dict(runtime.metadata)
    finally:
        runtime.close()
    return {
        'schema_version': '1.0', 'kind': 'holdout_raw_ranking_diagnostics',
        'model': model, 'model_id': MODELS[model].model_id, 'model_revision': MODELS[model].revision,
        'taxonomy_version': 'eden-v2', 'prompt_template': PROMPT_TEMPLATE,
        'score_kind': 'raw_image_text_logit',
        'score_note': 'Raw image-text logits are not probabilities or confidence and are not comparable across models.',
        'runtime_metadata': runtime_metadata, 'cache': cache,
        'population': {'total': 8, 'unknown': 8, 'known': 0}, 'images': results,
    }


def apply_expression(expression, features):
    """Interpret only the frozen AST; no fitting, sweep, or threshold selection."""
    if len(expression) != 1:
        raise ValueError('invalid frozen expression')
    op, values = next(iter(expression.items()))
    if op == 'and':
        return all(apply_expression(value, features) for value in values)
    if op == 'or':
        return any(apply_expression(value, features) for value in values)
    if op == 'eq':
        return features[values[0]] == values[1]
    if op == 'lte':
        return features[values[0]] <= values[1]
    raise ValueError(f'unsupported operator: {op}')


def recall_metrics(predictions):
    tp = sum(predictions)
    return {
        'tp': tp, 'fn': len(predictions) - tp, 'unknown_recall': tp / len(predictions),
        'fp': None, 'tn': None, 'known_fpr': None, 'specificity': None,
        'known_metrics_status': 'not_evaluable: zero known samples',
        'unknown_precision': None,
        'precision_note': 'All GT samples are UNKNOWN. Precision would be 1 if any are rejected, '
                          'or undefined if none are rejected; it cannot assess rejection of known samples.',
    }


def evaluate(prereg, cross_rows, review_sha, evaluated_at):
    by_id = indexed(cross_rows)
    if set(by_id) != set(IDS) or any(row['truth_unknown'] is not True for row in cross_rows):
        raise ValueError('evaluation requires eight UNKNOWN holdout rows')
    result = []
    for key in IDS:
        row = by_id[key]
        features = {
            'relation': row['cross_model_relation']['relation'],
            'clip_margin': row['clip_margin'], 'clip_top1_top5_spread': row['clip_spread'],
            'siglip2_margin': row['siglip2_margin'], 'siglip2_top1_top5_spread': row['siglip2_spread'],
        }
        result.append({
            'candidate_id': key, 'truth_unknown': True, **features,
            'primary_pred_unknown': apply_expression(prereg['primary_rule']['boolean_expression'], features),
            'challenger_pred_unknown': apply_expression(prereg['challenger_rule']['boolean_expression'], features),
        })
    return {
        'schema_version': '1.0', 'kind': 'holdout_frozen_rule_evaluation',
        'preregistration_sha256': PREREG_SHA, 'holdout_review_sha256': review_sha,
        'holdout_evaluated_at': evaluated_at, 'rules_applied_unchanged': True,
        'threshold_refit_performed': False, 'tuning_rerun': False,
        'population': {'total': 8, 'unknown': 8, 'known': 0},
        'limitation': LIMITATION,
        'primary_rule': prereg['primary_rule'], 'challenger_rule': prereg['challenger_rule'],
        'relation_semantics': prereg['relation_semantics'],
        'primary': recall_metrics([r['primary_pred_unknown'] for r in result]),
        'challenger': recall_metrics([r['challenger_pred_unknown'] for r in result]),
        'images': result,
    }


def render_text(report):
    lines = [LIMITATION, '', 'Thresholds were not changed using holdout results.',
             'Primary/challenger semantics were applied unchanged.',
             f'Preregistration SHA-256: {report["preregistration_sha256"]}',
             f'Holdout review SHA-256: {report["holdout_review_sha256"]}',
             f'Evaluated at: {report["holdout_evaluated_at"]}']
    for key in ('primary', 'challenger'):
        lines += [f'{key}: {report[key + "_rule"]["expression"]}', json.dumps(report[key], sort_keys=True)]
    lines += [json.dumps(row, sort_keys=True) for row in report['images']]
    return '\n'.join(lines) + '\n'


def run(root, runtime_factory=None):
    root = Path(root)
    for name in OUTPUTS:
        path = root / 'reports' / name
        if path.exists() or path.is_symlink():
            raise FileExistsError(f'refusing to overwrite or rerun: {path}')
    prereg, items, review_sha = validate_inputs(root)
    verified_at = utcnow()
    print(f'GT SHA verified BEFORE creating/loading models: {review_sha}', flush=True)
    print(f'Preregistration SHA verified: {PREREG_SHA}', flush=True)
    taxonomy = load_evaluation_taxonomy('eden-v2')
    resolver = HierarchyResolver(taxonomy)
    signals = {}
    for model in ('clip', 'siglip2'):
        # Recheck immutable inputs immediately before each model is constructed.
        _, items, current_sha = validate_inputs(root)
        if current_sha != review_sha:
            raise ValueError('GT changed between model runs')
        started_at = utcnow()
        print(f'Loading {model} for holdout inference on MPS at {started_at}', flush=True)
        signal = collect_signals(items, model, taxonomy, resolver, runtime_factory or create_runtime)
        signal.update({'preregistration_sha256': PREREG_SHA, 'holdout_review_sha256': review_sha,
                       'gt_fingerprint_verified_at': verified_at, 'model_inference_started_at': started_at})
        signals[model] = signal
        write_new(root / 'reports' / f'holdout-{model}-signals.json', encoded_json(signal))
        print(f'{model} holdout signals saved: 8 rows', flush=True)
    clip, siglip = (indexed(signals[model]['images']) for model in ('clip', 'siglip2'))
    cross_rows = []
    for key in IDS:
        truth = clip[key]['ground_truth'] | {'source_image_id': clip[key]['source_image_id']}
        cross = _cross_row(key, truth, clip[key], siglip[key], resolver)
        cross.update({'clip_top2': clip[key]['top2'], 'siglip2_top2': siglip[key]['top2']})
        cross_rows.append(cross)
    cross = {'taxonomy_version': 'eden-v2', 'kind': 'holdout_cross_model_diagnostic',
             'preregistration_sha256': PREREG_SHA, 'holdout_review_sha256': review_sha,
             'images': cross_rows}
    report = evaluate(prereg, cross_rows, review_sha, utcnow())
    validate_inputs(root)
    write_new(root / 'reports/holdout-cross-model.json', encoded_json(cross))
    write_new(root / 'reports/holdout-frozen-rule-evaluation.json', encoded_json(report))
    write_new(root / 'reports/holdout-frozen-rule-evaluation.txt', render_text(report).encode())
    print(render_text(report), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('review', 'evaluate'))
    parser.add_argument('--dataset', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.action == 'review':
        create_review(args.dataset)
    else:
        run(args.dataset)
