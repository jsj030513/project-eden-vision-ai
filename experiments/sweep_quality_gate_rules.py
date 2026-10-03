"""Compare fixed, interpretable quality rules on reviewed calibration-v2 only."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import tempfile


DEFAULT_ROOT = Path('/Users/dev/Datasets/eden-vision-open-set-quality-calibration-v2')
INPUT_NAMES = ('unknown-like-review.jsonl', 'known-hard-negative-review.jsonl',
               'reports/image-information-diagnostics.json')
IDS = tuple(f'quality-v2-{n:03d}' for n in range(1, 33))
FEATURES = ('mean_luminance', 'std_luminance', 'near_black_fraction', 'near_white_fraction',
            'percentile_01', 'percentile_99', 'dynamic_range', 'entropy', 'detail_metric',
            'edge_density', 'saturation_mean', 'saturation_std', 'robust_dynamic_range')
FIXED_GRID = {
    'near_black_fraction': (.50, .75, .90, .95, .99),
    'near_white_fraction': (.50, .75, .90, .95, .99),
    'std_luminance': (1, 2, 5, 10, 15, 20, 30),
    'entropy': (.5, 1, 2, 3, 4, 5, 6),
    'detail_metric': (.1, .25, .5, 1, 2, 5, 10),
    'dynamic_range': (5, 10, 20, 40, 60, 100),
    'robust_dynamic_range': (5, 10, 20, 40, 60, 100),
}
FAMILY_FEATURES = {
    'Q0': (), 'Q1': ('std_luminance', 'entropy'),
    'Q2': ('detail_metric', 'entropy'), 'Q3': ('std_luminance', 'detail_metric'),
    'Q4': ('robust_dynamic_range', 'entropy'),
    'Q5': ('std_luminance', 'entropy', 'detail_metric'),
}
UNKNOWN_TRANSFORMS = ('severe_underexposure', 'severe_overexposure', 'heavy_defocus',
                      'heavy_motion_blur', 'extreme_crop', 'texture_only_crop',
                      'low_contrast_image', 'near_uniform_image')
JSON_NAME = 'quality-gate-rule-sweep.json'
TEXT_NAME = 'quality-gate-rule-sweep.txt'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def index_rows(rows):
    result = {}
    for row in rows:
        key = row.get('candidate_id')
        if not isinstance(key, str) or key in result:
            raise ValueError('missing or duplicate candidate ID')
        result[key] = row
    return result


def load_population(root):
    """Only these three filenames and the 32 generated-image raw hashes are read.

    Historical/spent-holdout reports, metadata, source partitions, and source
    images are never opened. Truth comes solely from the human review files.
    """
    root = Path(root)
    raw_inputs = {}
    for name in INPUT_NAMES:
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('input symlinks or escaped paths are prohibited')
        raw_inputs[name] = path.read_bytes()
    reviews = {}
    for name, expected_ids in zip(INPUT_NAMES[:2], (set(IDS[:16]), set(IDS[16:])), strict=True):
        part = index_rows([json.loads(line) for line in raw_inputs[name].splitlines() if line.strip()])
        if set(part) != expected_ids:
            raise ValueError(f'review ID alignment mismatch: {name}')
        for key, row in part.items():
            reviews[key] = {**row, 'review_artifact': name}
    diagnostic = json.loads(raw_inputs[INPUT_NAMES[2]])
    metrics = index_rows(diagnostic['images'])
    if set(metrics) != set(IDS) or set(metrics) != set(reviews) or diagnostic.get('row_count') != 32:
        raise ValueError('diagnostic ID alignment mismatch: exactly calibration-v2 001..032 required')
    rows = []
    for key in IDS:
        truth, info = reviews[key], metrics[key]
        if info.get('origin') != 'quality_calibration_v2_candidate':
            raise ValueError('historical or holdout origin prohibited')
        if truth.get('review_method') != 'human_visual_review_without_model_signals':
            raise ValueError('human visual review required')
        if type(truth.get('unknown')) is not bool or not isinstance(truth.get('expected'), list):
            raise ValueError('invalid human truth fields')
        if truth['unknown'] != (truth['expected'] == []) or truth['unknown'] != ('unknown' in truth.get('tags', [])):
            raise ValueError('truth expected/unknown/tags inconsistent')
        if truth['source_id'] != info['source_id'] or truth['transformation'] != info['transform']:
            raise ValueError('review/diagnostic provenance mismatch')
        path = root / 'generated' / f'{key}.jpg'
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('generated image symlinks or escaped paths prohibited')
        if truth['generated_sha256'] != info['generated_sha256'] or sha(path.read_bytes()) != truth['generated_sha256']:
            raise ValueError(f'generated SHA mismatch: {key}')
        values = {}
        for feature in FEATURES[:-1]:
            value = info.get(feature)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'invalid raw metric {feature}: {key}')
            values[feature] = value
        if not 0 <= values['percentile_01'] <= values['percentile_99'] <= 255:
            raise ValueError('invalid percentile ordering or range')
        for feature in ('near_black_fraction', 'near_white_fraction', 'edge_density', 'saturation_mean', 'saturation_std'):
            if values[feature] > 1:
                raise ValueError('fraction outside 0..1')
        values['robust_dynamic_range'] = values['percentile_99'] - values['percentile_01']
        if 'robust_dynamic_range' in info and not math.isclose(info['robust_dynamic_range'], values['robust_dynamic_range'], abs_tol=1e-12):
            raise ValueError('stored robust dynamic range differs from percentiles')
        # The namespaced source_id is the original image ID grouped within its
        # dataset. Keep both parts; never confuse equal IDs from different datasets.
        source_id = truth['source_id']
        if not isinstance(source_id, str) or ':' not in source_id:
            raise ValueError('namespaced source ID required')
        rows.append({'candidate_id': key, 'source_id': source_id,
                     'source_image_id': source_id.split(':', 1)[1],
                     'source_group': source_id, 'transformation': truth['transformation'],
                     'truth_unknown': truth['unknown'], 'truth_expected': truth['expected'],
                     'review_notes': truth['notes'], 'review_artifact': truth['review_artifact'],
                     'generated_sha256': truth['generated_sha256'], 'metrics': values})
    if sum(r['truth_unknown'] for r in rows) != 14 or sum(not r['truth_unknown'] for r in rows) != 18:
        raise ValueError('population must be 32 / known18 / UNKNOWN14')
    if sum(reviews[key]['unknown'] for key in IDS[:16]) != 14 or any(reviews[key]['unknown'] for key in IDS[16:]):
        raise ValueError('human review subgroup counts mismatch')
    return rows, {name: sha(raw) for name, raw in raw_inputs.items()}


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    at = (len(ordered) - 1) * p
    lo, hi = math.floor(at), math.ceil(at)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (at - lo)


def distribution(values):
    return {'count': len(values), 'min': min(values, default=None),
            'median': percentile(values, .5), 'p25': percentile(values, .25),
            'p75': percentile(values, .75), 'p95': percentile(values, .95), 'max': max(values, default=None)}


def summarize(rows):
    def group(items):
        return {feature: distribution([r['metrics'][feature] for r in items]) for feature in FEATURES}
    result = {'by_truth': {}, 'by_truth_and_transformation': {}}
    for name, truth in (('known', False), ('unknown', True)):
        selected = [r for r in rows if r['truth_unknown'] is truth]
        result['by_truth'][name] = group(selected)
        transforms = sorted({r['transformation'] for r in selected} | (set(UNKNOWN_TRANSFORMS) if truth else set()))
        result['by_truth_and_transformation'][name] = {
            transform: group([r for r in selected if r['transformation'] == transform]) for transform in transforms}
    return result


def enumerate_rules():
    rules = []
    for family, features in FAMILY_FEATURES.items():
        keys = ('near_black_fraction', 'near_white_fraction', *features)
        for index, thresholds in enumerate(itertools.product(*(FIXED_GRID[key] for key in keys)), 1):
            parameters = dict(zip(keys, thresholds, strict=True))
            exposure = f'(near_black_fraction >= {parameters[keys[0]]} OR near_white_fraction >= {parameters[keys[1]]})'
            expression = exposure + (' OR (' + ' AND '.join(f'{key} <= {parameters[key]}' for key in features) + ')' if features else '')
            rules.append({'rule_id': f'{family}-{index:05d}', 'family': family, 'parameters': parameters,
                          'condition_count': len(keys), 'expression': expression})
    return rules


def rejects(rule, row):
    m, p = row['metrics'], rule['parameters']
    exposure = m['near_black_fraction'] >= p['near_black_fraction'] or m['near_white_fraction'] >= p['near_white_fraction']
    extra = FAMILY_FEATURES[rule['family']]
    return exposure or (bool(extra) and all(m[key] <= p[key] for key in extra))


def ratio(numerator, denominator):
    return {'numerator': numerator, 'denominator': denominator,
            'value': numerator / denominator if denominator else None}


def metrics(rows, predicted):
    tp = sum(r['truth_unknown'] and r['candidate_id'] in predicted for r in rows)
    fn = sum(r['truth_unknown'] and r['candidate_id'] not in predicted for r in rows)
    fp = sum(not r['truth_unknown'] and r['candidate_id'] in predicted for r in rows)
    tn = sum(not r['truth_unknown'] and r['candidate_id'] not in predicted for r in rows)
    ratios = {'unknown_recall': ratio(tp, tp + fn), 'unknown_precision': ratio(tp, tp + fp),
              'known_fpr': ratio(fp, fp + tn), 'specificity': ratio(tn, fp + tn)}
    values = {key: value['value'] for key, value in ratios.items()}
    balanced = (values['unknown_recall'] + values['specificity']) / 2 if tp + fn and fp + tn else None
    return {'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn, **values, 'balanced_accuracy': balanced,
            'ratios': ratios, 'balanced_accuracy_definition': '(TP/(TP+FN) + TN/(FP+TN))/2'}


def priority(rule):
    indices = [(FIXED_GRID[key].index(value), len(FIXED_GRID[key])) for key, value in rule['parameters'].items()]
    edge_count = sum(index in (0, count - 1) for index, count in indices)
    distance = sum(abs(index / (count - 1) - .5) for index, count in indices)
    return (int(rule['family'][1:]), rule['condition_count'], edge_count, distance, rule['rule_id'])


def frontier(candidates):
    """Keep all nondominated rules; equal metric points remain ties."""
    points = {(r['metrics']['fp'], r['metrics']['tp']) for r in candidates}
    best_tp = -1
    nondominated = set()
    for fp in sorted({point[0] for point in points}):
        tp = max(point[1] for point in points if point[0] == fp)
        if tp > best_tp:
            nondominated.add((fp, tp))
            best_tp = tp
    return [r for r in candidates if (r['metrics']['fp'], r['metrics']['tp']) in nondominated]


def representatives(candidates):
    chosen = {}
    for r in frontier(candidates):
        key = (r['metrics']['fp'], r['metrics']['tp'])
        if key not in chosen or priority(r) < priority(chosen[key]):
            chosen[key] = r
    return sorted(chosen.values(), key=lambda r: (r['metrics']['fp'], -r['metrics']['tp'], priority(r)))


def transformation_performance(rows, predicted):
    result = {}
    for name, predicate, truth in (
        ('unknown', lambda r: r['truth_unknown'], True),
        ('known', lambda r: not r['truth_unknown'], False),
        ('known_hard_negative_review', lambda r: not r['truth_unknown'] and r['review_artifact'] == INPUT_NAMES[1], False),
    ):
        selected = [r for r in rows if predicate(r)]
        transforms = sorted({r['transformation'] for r in selected} | (set(UNKNOWN_TRANSFORMS) if truth else set()))
        result[name] = {t: ratio(sum(r['candidate_id'] in predicted for r in selected if r['transformation'] == t),
                                 sum(r['transformation'] == t for r in selected)) for t in transforms}
    return result


def leave_one_source_group_out(rows, predicted):
    folds = []
    for group in sorted({r['source_group'] for r in rows}):
        subset = [r for r in rows if r['source_group'] != group]
        folds.append({'excluded_source_group': group, 'metrics': metrics(subset, predicted)})
    recalls = [f['metrics']['unknown_recall'] for f in folds if f['metrics']['unknown_recall'] is not None]
    fprs = [f['metrics']['known_fpr'] for f in folds if f['metrics']['known_fpr'] is not None]
    return {'group_count': len(folds), 'thresholds_retuned': False,
            'min_unknown_recall': min(recalls, default=None), 'median_unknown_recall': percentile(recalls, .5),
            'max_known_fpr': max(fprs, default=None), 'median_known_fpr': percentile(fprs, .5),
            'recall_na_folds': len(folds) - len(recalls), 'fpr_na_folds': len(folds) - len(fprs), 'folds': folds}


def build_report(rows, input_hashes, summary=None):
    if len(rows) != 32 or {r['candidate_id'] for r in rows} != set(IDS) or sum(r['truth_unknown'] for r in rows) != 14:
        raise ValueError('quality sweep requires exactly calibration-v2 32/18/14')
    rows = sorted(rows, key=lambda row: row['candidate_id'])
    rules = []
    for rule in enumerate_rules():
        predicted = {r['candidate_id'] for r in rows if rejects(rule, r)}
        rules.append({**rule, 'metrics': metrics(rows, predicted)})
    global_frontier = representatives(rules)
    family_frontiers = {family: representatives([r for r in rules if r['family'] == family]) for family in FAMILY_FEATURES}
    family_best = {family: min(values, key=lambda r: (-r['metrics']['tp'], r['metrics']['fp'], priority(r)))
                   for family, values in family_frontiers.items()}
    criteria = {
        'fp_eq_0': (lambda r: r['metrics']['fp'] == 0, lambda r: (-r['metrics']['tp'], r['metrics']['fp'])),
        'fp_le_1': (lambda r: r['metrics']['fp'] <= 1, lambda r: (-r['metrics']['tp'], r['metrics']['fp'])),
        'fp_le_2': (lambda r: r['metrics']['fp'] <= 2, lambda r: (-r['metrics']['tp'], r['metrics']['fp'])),
        'recall_ge_80_percent': (lambda r: r['metrics']['unknown_recall'] >= .8, lambda r: (r['metrics']['fp'], -r['metrics']['tp'])),
        'recall_ge_90_percent': (lambda r: r['metrics']['unknown_recall'] >= .9, lambda r: (r['metrics']['fp'], -r['metrics']['tp'])),
        'recall_eq_100_percent': (lambda r: r['metrics']['unknown_recall'] == 1., lambda r: (r['metrics']['fp'], -r['metrics']['tp'])),
    }
    selections = {}
    for name, (eligible, order) in criteria.items():
        candidates = [r for r in global_frontier if eligible(r)]
        selections[name] = min(candidates, key=lambda r: (*order(r), priority(r))) if candidates else None
    shortlisted = {}
    for rule in list(selections.values()) + global_frontier:
        if rule is not None and len(shortlisted) < 10:
            shortlisted[rule['rule_id']] = rule
    details = []
    for rule in sorted(shortlisted.values(), key=lambda r: (r['metrics']['fp'], -r['metrics']['tp'])):
        predicted = {r['candidate_id'] for r in rows if rejects(rule, r)}
        details.append({**rule,
            'transformation_performance': transformation_performance(rows, predicted),
            'source_group_robustness': leave_one_source_group_out(rows, predicted),
            'false_positives': [r for r in rows if not r['truth_unknown'] and r['candidate_id'] in predicted],
            'false_negatives': [r for r in rows if r['truth_unknown'] and r['candidate_id'] not in predicted]})
    return {'schema_version': '1.0', 'kind': 'quality_calibration_v2_fixed_grid_comparison',
            'population': {'total': 32, 'known': 18, 'unknown': 14},
            'truth_source': 'human review artifacts only; intended_type is ignored',
            'source_input_sha256': input_hashes, 'descriptive_summary': summary or summarize(rows),
            'fixed_grid': FIXED_GRID, 'q4_feature': 'robust_dynamic_range = percentile_99 - percentile_01',
            'edge_density_usage': 'descriptive only; not a feature in the user-specified Q0..Q5 families',
            'dynamic_range_grid_usage': 'recorded fallback grid; unused because stored percentiles are available',
            'tie_break': 'Pareto non-dominance; then family Q0..Q5, condition count, fewer grid endpoints, distance from grid midpoint, stable ID. No sample-distance fitting.',
            'rules_evaluated': len(rules), 'rules': rules, 'family_frontiers': family_frontiers,
            'family_best': family_best, 'pareto_frontier': global_frontier, 'operating_regions': selections,
            'shortlist_max_10': details, 'model_inference_used': False, 'taxonomy_relation_used': False,
            'spent_holdout_v1_used': False, 'future_holdout_v2_used': False,
            'production_rule_selected': False, 'thresholds_frozen': False}


def format_ratio(value):
    return f'{value["numerator"]}/{value["denominator"]}=' + (f'{100 * value["value"]:.4f}%' if value['value'] is not None else 'N/A')


def rule_line(rule):
    if rule is None:
        return 'none'
    m = rule['metrics']
    return (f'{rule["rule_id"]}: {rule["expression"]}; TP/FN/FP/TN={m["tp"]}/{m["fn"]}/{m["fp"]}/{m["tn"]}; '
            + '; '.join(f'{key} {format_ratio(value)}' for key, value in m['ratios'].items())
            + f'; balanced_accuracy={m["balanced_accuracy"]:.6f} (average of the above recall and specificity)')


def summary_text(summary):
    lines = ['Population: total=32, known=18, UNKNOWN=14', 'Descriptive raw metrics BEFORE rule sweep (linear-interpolated percentiles):']
    for group, values in summary['by_truth'].items():
        lines.append(group + ': count | min | median | p25 | p75 | p95 | max')
        for key, d in values.items():
            lines.append(f'  {key}: ' + ' | '.join(str(d[k]) for k in ('count', 'min', 'median', 'p25', 'p75', 'p95', 'max')))
    return '\n'.join(lines)


def results_text(report):
    lines = ['Family highest-recall Pareto representatives (lowest FP secondary; no production selection):']
    lines += [rule_line(rule) for rule in report['family_best'].values()]
    lines.append('Operating regions:')
    lines += [f'{name}: {rule_line(rule)}' for name, rule in report['operating_regions'].items()]
    lines.append('Top Pareto rules: transformation robustness, source-group robustness and FP/FN:')
    for rule in report['shortlist_max_10']:
        lines.append(rule_line(rule))
        for group, transforms in rule['transformation_performance'].items():
            lines.append(group + ': ' + '; '.join(f'{name} {format_ratio(r)}' for name, r in transforms.items()))
        robustness = rule['source_group_robustness']
        lines.append('LOGO without retuning: ' + json.dumps({k: v for k, v in robustness.items() if k != 'folds'}, sort_keys=True))
        lines.append('LOGO raw counts: ' + '; '.join(f'{f["excluded_source_group"]} recall {format_ratio(f["metrics"]["ratios"]["unknown_recall"])} FPR {format_ratio(f["metrics"]["ratios"]["known_fpr"])}' for f in robustness['folds']))
        for group in ('false_positives', 'false_negatives'):
            lines.append(group + ':')
            lines += [json.dumps(row, sort_keys=True) for row in rule[group]] or ['none']
    return '\n'.join(lines)


def transformation_summary_text(summary):
    lines = ['Transformation-specific raw distributions (count and all metric ranges):']
    for truth, transforms in summary['by_truth_and_transformation'].items():
        for transform, metrics_summary in transforms.items():
            lines.append(f'{truth}/{transform}: n={metrics_summary[FEATURES[0]]["count"]}; ' + '; '.join(
                f'{key} [{d["min"]}, {d["max"]}]' for key, d in metrics_summary.items()))
    return '\n'.join(lines)


def render_text(report):
    lines = [summary_text(report['descriptive_summary']), transformation_summary_text(report['descriptive_summary']),
              'Fixed user-specified grid: ' + json.dumps(report['fixed_grid'], sort_keys=True),
              'Tie handling: ' + report['tie_break'], results_text(report),
              'No spent holdout-v1 input; no future holdout-v2 access; no semantic/model signals.',
              'No production rule or freeze is selected by this comparison.']
    return '\n\n'.join(lines) + '\n'


def publish(root, report):
    targets = [Path(root) / 'reports' / name for name in (JSON_NAME, TEXT_NAME)]
    if any(p.exists() or p.is_symlink() for p in targets):
        raise FileExistsError('refusing to overwrite quality sweep reports')
    created = []
    try:
        for target, raw in zip(targets, (encode(report), render_text(report).encode()), strict=True):
            fd, name = tempfile.mkstemp(prefix='.quality-sweep-', dir=target.parent)
            try:
                with os.fdopen(fd, 'wb') as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(name, target)
                created.append(target)
            finally:
                Path(name).unlink(missing_ok=True)
    except BaseException:
        for target in created:
            target.unlink()
        raise


def run(root=DEFAULT_ROOT):
    rows, hashes = load_population(root)
    summary = summarize(rows)
    print(summary_text(summary), flush=True)
    print(transformation_summary_text(summary), flush=True)
    report = build_report(rows, hashes, summary)
    # Verify that the three inputs and every generated SHA remain identical.
    after_rows, after_hashes = load_population(root)
    if after_rows != rows or after_hashes != hashes:
        raise ValueError('inputs changed during sweep')
    publish(root, report)
    print(results_text(report), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_ROOT)
    run(parser.parse_args().dataset)
