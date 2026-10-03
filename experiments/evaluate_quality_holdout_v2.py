"""One-shot frozen quality evaluation. Never searches rules or overwrites reports."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/Users/dev/Datasets/eden-vision-open-set-quality-holdout-v2')
PREREG = ROOT.parent / 'eden-vision-open-set-quality-calibration-v2/reports/quality-gate-preregistration.json'
PREREG_SHA = '5e724f49d7efeb96a95fc19e4c18be89f2daa7a080172b28ca2abc0e32f69bdf'
UNKNOWN = {f'quality-holdout-v2-{i:03d}' for i in (5, 6, 9, 13, 14, 21, 22)}
HARD = {f'quality-holdout-v2-{i:03d}' for i in (1, 2, 10, 17, 18)}
NAMES = ('PRIMARY', 'CHALLENGER')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_sha(path, expected):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f'SHA mismatch: {path}: {actual} != {expected}')
    return actual


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate():
    gt_sha = require_sha(ROOT / 'holdout-v2-review.jsonl', (ROOT / 'holdout-v2-review.sha256').read_text().split()[0])
    pre_sha = require_sha(PREREG, PREREG_SHA)
    require_sha(PREREG, PREREG.with_suffix('.sha256').read_text().split()[0])
    pre = json.loads(PREREG.read_text())
    source = next(s for s in pre['source_code'] if Path(s['path']).name == 'image_information_diagnostics.py')
    source_sha = require_sha(source['path'], source['sha256'])
    require_sha(pre['decoder_source']['path'], pre['decoder_source']['sha256'])
    from experiments import image_information_diagnostics as info
    assert Path(info.__file__).resolve() == Path(source['path']).resolve()
    assert info.DEFINITIONS == pre['metric_definitions']
    gt, meta = jsonl(ROOT / 'holdout-v2-review.jsonl'), jsonl(ROOT / 'metadata.jsonl')
    ids = {f'quality-holdout-v2-{i:03d}' for i in range(1, 25)}
    assert len(gt) == len(meta) == 24
    assert {r['candidate_id'] for r in gt} == {r['candidate_id'] for r in meta} == ids
    assert {r['candidate_id'] for r in gt if r['unknown']} == UNKNOWN
    assert sum(not r['unknown'] for r in gt) == 17
    byid = {r['candidate_id']: r for r in gt}
    for m in meta:
        g = byid[m['candidate_id']]
        assert type(g['unknown']) is bool and g['unknown'] == (g['expected'] == [])
        for k in ('source_id', 'transformation', 'generated_sha256'):
            assert m[k] == g[k]
        require_sha(ROOT / 'generated' / (m['candidate_id'] + '.jpg'), m['generated_sha256'])
    return pre, byid, sorted(meta, key=lambda r: r['candidate_id']), {
        'quality_preregistration_sha256': pre_sha, 'holdout_review_sha256': gt_sha,
        'metric_source_code_sha256': source_sha, 'generation_plan_sha256': sha(ROOT / 'generation-plan.json'),
        'metadata_sha256': sha(ROOT / 'metadata.jsonl'), 'decoder_source_code_sha256': pre['decoder_source']['sha256'],
        'rules_applied_unchanged': True, 'thresholds_refit': False,
        'metric_semantics_changed': False, 'semantic_models_used': False,
    }


def conditions(m, rule):
    if rule == 'PRIMARY':
        return {'near_black_fraction >= 0.90': m['near_black_fraction'] >= .90,
                'near_white_fraction >= 0.90': m['near_white_fraction'] >= .90,
                'detail_metric <= 2.0 AND entropy <= 6.0': m['detail_metric'] <= 2.0 and m['entropy'] <= 6.0}
    assert rule == 'CHALLENGER'
    return {'near_black_fraction >= 0.90': m['near_black_fraction'] >= .90,
            'near_white_fraction >= 0.50': m['near_white_fraction'] >= .50,
            'std_luminance <= 30.0 AND detail_metric <= 2.0': m['std_luminance'] <= 30.0 and m['detail_metric'] <= 2.0}


def ratio(n, d):
    return {'numerator': n, 'denominator': d, 'value': n / d if d else None}


def confusion(rows, rule):
    tp = sum(r['truth_unknown'] and r[rule]['pred_unknown'] for r in rows)
    fn = sum(r['truth_unknown'] and not r[rule]['pred_unknown'] for r in rows)
    fp = sum(not r['truth_unknown'] and r[rule]['pred_unknown'] for r in rows)
    tn = len(rows) - tp - fn - fp
    recall, spec = ratio(tp, tp + fn), ratio(tn, tn + fp)
    return {'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn, 'correct': tp + tn, 'reject': tp + fp,
            'unknown_recall': recall, 'unknown_precision': ratio(tp, tp + fp),
            'known_fpr': ratio(fp, fp + tn), 'specificity': spec,
            'balanced_accuracy': (recall['value'] + spec['value']) / 2 if tp + fn and fp + tn else None,
            'balanced_accuracy_definition': '(TP/(TP+FN) + TN/(TN+FP))/2'}


def summarize(rows):
    return {'total': len(rows), 'known': sum(not r['truth_unknown'] for r in rows),
            'UNKNOWN': sum(r['truth_unknown'] for r in rows), **{n: confusion(rows, n) for n in NAMES}}


def evaluate(diagnostics, gt, pre, provenance):
    rows = []
    for m in diagnostics:
        g = gt[m['candidate_id']]
        r = {**m, 'truth_unknown': g['unknown'], 'expected': g['expected']}
        for n in NAMES:
            c = conditions(m, n)
            r[n] = {'pred_unknown': any(c.values()), 'conditions': c}
            if g['unknown'] and not any(c.values()):
                r[n]['fn_explanation'] = 'No frozen OR condition satisfied: ' + '; '.join(f'{k} = false' for k in c)
        rows.append(r)
    return {'provenance': provenance, 'population': summarize(rows), 'candidates': rows,
            'intended_type_notice': 'intended_type is generation metadata, not GT. Only human review unknown is GT.',
            'rules': {n: pre[k] for n, k in zip(NAMES, ('primary_rule', 'challenger_rule'))},
            'known_hard_negatives': [r for r in rows if r['candidate_id'] in HARD],
            'unknown_details': [r for r in rows if r['truth_unknown']],
            'by_intended_type': {v: summarize([r for r in rows if r['intended_type'] == v]) for v in sorted({r['intended_type'] for r in rows})},
            'by_transformation': {v: summarize([r for r in rows if r['transformation'] == v]) for v in sorted({r['transformation'] for r in rows})},
            'calibration_comparison': {n: {'calibration-v2': pre[k]['tuning_metrics'], 'holdout-v2': confusion(rows, n)} for n, k in zip(NAMES, ('primary_rule', 'challenger_rule'))}}


def render(report):
    lines = ['Frozen holdout-v2 quality evaluation', report['intended_type_notice'],
             'Predicted UNKNOWN = reject. Correct = TP + TN. Counts always accompany rates.',
             json.dumps(report['provenance'], indent=2, sort_keys=True)]
    for title, group in [('Overall', {'all': report['population']}), ('Intended type', report['by_intended_type']), ('Transformation', report['by_transformation'])]:
        lines.append('\n' + title)
        for key, s in group.items():
            lines.append(f"{key}: total={s['total']} known={s['known']} UNKNOWN={s['UNKNOWN']}")
            for n in NAMES:
                c = s[n]
                lines.append(f"{n}: TP/FN/FP/TN={c['TP']}/{c['FN']}/{c['FP']}/{c['TN']}; correct={c['correct']}/{s['total']}; reject={c['reject']}/{s['total']}")
                for metric in ('unknown_recall', 'unknown_precision', 'known_fpr', 'specificity'):
                    a = c[metric]
                    lines.append(f"  {metric}: {a['numerator']}/{a['denominator']} = " + (f"{a['value']:.2%}" if a['value'] is not None else 'N/A'))
                lines.append(f"  balanced_accuracy: {c['balanced_accuracy']} (TP/(TP+FN) + TN/(TN+FP))/2 using counts above")
    keys = ('near_black_fraction', 'near_white_fraction', 'std_luminance', 'entropy', 'detail_metric', 'robust_dynamic_range')
    for title, rows in [('All candidates', report['candidates']), ('Known hard negatives', report['known_hard_negatives']), ('UNKNOWN details', report['unknown_details'])]:
        lines.extend(['\n' + title, 'candidate_id | transformation | intended_type | truth_unknown | expected | ' + ' | '.join(keys) + ' | PRIMARY | CHALLENGER'])
        for r in rows:
            lines.append(' | '.join(str(r[k]) for k in ('candidate_id', 'transformation', 'intended_type', 'truth_unknown', 'expected', *keys)) + ' | ' + ' | '.join('UNKNOWN' if r[n]['pred_unknown'] else 'known' for n in NAMES))
            for n in NAMES:
                if 'fn_explanation' in r[n]:
                    lines.append(n + ': ' + r[n]['fn_explanation'])
    lines.append('\nCalibration-v2 versus holdout-v2 (descriptive only)')
    for n in NAMES:
        comp = report['calibration_comparison'][n]
        for metric in ('unknown_recall', 'known_fpr'):
            a, b = comp['calibration-v2']['ratios'][metric], comp['holdout-v2'][metric]
            lines.append(f"{n} {metric}: {a['numerator']}/{a['denominator']} = {a['value']:.2%} | {b['numerator']}/{b['denominator']} = {b['value']:.2%}")
    return '\n'.join(lines) + '\n'


def snapshot():
    return {str(p): sha(p) for p in sorted(ROOT.parent.rglob('*')) if p.is_file()}


def main():
    out = ROOT / 'reports'
    paths = [out / ('holdout-v2-' + s) for s in ('image-information-diagnostics.json', 'frozen-quality-evaluation.json', 'frozen-quality-evaluation.txt')]
    if any(p.exists() for p in paths):
        raise FileExistsError('One-shot reports already exist; refusing recalculation/overwrite')
    pre, gt, meta, provenance = validate()
    before = snapshot()
    provenance['evaluation_timestamp'] = datetime.now(timezone.utc).isoformat()
    from experiments.image_information_diagnostics import measure_path, DEFINITIONS
    diagnostics = [{**{k: m[k] for k in ('candidate_id', 'source_id', 'transformation', 'intended_type', 'generated_sha256')},
                    **measure_path(ROOT / 'generated' / (m['candidate_id'] + '.jpg'))} for m in meta]
    report = evaluate(diagnostics, gt, pre, provenance)
    assert all(sha(p) == s for p, s in before.items())
    report['artifact_protection'] = {'existing_artifacts_unchanged': True, 'protected_file_count': len(before), 'sha256_before': before}
    out.mkdir(exist_ok=True)
    contents = [json.dumps({'provenance': provenance, 'metric_definitions': DEFINITIONS, 'candidates': diagnostics}, indent=2, sort_keys=True) + '\n', json.dumps(report, indent=2, sort_keys=True) + '\n', render(report)]
    for p, content in zip(paths, contents):
        with p.open('x') as f:
            f.write(content)
    assert all(sha(p) == s for p, s in before.items())
    print(json.dumps(report['population'], indent=2))


if __name__ == '__main__':
    main()
