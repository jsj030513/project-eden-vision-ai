"""Generate pending quality candidates and model-free diagnostics in a new directory."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
from experiments.image_information_diagnostics import DEFINITIONS, decode_rgb, measure_bytes


BASELINE = Path('/Users/dev/Datasets/eden-vision-eval')
CANONICAL = Path('/Users/dev/Datasets/eden-vision-open-set-eval')
V1 = Path('/Users/dev/Datasets/eden-vision-open-set-candidates')
OUTPUT = Path('/Users/dev/Datasets/eden-vision-open-set-quality-calibration-v2')
SEED = 20260925
SOURCE_IDS = ('eval-001', 'eval-022', 'eval-031', 'eval-040', 'eval-043', 'eval-045', 'eval-054', 'eval-056')
RESERVED_IDS = ('eval-010', 'eval-020', 'eval-028', 'eval-044', 'eval-050', 'eval-053')
UNKNOWN_TRANSFORMS = ('severe_underexposure', 'severe_overexposure', 'heavy_defocus',
                      'heavy_motion_blur', 'extreme_crop', 'texture_only_crop',
                      'low_contrast_image', 'near_uniform_image')
KNOWN_TRANSFORMS = (
    ('moderate_underexposure', 'moderate_defocus'),
    ('moderate_overexposure', 'partial_crop'),
    ('moderate_defocus', 'moderate_motion_blur'),
    ('reduced_contrast', 'moderate_underexposure'),
    ('moderate_motion_blur', 'reduced_contrast'),
    ('moderate_defocus', 'moderate_underexposure'),
    ('moderate_motion_blur', 'partial_crop'),
    ('partial_crop', 'moderate_overexposure'),
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def tree_snapshot(roots):
    return {str(path): sha(path.read_bytes()) for root in roots
            for path in sorted(root.rglob('*')) if path.is_file()}


def verify_snapshot(snapshot, roots):
    after = tree_snapshot(roots)
    if snapshot != after:
        raise RuntimeError('protected dataset listing or bytes changed')
    return {'files': len(snapshot), 'sha256': sha(json.dumps(snapshot, sort_keys=True).encode()),
            'listing_and_bytes_unchanged': True}


def source_plan(baseline, canonical, v1):
    rows = read_jsonl(baseline / 'manifest.jsonl')
    by_id = {row['id']: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError('duplicate baseline source IDs')
    old_meta = read_jsonl(v1 / 'metadata.jsonl')
    old_canonical = read_jsonl(canonical / 'manifest.jsonl')
    excluded_hashes = {r['sha256'] for r in old_canonical} | {r['source_sha256'] for r in old_meta}
    excluded_generated = {r['generated_sha256'] for r in old_meta}
    selected, reserved = [], []
    for ids, destination in ((SOURCE_IDS, selected), (RESERVED_IDS, reserved)):
        for key in ids:
            row = by_id[key]
            path = (baseline / row['file']).resolve()
            if not path.is_relative_to(baseline.resolve()):
                raise ValueError('source path escapes dataset')
            actual_sha = sha(path.read_bytes())
            if actual_sha != row['sha256'] or actual_sha in excluded_hashes | excluded_generated:
                raise ValueError('source hash mismatch or overlap with open-set-v1')
            destination.append({'source_id': f'baseline:{key}', 'source_image_id': key,
                                'source_dataset': 'eden-vision-eval', 'source_path': str(path),
                                'source_sha256': actual_sha})
    all_hashes = [r['source_sha256'] for r in selected + reserved]
    if len(set(all_hashes)) != len(all_hashes) or set(SOURCE_IDS) & set(RESERVED_IDS):
        raise ValueError('duplicate or overlapping source groups')
    return {
        'schema_version': '1.0', 'seed': SEED,
        'partition_fixed_before_generation': True,
        'development_calibration_v2': selected,
        'future_holdout_v2_reserved_only': reserved,
        'future_holdout_v2_generated': False,
        'source_overlap_with_open_set_canonical_and_v1_sources': 0,
        'development_future_holdout_source_overlap': 0,
        'comparison': 'Namespaced source IDs and exact SHA-256; not a perceptual near-duplicate guarantee',
        'intended_types_are_not_ground_truth': True,
    }


def horizontal_motion(image, length):
    """Exact horizontal box motion kernel; replicate border pixels, round once."""
    length = max(1, int(length) | 1)
    radius = length // 2
    raw = image.tobytes()
    w, h = image.size
    result = bytearray(len(raw))
    for y in range(h):
        start = y * w * 3
        for channel in range(3):
            values = list(raw[start + channel:start + w * 3:3])
            padded = [values[0]] * radius + values + [values[-1]] * radius
            total = sum(padded[:length])
            for x in range(w):
                result[start + x * 3 + channel] = round(total / length)
                if x + 1 < w:
                    total += padded[x + length] - padded[x]
    return Image.frombytes('RGB', image.size, bytes(result))


def transform(image, name, seed):
    rng = random.Random(seed)
    base = image.copy()
    base.thumbnail((640, 640), Image.Resampling.LANCZOS)
    params = {'base_size': list(base.size), 'base_resize': 'aspect preserved, max side 640, no upscale'}
    if name in ('severe_underexposure', 'moderate_underexposure'):
        factor = rng.choice((.018, .025)) if name.startswith('severe') else .30
        result = ImageEnhance.Brightness(base).enhance(factor)
        params['brightness_factor'] = factor
    elif name in ('severe_overexposure', 'moderate_overexposure'):
        factor = 5.0 if name.startswith('severe') else 1.6
        wash = .97 if name.startswith('severe') else 0.
        result = ImageEnhance.Brightness(base).enhance(factor)
        if wash:
            result = Image.blend(result, Image.new('RGB', base.size, 'white'), wash)
        params.update(brightness_factor=factor, white_wash_fraction=wash)
    elif name in ('heavy_defocus', 'moderate_defocus'):
        radius = min(base.size) * .17 if name.startswith('heavy') else 2.2
        result = base.filter(ImageFilter.GaussianBlur(radius))
        params['gaussian_radius_pixels'] = radius
    elif name in ('heavy_motion_blur', 'moderate_motion_blur'):
        length = round(base.width * .85) | 1 if name.startswith('heavy') else 13
        result = horizontal_motion(base, length)
        params.update(length_pixels=length, angle_degrees=0, border='replicate', accumulation='integer sum then single rounding')
    elif name in ('extreme_crop', 'texture_only_crop', 'partial_crop'):
        area = {'extreme_crop': .035, 'texture_only_crop': .012, 'partial_crop': .70}[name]
        w, h = base.size
        cw, ch = max(1, round(w * math.sqrt(area))), max(1, round(h * math.sqrt(area)))
        left = (w - cw) // 2 if name == 'partial_crop' else rng.randrange(w - cw + 1)
        top = (h - ch) // 2 if name == 'partial_crop' else rng.randrange(h - ch + 1)
        box = (left, top, left + cw, top + ch)
        result = base.crop(box).resize(base.size, Image.Resampling.LANCZOS)
        radius = {'extreme_crop': 16., 'texture_only_crop': 6., 'partial_crop': 0.}[name]
        if radius:
            result = result.filter(ImageFilter.GaussianBlur(radius))
        params.update(target_area_fraction=area, actual_area_fraction=cw * ch / (w * h),
                      crop_box_on_base=list(box), post_crop_gaussian_radius_pixels=radius)
    elif name in ('low_contrast_image', 'reduced_contrast'):
        factor = .035 if name == 'low_contrast_image' else .35
        radius = 10. if name == 'low_contrast_image' else 0.
        result = ImageEnhance.Contrast(base.filter(ImageFilter.GaussianBlur(radius))).enhance(factor)
        params.update(contrast_factor=factor, pre_contrast_blur_radius=radius)
    elif name == 'near_uniform_image':
        radius = min(base.size) * .2
        blurred = base.filter(ImageFilter.GaussianBlur(radius))
        color = tuple(base.resize((1, 1), Image.Resampling.BOX).getpixel((0, 0)))
        result = Image.blend(blurred, Image.new('RGB', base.size, color), .97)
        params.update(gaussian_radius_pixels=radius, uniform_blend_fraction=.97, uniform_rgb=list(color))
    else:
        raise ValueError(f'unsupported transform: {name}')
    params['output_size'] = list(result.size)
    return result, params


def jpeg_bytes(image):
    output = BytesIO()
    image.save(output, 'JPEG', quality=95, subsampling=0, optimize=False)
    return output.getvalue()


def contact_sheet(rows, generated, path):
    columns, cw, ih, caption = 4, 380, 270, 92
    header = 46
    sheet = Image.new('RGB', (columns * cw, header + math.ceil(len(rows) / columns) * (ih + caption)), 'white')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=15)
    draw.text((12, 12), 'Calibration-v2 candidates | intended type is NOT GT | human review pending', fill='black', font=font)
    for i, row in enumerate(rows):
        x, y = (i % columns) * cw, header + (i // columns) * (ih + caption)
        with Image.open(generated / f'{row["candidate_id"]}.jpg') as image:
            preview = ImageOps.contain(image, (cw - 16, ih - 12))
            sheet.paste(preview, (x + (cw - preview.width) // 2, y + (ih - preview.height) // 2))
        for line, text in enumerate((row['candidate_id'], row['source_id'], row['transform'], row['intended_type'])):
            draw.text((x + 9, y + ih + line * 21), text, fill='black', font=font)
    sheet.save(path, 'JPEG', quality=94, subsampling=0)


def diagnostic_report(rows, scope):
    return {'schema_version': '1.0', 'scope': scope, 'metric_definitions': DEFINITIONS,
            'model_inference_used': False, 'semantic_signals_used': False,
            'threshold_fitting_performed': False, 'row_count': len(rows), 'images': rows}


def historical_reports(canonical, v1, output):
    canonical_rows = []
    manifest = read_jsonl(canonical / 'manifest.jsonl')
    if len(manifest) != 33:
        raise ValueError('expected canonical 33')
    for source in manifest:
        raw = (canonical / source['file']).read_bytes()
        if sha(raw) != source['sha256']:
            raise ValueError('canonical image hash mismatch')
        canonical_rows.append({'candidate_id': source['id'], 'source_id': f'open-set:{source["id"]}',
                               'transform': 'original', 'origin': 'open_set_canonical_v1',
                               'image_sha256': sha(raw), **measure_bytes(raw)})
    write_json(output / 'image-information-diagnostics-canonical-v1.json',
               diagnostic_report(canonical_rows, 'read-only historical diagnostics'))
    old_rows = read_jsonl(v1 / 'metadata.jsonl')
    for split in ('calibration', 'holdout'):
        selected = [r for r in old_rows if r['intended_split'] == split]
        required = {f'unknown-candidate-{i:03d}' for i in (range(1, 9) if split == 'calibration' else range(9, 17))}
        if len(selected) != 8 or {r['candidate_id'] for r in selected} != required:
            raise ValueError('historical split IDs mismatch')
        metrics = []
        for row in sorted(selected, key=lambda r: r['candidate_id']):
            raw = (v1 / 'generated' / f'{row["candidate_id"]}.jpg').read_bytes()
            if sha(raw) != row['generated_sha256']:
                raise ValueError('historical generated SHA mismatch')
            item = {'candidate_id': row['candidate_id'], 'source_id': f'open-set:{row["source_image_id"]}',
                    'transform': row['transformation'], 'origin': f'{"spent_holdout" if split == "holdout" else "calibration"}_v1',
                    'image_sha256': sha(raw), **measure_bytes(raw)}
            if split == 'holdout':
                item.update(usage='historical evaluation only', threshold_fitting_permitted=False,
                            historical_detection_from_user=('missed' if row['candidate_id'] in
                            {'unknown-candidate-009', 'unknown-candidate-010', 'unknown-candidate-012'} else 'detected'))
            metrics.append(item)
        scope = 'historical evaluation only; SPENT HOLDOUT; excluded from threshold fitting' if split == 'holdout' else 'read-only historical diagnostics'
        name = 'spent-holdout-v1' if split == 'holdout' else 'calibration-v1'
        write_json(output / f'image-information-diagnostics-{name}.json', diagnostic_report(metrics, scope))
        if split == 'holdout':
            fields = ('mean_luminance', 'std_luminance', 'near_black_fraction', 'near_white_fraction',
                      'percentile_01', 'percentile_99', 'dynamic_range', 'entropy', 'detail_metric',
                      'edge_density', 'saturation_mean', 'saturation_std')
            lines = [scope, 'Raw measurements only. No rejection thresholds, recommendations, or fitting.',
                     'Analysis raster: aspect-preserving max side 512; no upscaling.',
                     '\t'.join(('candidate_id', 'historical_detection_from_user', *fields))]
            lines += ['\t'.join([r['candidate_id'], r['historical_detection_from_user']] +
                                [f'{r[key]:.9g}' for key in fields]) for r in metrics]
            (output / 'spent-holdout-v1-quality-comparison.txt').write_text('\n'.join(lines) + '\n')


def generate(baseline=BASELINE, canonical=CANONICAL, v1=V1, output=OUTPUT):
    baseline, canonical, v1, output = map(Path, (baseline, canonical, v1, output))
    if output.exists():
        raise FileExistsError('output already exists; existing artifacts are never overwritten')
    protected_roots = (baseline, canonical, v1)
    snapshot = tree_snapshot(protected_roots)
    plan = source_plan(baseline, canonical, v1)
    staging = Path(tempfile.mkdtemp(prefix='.quality-v2-', dir=output.parent))
    try:
        for name in ('generated', 'contact', 'reports'):
            (staging / name).mkdir()
        # Persist source partition BEFORE creating any candidate image.
        write_json(staging / 'source-partition.json', plan)
        rows = []
        for intended_type in ('unknown_like', 'known_hard_negative'):
            for index, source in enumerate(plan['development_calibration_v2']):
                raw = Path(source['source_path']).read_bytes()
                if sha(raw) != source['source_sha256']:
                    raise ValueError('source changed before generation')
                with decode_rgb(raw) as image:
                    names = tuple(UNKNOWN_TRANSFORMS[(2 * index + j) % 8] for j in range(2)) if intended_type == 'unknown_like' else KNOWN_TRANSFORMS[index]
                    for name in names:
                        key = f'quality-v2-{len(rows) + 1:03d}'
                        seed = int.from_bytes(hashlib.sha256(f'{SEED}:{key}'.encode()).digest()[:8], 'big')
                        result, params = transform(image, name, seed)
                        data = jpeg_bytes(result)
                        (staging / 'generated' / f'{key}.jpg').write_bytes(data)
                        rows.append({'candidate_id': key, **source, 'transform': name, 'parameters': params,
                                     'seed': seed, 'global_seed': SEED, 'generated_sha256': sha(data),
                                     'intended_type': intended_type, 'intended_split': 'development/calibration-v2',
                                     'review_status': 'pending', 'intended_type_is_ground_truth': False})
        if Counter(r['intended_type'] for r in rows) != {'unknown_like': 16, 'known_hard_negative': 16}:
            raise RuntimeError('candidate counts differ from fixed design')
        (staging / 'metadata.jsonl').write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
        contact_sheet(rows, staging / 'generated', staging / 'contact/calibration-v2.jpg')
        for kind in ('unknown_like', 'known_hard_negative'):
            contact_sheet([r for r in rows if r['intended_type'] == kind], staging / 'generated', staging / f'contact/{kind}.jpg')
        metrics = [{**{k: row[k] for k in ('candidate_id', 'source_id', 'transform', 'intended_type', 'generated_sha256')},
                    'origin': 'quality_calibration_v2_candidate', 'review_status': 'pending',
                    **measure_bytes((staging / 'generated' / f'{row["candidate_id"]}.jpg').read_bytes())} for row in rows]
        write_json(staging / 'reports/image-information-diagnostics.json',
                   diagnostic_report(metrics, 'development candidates only; no GT assigned'))
        # Historical data are diagnosed only AFTER the fixed new generation plan is realized.
        historical_reports(canonical, v1, staging / 'reports')
        protection = verify_snapshot(snapshot, protected_roots)
        write_json(staging / 'reports/protection-verification.json', protection)
        os.rename(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    parser.add_argument('--canonical', type=Path, default=CANONICAL)
    parser.add_argument('--v1', type=Path, default=V1)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    rows = generate(args.baseline, args.canonical, args.v1, args.output)
    print(json.dumps({'output': str(args.output), 'candidates': len(rows),
                      'types': dict(Counter(r['intended_type'] for r in rows)),
                      'unknown_like_transform_counts': dict(Counter(r['transform'] for r in rows if r['intended_type'] == 'unknown_like'))}, indent=2))
