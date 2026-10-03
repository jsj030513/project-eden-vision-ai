from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image, ImageFilter, ImageOps
import pytest

from experiments import image_information_diagnostics as info
from experiments import generate_quality_calibration_v2 as gen


def encoded(image, fmt='PNG', **options):
    stream = BytesIO()
    image.save(stream, fmt, **options)
    return stream.getvalue()


def pattern(seed=0, size=(32, 24)):
    image = Image.new('RGB', size)
    image.putdata([((x * 17 + seed) % 256, (y * 21 + seed * 3) % 256,
                    (x * y + seed * 7) % 256) for y in range(size[1]) for x in range(size[0])])
    return image


def test_deterministic_raw_metrics_and_ranges():
    raw = encoded(pattern())
    first = info.measure_bytes(raw)
    assert first == info.measure_bytes(raw)
    assert 0 <= first['mean_luminance'] <= 255
    assert 0 <= first['percentile_01'] <= first['percentile_99'] <= 255
    assert 0 <= first['entropy'] <= 8
    assert first['detail_metric'] >= 0
    for key in ('dark_pixel_fraction', 'bright_pixel_fraction', 'near_black_fraction',
                'near_white_fraction', 'edge_density', 'saturation_mean', 'saturation_std'):
        assert 0 <= first[key] <= 1
    assert first['grayscale_std'] == first['std_luminance']
    assert first['dark_pixel_fraction'] == first['near_black_fraction']
    assert first['bright_pixel_fraction'] == first['near_white_fraction']


@pytest.mark.parametrize('level,black,white', [(0, 1, 0), (10, 1, 0), (11, 0, 0),
                                             (244, 0, 0), (245, 0, 1), (255, 0, 1)])
def test_fixed_metric_definitions_and_uniform_images(level, black, white):
    values = info.measure_bytes(encoded(Image.new('L', (10, 10), level)))
    assert values['mean_luminance'] == level
    assert values['std_luminance'] == values['entropy'] == values['detail_metric'] == values['edge_density'] == 0
    assert values['near_black_fraction'] == black
    assert values['near_white_fraction'] == white
    assert values['dynamic_range'] == 0


def test_known_entropy_percentiles_and_signed_detail():
    image = Image.new('RGB', (20, 10), 'black')
    image.paste('white', (10, 0, 20, 10))
    stats = info.measure_rgb(image)
    assert stats['entropy'] == 1
    assert stats['mean_luminance'] == 127.5
    assert stats['std_luminance'] == 127.5
    assert stats['percentile_01'] == 0 and stats['percentile_99'] == 255
    assert stats['dynamic_range'] == 255
    assert stats['detail_metric'] > info.measure_rgb(image.filter(ImageFilter.GaussianBlur(2)))['detail_metric']
    assert stats['edge_density'] > 0


def test_error_and_tiny_image_handling():
    for raw in (b'', b'not an image'):
        with pytest.raises(ValueError):
            info.measure_bytes(raw)
    with pytest.raises(ValueError, match='positive dimensions'):
        info.measure_rgb(Image.new('RGB', (0, 1)))
    tiny = info.measure_rgb(Image.new('RGB', (1, 1)))
    assert tiny['detail_metric'] is None and tiny['edge_density'] is None
    with pytest.raises(ValueError):
        info.histogram_stats([0] * 256)


@pytest.mark.parametrize('mode,color', [('RGB', (32, 100, 180)), ('RGBA', (255, 0, 0, 0)), ('L', 60)])
def test_rgb_rgba_grayscale(mode, color):
    raw = encoded(Image.new(mode, (8, 6), color))
    stats = info.measure_bytes(raw)
    assert (stats['width'], stats['height']) == (8, 6)
    if mode == 'RGBA':
        assert stats['mean_luminance'] == 128  # existing neutral-gray alpha compositing
    if mode == 'L':
        assert stats['saturation_mean'] == stats['saturation_std'] == 0


def test_exif_transpose():
    exif = Image.Exif()
    exif[274] = 6
    raw = encoded(pattern(size=(8, 6)), exif=exif)
    result = info.measure_bytes(raw)
    assert (result['width'], result['height']) == (6, 8)
    with Image.open(BytesIO(raw)) as image:
        assert result == info.measure_rgb(ImageOps.exif_transpose(image).convert('RGB'))


def test_mpo_primary_frame_only(monkeypatch):
    from PIL import MpoImagePlugin
    raw = encoded(pattern(size=(8, 6)), 'MPO', save_all=True,
                  append_images=[Image.new('RGB', (16, 12), 'blue')])
    with Image.open(BytesIO(raw)) as image:
        expected = info.measure_rgb(image.convert('RGB'))
    seek = MpoImagePlugin.MpoImageFile.seek

    def primary_only(self, frame):
        assert frame == 0
        return seek(self, frame)

    monkeypatch.setattr(MpoImagePlugin.MpoImageFile, 'seek', primary_only)
    assert info.measure_bytes(raw) == expected


def test_heic_current_service_support():
    raw = encoded(pattern(size=(8, 6)), 'HEIF')
    values = info.measure_bytes(raw)
    assert (values['width'], values['height']) == (8, 6)
    assert values == info.measure_bytes(raw)


def test_analysis_scale_is_explicit():
    result = info.measure_rgb(Image.new('RGB', (1024, 512), 'gray'))
    assert (result['width'], result['height']) == (1024, 512)
    assert (result['analysis_width'], result['analysis_height']) == (512, 256)


def test_no_model_or_semantic_imports_required():
    code = '''
import importlib.abc, sys
class BlockModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','transformers','numpy','scipy','cv2'} or fullname in {'experiments.model_runtime','experiments.calibration_diagnostics','experiments.sweep_open_set_rules'}:
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, BlockModels())
from experiments.image_information_diagnostics import measure_rgb
from experiments.generate_quality_calibration_v2 import transform
from PIL import Image
image = Image.new('RGB', (12, 10), 'gray')
assert measure_rgb(transform(image, 'moderate_defocus', 1)[0])['mean_luminance'] == 128
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.fixture
def dataset_roots(tmp_path):
    baseline, canonical, v1 = [tmp_path / name for name in ('baseline', 'canonical', 'v1')]
    for root in (baseline, canonical, v1):
        (root / 'images').mkdir(parents=True)
    base_rows = []
    for i, key in enumerate(gen.SOURCE_IDS + gen.RESERVED_IDS):
        raw = encoded(pattern(100 + i))
        relative = f'images/{key}.png'
        (baseline / relative).write_bytes(raw)
        base_rows.append({'id': key, 'file': relative, 'sha256': gen.sha(raw)})
    (baseline / 'manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in base_rows))
    can_rows = []
    for i in range(1, 34):
        raw = encoded(pattern(i))
        relative = f'images/eval-{i:03d}.png'
        (canonical / relative).write_bytes(raw)
        can_rows.append({'id': f'eval-{i:03d}', 'file': relative, 'sha256': gen.sha(raw)})
    (canonical / 'manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in can_rows))
    (v1 / 'generated').mkdir()
    (v1 / 'reports').mkdir()
    (v1 / 'reports/holdout-frozen-rule-evaluation.json').write_text('protected historical evidence')
    metadata = []
    for i in range(1, 17):
        key = f'unknown-candidate-{i:03d}'
        raw = gen.jpeg_bytes(pattern(200 + i))
        (v1 / 'generated' / f'{key}.jpg').write_bytes(raw)
        metadata.append({'candidate_id': key, 'source_image_id': can_rows[i]['id'],
                         'source_sha256': can_rows[i]['sha256'], 'generated_sha256': gen.sha(raw),
                         'transformation': 'fixture', 'intended_split': 'calibration' if i < 9 else 'holdout'})
    (v1 / 'metadata.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in metadata))
    return baseline, canonical, v1


def test_generation_reproducible_source_partition_no_gt_and_historical_immutability(dataset_roots, tmp_path, monkeypatch):
    before = gen.tree_snapshot(dataset_roots)
    original_transform = gen.transform

    def partition_before_transform(*args, **kwargs):
        plans = list(tmp_path.glob('.quality-v2-*/source-partition.json'))
        assert len(plans) == 1
        plan = json.loads(plans[0].read_text())
        assert plan['partition_fixed_before_generation'] is True
        return original_transform(*args, **kwargs)

    monkeypatch.setattr(gen, 'transform', partition_before_transform)
    first, second = tmp_path / 'new1', tmp_path / 'new2'
    rows = gen.generate(*dataset_roots, first)
    again = gen.generate(*dataset_roots, second)
    assert rows == again and len(rows) == 32
    assert sum(r['intended_type'] == 'unknown_like' for r in rows) == 16
    assert sum(r['intended_type'] == 'known_hard_negative' for r in rows) == 16
    assert all(not {'expected', 'unknown', 'tags'} & row.keys() and row['review_status'] == 'pending' for row in rows)
    assert gen.Counter(r['transform'] for r in rows[:16]) == {key: 2 for key in gen.UNKNOWN_TRANSFORMS}
    plan = json.loads((first / 'source-partition.json').read_text())
    assert not {r['source_sha256'] for r in plan['development_calibration_v2']} & {r['source_sha256'] for r in plan['future_holdout_v2_reserved_only']}
    assert len({r['source_id'] for r in rows}) == 8
    for path in first.rglob('*'):
        if path.is_file():
            assert path.read_bytes() == (second / path.relative_to(first)).read_bytes()
    assert (first / 'contact/calibration-v2.jpg').is_file()
    historical = json.loads((first / 'reports/image-information-diagnostics-spent-holdout-v1.json').read_text())
    assert historical['row_count'] == 8
    assert all(r['usage'] == 'historical evaluation only' and r['threshold_fitting_permitted'] is False for r in historical['images'])
    assert gen.tree_snapshot(dataset_roots) == before
    with pytest.raises(FileExistsError):
        gen.generate(*dataset_roots, first)


def test_source_overlap_is_rejected(dataset_roots):
    baseline, canonical, v1 = dataset_roots
    rows = gen.read_jsonl(canonical / 'manifest.jsonl')
    rows[0]['sha256'] = gen.read_jsonl(baseline / 'manifest.jsonl')[0]['sha256']
    (canonical / 'manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match='overlap'):
        gen.source_plan(baseline, canonical, v1)
