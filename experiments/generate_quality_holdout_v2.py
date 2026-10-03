"""Generate reserved-source holdout candidates only; no GT, metrics, or rule evaluation."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from collections import Counter

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageDraw, ImageFont, ImageOps
from experiments.generate_quality_calibration_v2 import (
    transform, jpeg_bytes, decode_rgb, UNKNOWN_TRANSFORMS, read_jsonl,
)

ROOT = Path('/Users/dev/Datasets')
CAL = ROOT / 'eden-vision-open-set-quality-calibration-v2'
OUTPUT = ROOT / 'eden-vision-open-set-quality-holdout-v2'
EXPECTED_SHA = '5e724f49d7efeb96a95fc19e4c18be89f2daa7a080172b28ca2abc0e32f69bdf'
SEED = 20260925
MODERATE = ('moderate_underexposure', 'moderate_overexposure', 'moderate_defocus',
            'moderate_motion_blur', 'partial_crop', 'reduced_contrast')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    with path.open('x') as f:
        f.write(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + '\n')


def validate_preregistration(path):
    raw = path.read_bytes()
    if sha(raw) != EXPECTED_SHA or path.with_suffix('.sha256').read_text().strip() != EXPECTED_SHA:
        raise ValueError('preregistration fingerprint mismatch')
    artifact = json.loads(raw)
    for item in artifact['source_code'] + [artifact['decoder_source']]:
        if sha(Path(item['path']).read_bytes()) != item['sha256']:
            raise ValueError('frozen implementation changed')
    return artifact


def validate_sources(partition, old):
    sources = partition['future_holdout_v2_reserved_only']
    calibration = partition['development_calibration_v2']
    if len(sources) != 6 or len(calibration) != 8:
        raise ValueError('expected six reserved and eight calibration sources')
    ids = {s['source_id'] for s in sources}
    hashes = {s['source_sha256'] for s in sources}
    if len(ids) != 6 or len(hashes) != 6:
        raise ValueError('duplicate reserved sources')
    other_ids = {s['source_id'] for s in calibration} | {f"open-set:{s['source_image_id']}" for s in old}
    other_hashes = {s['source_sha256'] for s in calibration + old}
    if ids & other_ids or hashes & other_hashes:
        raise ValueError('source overlap')
    for source in sources:
        if sha(Path(source['source_path']).read_bytes()) != source['source_sha256']:
            raise ValueError('reserved source SHA mismatch')
    return sources


def parameters(image, name, seed):
    """Resolve existing generator policy before generating candidates; no diagnostics."""
    rng = random.Random(seed)
    base = image.copy()
    base.thumbnail((640, 640), Image.Resampling.LANCZOS)
    w, h = base.size
    p = {'base_size': [w, h], 'base_resize': 'aspect preserved, max side 640, no upscale', 'output_size': [w, h]}
    if name in ('severe_underexposure', 'moderate_underexposure'):
        p['brightness_factor'] = rng.choice((.018, .025)) if name.startswith('severe') else .30
    elif name in ('severe_overexposure', 'moderate_overexposure'):
        p.update(brightness_factor=5. if name.startswith('severe') else 1.6,
                 white_wash_fraction=.97 if name.startswith('severe') else 0.)
    elif name in ('heavy_defocus', 'moderate_defocus'):
        p['gaussian_radius_pixels'] = min(w, h) * .17 if name.startswith('heavy') else 2.2
    elif name in ('heavy_motion_blur', 'moderate_motion_blur'):
        p.update(length_pixels=(round(w * .85) | 1) if name.startswith('heavy') else 13,
                 angle_degrees=0, border='replicate', accumulation='integer sum then single rounding')
    elif name in ('extreme_crop', 'texture_only_crop', 'partial_crop'):
        area = {'extreme_crop': .035, 'texture_only_crop': .012, 'partial_crop': .70}[name]
        cw, ch = max(1, round(w * math.sqrt(area))), max(1, round(h * math.sqrt(area)))
        left = (w-cw)//2 if name == 'partial_crop' else rng.randrange(w-cw+1)
        top = (h-ch)//2 if name == 'partial_crop' else rng.randrange(h-ch+1)
        p.update(target_area_fraction=area, actual_area_fraction=cw*ch/(w*h),
                 crop_box_on_base=[left, top, left+cw, top+ch],
                 post_crop_gaussian_radius_pixels={'extreme_crop':16., 'texture_only_crop':6., 'partial_crop':0.}[name])
    elif name in ('low_contrast_image', 'reduced_contrast'):
        p.update(contrast_factor=.035 if name == 'low_contrast_image' else .35,
                 pre_contrast_blur_radius=10. if name == 'low_contrast_image' else 0.)
    elif name == 'near_uniform_image':
        p.update(gaussian_radius_pixels=min(w,h)*.2, uniform_blend_fraction=.97,
                 uniform_rgb=list(base.resize((1,1), Image.Resampling.BOX).getpixel((0,0))))
    else:
        raise ValueError(name)
    return p


def make_plan(sources):
    rows = []
    for i, source in enumerate(sources):
        with decode_rgb(Path(source['source_path']).read_bytes()) as image:
            names = [(UNKNOWN_TRANSFORMS[(2*i+j)%8], 'severe_candidate') for j in range(2)]
            names += [(MODERATE[(2*i+j)%6], 'moderate_candidate') for j in range(2)]
            for name, kind in names:
                key = f'quality-holdout-v2-{len(rows)+1:03d}'
                seed = int.from_bytes(hashlib.sha256(f'{SEED}:{key}'.encode()).digest()[:8], 'big')
                rows.append(dict(source, candidate_id=key, transformation=name,
                                 parameters=parameters(image,name,seed), seed=seed,
                                 intended_type=kind, review_status='pending'))
    return {'schema_version':'1.0', 'global_seed':SEED, 'preregistration_sha256':EXPECTED_SHA,
            'intended_types_are_not_ground_truth':True,
            'parameter_policy':'Unchanged calibration-v2 transform policy; parameters resolved before candidate generation',
            'jpeg_encoding':{'quality':95,'subsampling':0,'optimize':False},
            'candidate_count':24, 'candidates':rows}


def contact(rows, generated, path):
    cw, ih, ch, columns = 400, 280, 96, 4
    sheet = Image.new('RGB', (cw*columns, math.ceil(len(rows)/columns)*(ih+ch)), 'white')
    draw = ImageDraw.Draw(sheet); font = ImageFont.load_default(size=15)
    for i,row in enumerate(rows):
        x,y = (i%columns)*cw,(i//columns)*(ih+ch)
        with Image.open(generated/f'{row["candidate_id"]}.jpg') as image:
            preview = ImageOps.contain(image,(cw-16,ih-12))
            sheet.paste(preview,(x+(cw-preview.width)//2,y+(ih-preview.height)//2))
        for line,key in enumerate(('candidate_id','source_image_id','transformation','intended_type')):
            draw.text((x+8,y+ih+line*22),row[key],font=font,fill='black')
    with path.open('xb') as f:
        sheet.save(f,'JPEG',quality=94,subsampling=0)


def materialize(plan, output):
    """Require a persisted identical plan before any candidate transform is called."""
    if json.loads((output/'generation-plan.json').read_text()) != plan:
        raise ValueError('persisted generation plan mismatch')
    rows = []
    for item in plan['candidates']:
        raw = Path(item['source_path']).read_bytes()
        if sha(raw) != item['source_sha256']:
            raise ValueError('source changed')
        with decode_rgb(raw) as image:
            result, params = transform(image,item['transformation'],item['seed'])
            if params != item['parameters']:
                raise ValueError('parameters differ from pre-generation plan')
            data = jpeg_bytes(result)
            # Independent repeat of the identical transform, no metrics or inspection.
            repeated, repeated_params = transform(image,item['transformation'],item['seed'])
            if repeated_params != params or jpeg_bytes(repeated) != data:
                raise ValueError('non-reproducible generation')
        with (output/'generated'/f'{item["candidate_id"]}.jpg').open('xb') as f:
            f.write(data)
        rows.append(dict(item,generated_sha256=sha(data),global_seed=SEED))
    with (output/'metadata.jsonl').open('x') as f:
        f.write(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    for name,subset in [('all',rows),('severe',[r for r in rows if r['intended_type']=='severe_candidate']),
                        ('moderate',[r for r in rows if r['intended_type']=='moderate_candidate'])]:
        contact(subset,output/'generated',output/'contact'/f'{name}.jpg')
    for item in rows:
        if sha(Path(item['source_path']).read_bytes()) != item['source_sha256']:
            raise ValueError('source changed after generation')
    return rows


def generate():
    validate_preregistration(CAL/'reports/quality-gate-preregistration.json')
    partition_path = CAL/'source-partition.json'
    partition = json.loads(partition_path.read_text())
    sources = validate_sources(partition,read_jsonl(ROOT/'eden-vision-open-set-candidates/metadata.jsonl'))
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    protected = [CAL,ROOT/'eden-vision-open-set-candidates',ROOT/'eden-vision-open-set-eval',ROOT/'eden-vision-eval']
    def snapshot():
        return {str(p):sha(p.read_bytes()) for root in protected for p in root.rglob('*') if p.is_file()}
    before = snapshot()
    plan = make_plan(sources)
    plan['source_partition_sha256'] = sha(partition_path.read_bytes())
    plan['generator_sha256'] = sha(Path(__file__).read_bytes())
    plan['transformation_implementation_sha256'] = sha(Path(__file__).with_name('generate_quality_calibration_v2.py').read_bytes())
    OUTPUT.mkdir()
    for name in ('source','generated','contact'):
        (OUTPUT/name).mkdir()
    write_json(OUTPUT/'generation-plan.json',plan)
    rows = materialize(plan,OUTPUT)
    assert snapshot() == before, 'protected artifacts changed'
    print(json.dumps({'output':str(OUTPUT),'plan_sha256':sha((OUTPUT/'generation-plan.json').read_bytes()),
                      'candidates':len(rows),'transform_counts':dict(Counter(r['transformation'] for r in rows)),
                      'protected_files_unchanged':len(before),'reproducible':True},indent=2))


if __name__ == '__main__':
    generate()
