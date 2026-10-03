"""Offline v3 planning and raw-score summaries; no model loader or reject decision."""
import hashlib
import json
import math
from pathlib import Path

EXPERIMENT = 'semantic-identifiability-v3'
REJECT_PROMPTS = (
    'a blurry photograph with no identifiable subject',
    'an overexposed photograph with no identifiable subject',
    'an underexposed photograph with no identifiable subject',
    'an abstract texture with no identifiable object',
    'an extreme close-up with no identifiable subject',
    'an unrecognizable photograph',
    'a photograph with no clear object or scene',
)
TAXONOMY_TEMPLATE = 'a photo of a {display_name}'
MODELS = {
    'clip': {'model_id': 'openai/clip-vit-base-patch32', 'revision': '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'},
    'siglip2': {'model_id': 'google/siglip2-base-patch16-224', 'revision': '75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2'},
}
FAMILIES = ('heavy_defocus', 'heavy_motion_blur', 'severe_underexposure', 'severe_overexposure', 'extreme_or_texture_crop')
REPO = Path(__file__).resolve().parents[1]
TAXONOMY = REPO / 'src/eden_vision/resources/taxonomy_v2.json'
DATASETS = Path('/Users/dev/Datasets')
OUTPUT = DATASETS / 'eden-vision-semantic-identifiability-v3'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'


def write_new(path, value):
    with Path(path).open('x') as stream:
        stream.write(canonical(value))


def prompt_registration():
    taxonomy = json.loads(TAXONOMY.read_text())
    concepts = taxonomy['concepts']
    if taxonomy['taxonomy_version'] != 'eden-v2' or len(concepts) != 78:
        raise ValueError('expected eden-v2 with 78 concepts')
    return {
        'experiment': EXPERIMENT, 'prompt_set_frozen': True,
        'reject_prompts': list(REJECT_PROMPTS), 'taxonomy_prompt_template': TAXONOMY_TEMPLATE,
        'taxonomy_display_name_normalization': 'lowercase, matching existing evaluation runtime',
        'taxonomy_prompts': [{'concept_id': c['concept_id'], 'prompt': TAXONOMY_TEMPLATE.format(display_name=c['display_name'].lower())} for c in concepts],
        'models': MODELS, 'taxonomy_sha256': sha(TAXONOMY), 'source_code_sha256': sha(__file__),
        'diagnostic_prompts_are_taxonomy_concepts': False,
        'diagnostic_prompts_returned_as_classification': False,
        'score_kind': 'raw_image_text_logit',
        'score_comparison_scope': 'within the same pinned model only; no CLIP/SigLIP2 score comparison',
        'inference_run': False, 'production_integration': False,
        'prompt_changes_after_results_forbidden': True,
        'future_fitting_population': 'new calibration-v3 human-reviewed examples only',
        'spent_holdout_v1_v2_usage': 'historical failure analysis only; excluded from prompt selection and fitting',
        'calibration_v2_usage': 'descriptive comparison only',
    }


def summarize_raw_scores(model, revision, taxonomy_scores, reject_scores):
    """Pure reducer for future same-model logits. Does not run inference or classify."""
    if model not in MODELS or revision != MODELS[model]['revision']:
        raise ValueError('unpinned model/revision')
    ids = {c['concept_id'] for c in json.loads(TAXONOMY.read_text())['concepts']}
    if set(taxonomy_scores) != ids or set(reject_scores) != set(REJECT_PROMPTS):
        raise ValueError('expected exactly 78 taxonomy and seven diagnostic scores')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in [*taxonomy_scores.values(), *reject_scores.values()]):
        raise ValueError('scores must be finite raw logits')
    tax = sorted(taxonomy_scores.items(), key=lambda x: (-x[1], x[0]))
    rej = sorted(reject_scores.items(), key=lambda x: (-x[1], REJECT_PROMPTS.index(x[0])))
    return {'model': model, 'model_revision': revision, 'score_kind': 'raw_image_text_logit',
            'best_taxonomy_concept': tax[0][0], 'best_taxonomy_raw_score': tax[0][1],
            'best_reject_prompt': rej[0][0], 'best_reject_raw_score': rej[0][1],
            'taxonomy_reject_gap': tax[0][1] - rej[0][1],
            'taxonomy_top1_top2_margin': tax[0][1] - tax[1][1],
            'taxonomy_top5': [{'concept_id': k, 'raw_score': v} for k, v in tax[:5]]}


def historical_inventory(roots):
    """Conservatively exclude every historical metadata ID and content/source hash."""
    ids, hashes, files = set(), set(), {}
    def walk(value):
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, str):
                    if k in ('id', 'source_id', 'source_image_id', 'candidate_id', 'image_id'):
                        ids.add(v)
                        if v.startswith('eval-'):
                            ids.add('baseline:' + v)
                    if len(v) == 64 and all(c in '0123456789abcdef' for c in v):
                        hashes.add(v)
                walk(v)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    for root in roots:
        for p in sorted(Path(root).rglob('*')):
            if p.is_file() and p.suffix in ('.json', '.jsonl'):
                files[str(p)] = sha(p)
                if p.suffix == '.jsonl':
                    for line in p.read_text().splitlines():
                        if line.strip():
                            walk(json.loads(line))
                else:
                    walk(json.loads(p.read_text()))
    return {'excluded_ids': sorted(ids), 'excluded_sha256': sorted(hashes), 'evidence_files': files,
            'scope': 'all historical dataset JSON/JSONL, including baseline model evaluation; conservative exact ID/SHA exclusion'}


def reserve_sources(calibration, holdout, inventory):
    """Validate explicit new source manifests without decoding/viewing images."""
    if len(calibration) != 10 or len(holdout) != 10:
        raise ValueError('reserve 10 distinct sources per partition')
    used_ids, used_sha = set(inventory['excluded_ids']), set(inventory['excluded_sha256'])
    for row in [*calibration, *holdout]:
        sid, digest = row['source_id'], row['source_sha256']
        if sid in used_ids or digest in used_sha:
            raise ValueError('historical or cross-partition overlap')
        if len(digest) != 64 or sha(row['source_path']) != digest:
            raise ValueError('source SHA mismatch')
        used_ids.add(sid)
        used_sha.add(digest)
    return {'status': 'reserved_before_generation', 'calibration_v3': calibration, 'future_holdout_v3': holdout,
            'historical_source_overlap': 0, 'calibration_holdout_overlap': 0,
            'holdout_images_generated': False, 'holdout_images_viewed': False, 'holdout_inference_run': False}


def generation_plan(partition):
    slots = []
    for index, family in enumerate(FAMILIES):
        for source_offset in range(2):
            source_index = index * 2 + source_offset
            sources = partition['calibration_v3']
            for target in ('known', 'UNKNOWN'):
                for replicate in range(2):
                    slots.append({'candidate_id': f'semantic-v3-cal-{len(slots)+1:03d}', 'transformation_family': family,
                                  'source_slot': source_index + 1,
                                  'source_id': sources[source_index]['source_id'] if sources else None,
                                  'recruitment_target': target, 'replicate': replicate + 1,
                                  'human_gt': None})
    return {'experiment': EXPERIMENT, 'status': 'design_only', 'planned_total': 40,
            'target_known': 20, 'target_UNKNOWN': 20, 'families': {f: {'known_target': 4, 'UNKNOWN_target': 4} for f in FAMILIES},
            'candidates': slots, 'images_generated': False, 'inference_run': False,
            'generation_parameters_status': 'not specified; generation not authorized in this stage',
            'ground_truth_policy': 'Human semantic-identifiability review before diagnostics; recruitment target and transformation are not GT. Record actual labels even if quotas are missed; never relabel to meet quotas.',
            'paired_source_design': 'Two calibration sources per family, four planned variants each; known/UNKNOWN recruitment targets within every family.',
            'future_holdout_policy': 'Source ID/SHA reservation only; no generation, viewing or inference.',
            'source_partition_status': partition['status']}
