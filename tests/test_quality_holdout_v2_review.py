"""Read-only checks of human GT recorded before quality measurements."""
import hashlib
import json
from pathlib import Path
import pytest
from eden_vision.classification.taxonomy import parse_taxonomy
from eden_vision.classification.hierarchy import HierarchyResolver

ROOT=Path('/Users/dev/Datasets/eden-vision-open-set-quality-holdout-v2')
REVIEW=ROOT/'holdout-v2-review.jsonl'
UNKNOWN={f'quality-holdout-v2-{i:03d}' for i in (5,6,9,13,14,21,22)}

@pytest.fixture(scope='module')
def rows():
    if not REVIEW.exists():pytest.skip('external human review not installed')
    return [json.loads(l) for l in REVIEW.read_text().splitlines()]


def test_counts_exact_unknown_set_and_human_method(rows):
    assert len(rows)==24
    assert len({r['candidate_id'] for r in rows})==24
    assert {r['candidate_id'] for r in rows if r['unknown']}==UNKNOWN
    assert sum(not r['unknown'] for r in rows)==17
    assert all(r['review_method']=='human_visual_review_before_quality_metrics' for r in rows)
    assert all(not rows[i-1]['unknown'] for i in (1,2,10,17,18))


def test_metadata_and_image_sha_alignment(rows):
    metadata=[json.loads(l) for l in (ROOT/'metadata.jsonl').read_text().splitlines()]
    byid={r['candidate_id']:r for r in metadata}
    assert len(metadata)==len(byid)==24
    assert set(byid)=={r['candidate_id'] for r in rows}
    for r in rows:
        for key in ('source_id','source_image_id','transformation','generated_sha256'):
            assert r[key]==byid[r['candidate_id']][key]
        image=ROOT/'generated'/f'{r["candidate_id"]}.jpg'
        assert hashlib.sha256(image.read_bytes()).hexdigest()==r['generated_sha256']


def test_taxonomy_ancestors_and_consistency(rows):
    path=Path(__file__).resolve().parents[1]/'src/eden_vision/resources/taxonomy_v2.json'
    resolver=HierarchyResolver(parse_taxonomy(path.read_text()))
    assert resolver.taxonomy_version=='eden-v2'
    for r in rows:
        ancestors=list(dict.fromkeys(a for c in r['expected'] for a in resolver.ancestor_path(c)[:-1]
                                     if a not in r['expected']))
        assert r['acceptable_ancestors']==ancestors
        assert r['unknown']==(not r['expected'])==('unknown' in r['tags'])
        assert len(r['tags'])==len(set(r['tags']))
        assert set(r['tags']) <= {'unknown','ambiguous','single_object','multi_object','scene','broad_only'}


def test_gt_fingerprint_and_preregistration(rows):
    assert hashlib.sha256(REVIEW.read_bytes()).hexdigest()==REVIEW.with_suffix('.sha256').read_text().strip()
    p=ROOT.parent/'eden-vision-open-set-quality-calibration-v2/reports/quality-gate-preregistration.json'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==p.with_suffix('.sha256').read_text().strip()==(
        '5e724f49d7efeb96a95fc19e4c18be89f2daa7a080172b28ca2abc0e32f69bdf')


def test_review_has_no_metrics_or_rule_predictions(rows):
    required={'candidate_id','source_id','source_image_id','transformation','generated_sha256',
              'expected','acceptable_ancestors','unknown','tags','notes','review_method'}
    assert all(set(r)==required for r in rows)
