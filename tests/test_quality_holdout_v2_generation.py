import json
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image
from experiments import generate_quality_holdout_v2 as g


@pytest.fixture
def sources(tmp_path):
    rows=[]
    for i in range(6):
        p=tmp_path/f'{i}.png'
        image=Image.new('RGB',(24,20))
        image.putdata([((j+i)%256,(j*7)%256,(j*3)%256) for j in range(480)])
        image.save(p)
        rows.append({'source_id':f'baseline:eval-{i:03d}','source_image_id':f'eval-{i:03d}',
                     'source_path':str(p),'source_sha256':g.sha(p.read_bytes())})
    return rows


def test_preregistration_validation(tmp_path):
    p=tmp_path/'p.json';p.write_text('{}');p.with_suffix('.sha256').write_text(g.EXPECTED_SHA)
    with pytest.raises(ValueError,match='fingerprint'):
        g.validate_preregistration(p)
    assert g.validate_preregistration(g.CAL/'reports/quality-gate-preregistration.json')['rules_frozen']


def test_source_partition_validation(sources):
    calibration=[{'source_id':f'other:{i}','source_sha256':str(i)} for i in range(8)]
    p={'future_holdout_v2_reserved_only':sources,'development_calibration_v2':calibration}
    assert len(g.validate_sources(p,[]))==6
    with pytest.raises(ValueError,match='six'):
        g.validate_sources(dict(p,future_holdout_v2_reserved_only=sources[:5]),[])
    with pytest.raises(ValueError,match='overlap'):
        g.validate_sources(p,[{'source_image_id':'old','source_sha256':sources[0]['source_sha256']}])
    calibration[0]['source_id']=sources[0]['source_id']
    with pytest.raises(ValueError,match='overlap'):
        g.validate_sources(p,[])


def test_deterministic_balanced_plan(sources):
    plan=g.make_plan(sources)
    assert plan==g.make_plan(sources)
    assert plan['global_seed']==20260925
    rows=plan['candidates'];assert len(rows)==24
    assert [r['candidate_id'] for r in rows]==[f'quality-holdout-v2-{i:03d}' for i in range(1,25)]
    for source in sources:
        group=[r for r in rows if r['source_id']==source['source_id']]
        assert len(group)==4
        assert len({r['transformation'] for r in group})==4
        assert Counter(r['intended_type'] for r in group)=={'severe_candidate':2,'moderate_candidate':2}
    for row in rows:
        with g.decode_rgb(Path(row['source_path']).read_bytes()) as im:
            _,params=g.transform(im,row['transformation'],row['seed'])
        assert params==row['parameters']


def test_generation_requires_persisted_plan_and_is_reproducible(sources,tmp_path,monkeypatch):
    import experiments.image_information_diagnostics as diagnostic
    def forbidden(*args,**kwargs):
        raise AssertionError('quality metrics forbidden')
    monkeypatch.setattr(diagnostic,'measure_bytes',forbidden)
    monkeypatch.setattr(diagnostic,'measure_rgb',forbidden)
    import experiments.generate_quality_calibration_v2 as original
    monkeypatch.setattr(original,'measure_bytes',forbidden)
    before={r['source_path']:Path(r['source_path']).read_bytes() for r in sources}
    plan=g.make_plan(sources)
    out=tmp_path/'output';out.mkdir()
    for d in ('generated','contact','source'):(out/d).mkdir()
    with pytest.raises(FileNotFoundError):g.materialize(plan,out)
    assert not list((out/'generated').iterdir())
    g.write_json(out/'generation-plan.json',plan)
    rows=g.materialize(plan,out)
    assert len(rows)==24
    allowed={'source_id','source_image_id','source_path','source_sha256','candidate_id','transformation',
             'parameters','seed','intended_type','review_status','generated_sha256','global_seed'}
    for r in rows:
        assert set(r)<=allowed
        assert r['review_status']=='pending'
        raw=(out/'generated'/f'{r["candidate_id"]}.jpg').read_bytes()
        assert g.sha(raw)==r['generated_sha256']
        with g.decode_rgb(before[r['source_path']]) as image:
            result,_=g.transform(image,r['transformation'],r['seed'])
            assert g.jpeg_bytes(result)==raw
    assert all(Path(p).read_bytes()==raw for p,raw in before.items())
    assert {p.name for p in (out/'contact').iterdir()}=={'all.jpg','severe.jpg','moderate.jpg'}
    assert json.loads((out/'generation-plan.json').read_text())==plan
