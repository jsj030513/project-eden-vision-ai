"""Synthetic contract fixtures only; no model dependencies or inference."""
from dataclasses import replace
from io import BytesIO
import json
from uuid import UUID

from fastapi.testclient import TestClient
from PIL import Image
from pydantic import TypeAdapter, ValidationError
import pytest

from eden_vision.config import Settings
from eden_vision.main import create_app
from eden_vision.schemas.classification import (
    ClassificationResponse, ImageInfo, RankingItemResponse, RankingResponse,
    VisionResponse, response_from_result,
)
from eden_vision.vision.types import ModelInfo, RankingItem, RankingResult, UnknownDecision


@pytest.fixture
def ranked():
    return RankingResult(
        ranking=(
            RankingItem(1, 'cat', 'cat', 'animal', ('living_thing', 'animal', 'cat'), 27.1),
            RankingItem(2, 'animal', 'animal', 'animal', ('living_thing', 'animal'), 25.2),
            RankingItem(3, 'dog', 'dog', 'animal', ('living_thing', 'animal', 'dog'), 24.8),
        ), model=ModelInfo('synthetic-ranking-fixture', 'v1'), taxonomy_version='eden-v2',
    )


def wire(ranked):
    return response_from_result(ranked, request_id='synthetic-request', image=ImageInfo(width=4, height=3))


def test_ranking_only_roundtrip_and_separation(ranked):
    response = wire(ranked)
    data = json.loads(response.model_dump_json())
    assert data['schema_version'] == '1.1' and data['status'] == 'ranked'
    assert data['predictions'] == [] and len(data['ranking']) == 3
    assert data['unknown_decision'] == 'not_applied'
    assert data['meta']['policy_version'] == 'not_applied'
    assert data['meta']['calibration_version'] is None
    assert response.to_domain() == ranked
    assert TypeAdapter(VisionResponse).validate_json(response.model_dump_json()) == response
    for r in data['ranking']:
        assert not {'unknown', 'confidence', 'probability', 'confidence_kind'} & r.keys()
    assert set(UnknownDecision) == {UnknownDecision.NOT_APPLIED, UnknownDecision.KNOWN, UnknownDecision.UNKNOWN}


@pytest.mark.parametrize('score', [-27.1, 0., 27.1])
def test_unbounded_finite_raw_scores(ranked, score):
    value = replace(ranked, ranking=(replace(ranked.ranking[0], raw_score=score),))
    assert wire(value).ranking[0].raw_score == score


@pytest.mark.parametrize('changes', [
    {'rank': 0}, {'rank': True}, {'rank': 1.5}, {'rank': '1'},
    {'raw_score': float('nan')}, {'raw_score': float('inf')}, {'raw_score': -float('inf')},
    {'raw_score': True}, {'raw_score': '27.1'}, {'concept_id': ''}, {'display_name': ' '},
    {'score_kind': 'probability'}, {'hierarchy': ()}, {'hierarchy': ('cat', 'animal')},
    {'hierarchy': ('animal', 'animal', 'cat')}, {'broad_category': 'plant'},
    {'concept_id': 'UNKNOWN', 'hierarchy': ('UNKNOWN',)},
])
def test_invalid_item_domain_and_schema_match(ranked, changes):
    with pytest.raises((ValueError, TypeError)):
        replace(ranked.ranking[0], **changes)
    payload = wire(ranked).ranking[0].model_dump()
    with pytest.raises(ValidationError):
        RankingItemResponse.model_validate(payload | changes)


@pytest.mark.parametrize('field', ['unknown', 'confidence', 'probability', 'confidence_kind'])
def test_ranking_rejects_decision_fields(ranked, field):
    with pytest.raises(ValidationError):
        RankingItemResponse.model_validate(wire(ranked).ranking[0].model_dump() | {field: None})


@pytest.mark.parametrize('mutation', ['empty', 'gap', 'reverse', 'score_order', 'duplicate'])
def test_order_and_identity_invariants(ranked, mutation):
    rows = list(ranked.ranking)
    if mutation == 'empty':
        rows = []
    elif mutation == 'gap':
        rows[1] = replace(rows[1], rank=4)
    elif mutation == 'reverse':
        rows.reverse()
    elif mutation == 'score_order':
        rows[1] = replace(rows[1], raw_score=30.)
    else:
        rows[1] = replace(rows[0], rank=2)
    with pytest.raises(ValueError):
        replace(ranked, ranking=tuple(rows))
    data = wire(ranked).model_dump()
    data['ranking'] = [RankingItemResponse(**{f: getattr(r,f) for f in RankingItemResponse.model_fields}).model_dump() for r in rows]
    with pytest.raises(ValidationError):
        RankingResponse.model_validate(data)


def test_equal_scores_allowed_without_rescoring(ranked):
    value = replace(ranked, ranking=tuple(replace(r, raw_score=-2.) for r in ranked.ranking))
    assert [r.concept_id for r in wire(value).ranking] == ['cat', 'animal', 'dog']


@pytest.mark.parametrize('decision', [False, True, None, 'known', 'unknown'])
def test_ranked_cannot_claim_unknown_decision_applied(ranked, decision):
    with pytest.raises(ValueError):
        replace(ranked, unknown_decision=decision)
    with pytest.raises(ValidationError):
        RankingResponse.model_validate(wire(ranked).model_dump() | {'unknown_decision': decision})


@pytest.mark.parametrize('changes', [
    {'status': 'classified'}, {'predictions': [object()]}, {'policy_version': 'test-policy'},
    {'calibration_version': 'calibrated-v1'}, {'ranking': []}, {'taxonomy_version': ''},
])
def test_ranked_domain_rejects_decisions_and_invalid_metadata(ranked, changes):
    with pytest.raises(ValueError):
        replace(ranked, **changes)


@pytest.mark.parametrize('field,value', [('policy_version', 'policy-v1'), ('calibration_version', 'c1')])
def test_ranked_schema_cannot_claim_applied_policy(ranked, field, value):
    data = wire(ranked).model_dump()
    data['meta'][field] = value
    with pytest.raises(ValidationError):
        RankingResponse.model_validate(data)


def png():
    b = BytesIO()
    Image.new('RGB', (4, 3)).save(b, format='PNG')
    return b.getvalue()


def test_default_dummy_stays_10_and_old_parser_rejects_11(ranked):
    with TestClient(create_app(Settings(_env_file=None))) as client:
        response = client.post('/v1/vision/classify', files={'image': ('test.png', png(), 'image/png')})
    data = response.json()
    assert response.status_code == 200
    assert data['schema_version'] == '1.0'
    assert 'ranking' not in data and 'unknown_decision' not in data
    assert data['predictions'][0]['label'] == 'plant'
    assert data['predictions'][0]['confidence'] is None
    assert data['meta']['model_id'] == 'dummy-injected-candidates'
    assert data['meta']['taxonomy_version'] == 'eden-v1'
    assert TypeAdapter(VisionResponse).validate_python(data) == ClassificationResponse.model_validate(data)
    with pytest.raises(ValidationError):
        ClassificationResponse.model_validate(wire(ranked).model_dump())
    with pytest.raises(ValidationError):
        RankingResponse.model_validate(data)
    # Relabeling a ranking-only payload as 1.0 never silently makes it accepted.
    with pytest.raises(ValidationError):
        TypeAdapter(VisionResponse).validate_python(wire(ranked).model_dump() | {'schema_version': '1.0'})


def test_endpoint_ranking_fixture_request_id_errors_and_openapi(ranked):
    class RankingFixture:
        def classify(self, image):
            assert (image.width, image.height) == (4, 3)
            return ranked

    with TestClient(create_app(Settings(_env_file=None))) as client:
        # Replace only the result handler in this test; default factory/DI is unchanged.
        client.app.state.classification = RankingFixture()
        response = client.post('/v1/vision/classify', files={'image': ('test.png', png(), 'image/png')})
        assert response.status_code == 200
        data = response.json()
        assert UUID(data['request_id']).version == 4
        assert data['request_id'] == response.headers['X-Request-ID']
        assert RankingResponse.model_validate(data).to_domain() == ranked
        error = client.post('/v1/vision/classify')
        assert error.status_code == 422
        assert error.json()['error'] == {'code': 'INVALID_IMAGE', 'message': 'A multipart image file is required', 'retryable': False}
        assert error.headers['X-Request-ID'] == error.json()['request_id']
        openapi = client.get('/openapi.json').json()
        spec = openapi['paths']['/v1/vision/classify']['post']
        schema = spec['responses']['200']['content']['application/json']['schema']
        if '$ref' in schema:
            schema = openapi['components']['schemas'][schema['$ref'].split('/')[-1]]
        assert schema['discriminator']['propertyName'] == 'schema_version'
        assert set(schema['discriminator']['mapping']) == {'1.0', '1.1'}
        assert not spec.get('parameters')
        assert set(spec['requestBody']['content']) == {'multipart/form-data'}
