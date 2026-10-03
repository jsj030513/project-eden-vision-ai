# Open-set calibration/evaluation 기반

이 데이터셋은 CLIP + eden-v2(production candidate)와 SigLIP2 + eden-v2(challenger)의
open-set, broad fallback, UNKNOWN 처리를 검토하기 위한 별도 human-labeled 자료입니다.
현재 production API에는 연결하지 않으며, 이 단계에서는 threshold나 UNKNOWN 판정
규칙을 만들지 않습니다. Raw logit은 확률이나 confidence가 아닙니다.
모든 진단·향후 calibration은 모델별로 수행하며 모델 간 raw score 크기는 비교하지 않습니다.

## 기존 기준 데이터 동결

`/Users/dev/Datasets/eden-vision-eval`의 56개 이미지, metadata, canonical/draft manifest와
`reports/`, `reports-paddingfix/`, `reports-annotationfix/`는 역사적 baseline입니다.
새 데이터셋 작업에서 이 경로를 import/promote 대상이나 output으로 사용하지 마세요.
기존 이미지나 Ground Truth를 복사하거나 재라벨링하는 작업도 이 단계에 포함되지 않습니다.

## 소스 코드에서 확인한 계약

- 허용 태그의 단일 소스는 `experiments.evaluation_core.ALLOWED_TAGS`입니다:
  `single_object`, `multi_object`, `fine_grained`, `broad_only`, `ambiguous`, `unknown`, `scene`.
- `validate_dataset_rows` / `load_dataset`은 canonical metadata, ID, 파일 존재/경로,
  concept, 태그, UNKNOWN 일관성, image_count 및 기록된 SHA-256을 검증합니다.
- UNKNOWN의 canonical annotation 부분은 다음과 같습니다. 전체 row에는 기존 포맷의
  `id`, `file`, `sha256`, `notes`도 저장합니다. 새 manifest 포맷은 만들지 않습니다.

  ```json
  {"expected": [], "unknown": true, "tags": ["unknown"]}
  ```

- Known은 `unknown=false`, expected 최소 1개, `unknown` 태그 없음이 필수입니다.
  모든 항목에는 태그가 최소 1개 있어야 합니다. `ambiguous`는 UNKNOWN 여부와 독립적으로
  사용할 수 있습니다. 검증기는 태그 이름·일관성을 검증하며 의미 판단은 사람이 합니다.
- `single_object`는 한 종류의 주요 피사체 중심, `multi_object`는 여러 주요 대상이 평가
  대상인 경우입니다. 단순 expected 배열 길이와 동일한 개념이 아닙니다. 라벨링 UI/API는
  두 태그의 동시 선택을 금지합니다. 기존 canonical validator 자체에는 이 상호배타 검사가
  없으므로 기존 UI를 통해 라벨링합니다.
- Ancestor는 기존 라벨러의 `HierarchyResolver.ancestor_path(concept_id)[:-1]`로 생성합니다.
  루트부터 부모까지 순서이며 자기 자신은 제외합니다. 부모와 자식을 expected에 중복 삽입하지 않습니다.
- Import는 inbox 이미지 원본 바이트를 images에 복사하고 미완료 draft를 등록합니다.
  Promote는 사람이 완료한 draft만 검증하여 canonical에 추가하고 image_count를 갱신합니다.
  미완료 draft는 남기며 collector는 canonical 항목만 읽습니다.

## 샘플 라벨 정책

| 유형 | 판단 및 annotation |
|---|---|
| Broad fallback | 세부 class가 없어도 안전한 상위 concept가 있으면 known. `unknown=false`, expected에 안전한 broad concept, 필요 시 `broad_only`. smartwatch → electronic_device, tiger → animal은 현재 eden-v2의 예시입니다. UNKNOWN으로 라벨하지 않습니다. |
| Ambiguous / low-information | 심한 blur, 극단적 crop, 부분 가림, 여러 해석이 가능한 이미지에는 `ambiguous`. 안전한 broad concept가 있다면 그것을 expected로 사용하고 known을 유지합니다. 이때 `broad_only`와 함께 사용할 수 있습니다. 안전한 label이 전혀 없을 때만 UNKNOWN 후보입니다. |
| True UNKNOWN | 현재 eden-v2의 어떤 concept로도 안전하고 의미 있게 표현할 수 없을 때만 `unknown=true`, `expected=[]`, `unknown` 태그. artifact, scene, living_thing 등의 broad concept가 있으므로 세부 class 부재만으로 UNKNOWN을 선택하지 않습니다. UNKNOWN은 최후 fallback입니다. |

수집에는 일반 known 대조 샘플도 포함할 수 있습니다. 모델 예측을 보고 GT를 추천하거나
자동 생성하지 않습니다. 애매한 이유, 안전한 broad label을 선택한 이유는 사람이 notes에
기록합니다. UNKNOWN은 사람이 검수한 annotation 상태이며 runtime의 출력 class가 아닙니다.

## 별도 데이터셋 생성

Root: `/Users/dev/Datasets/eden-vision-open-set-eval`

기존 tooling에는 일반용 init 명령이 없었습니다. `prepare_eval_dataset.py init`을
추가해 기존 JSONL/metadata 포맷과 atomic write helper를 재사용합니다.
**존재하는 root는 비어 있어도 거부**하며 reset/overwrite하지 않습니다.
생성이 중간에 실패하면 부분 디렉터리를 남기므로 다른 새 경로에서 재시도하기 전에 확인하세요.

```sh
cd /Users/dev/Developer/eden-vision-ai
./.venv/bin/python experiments/prepare_eval_dataset.py \
  --dataset /Users/dev/Datasets/eden-vision-open-set-eval \
  --taxonomy eden-v2 init --dataset-id eden-vision-open-set-eval-v1
```

```text
eden-vision-open-set-eval/
  dataset.json            # id=eden-vision-open-set-eval-v1, taxonomy_version=eden-v2, image_count=0
  manifest.jsonl          # empty
  manifest.draft.jsonl    # empty
  inbox/
  images/
  rejected/
  reports/
  backups/
```

`rejected/`는 수동 보관용입니다. 기존 import는 실패 개수를 보고하고 파일을 inbox에
그대로 두며 이 폴더로 자동 이동하지 않습니다. `backups/`도 자동 백업 기능을 뜻하지 않습니다.
빈 초기 상태는 기존 canonical validator로 정상 검증됩니다. Validator를 완화하지 않았습니다.

## 향후 사람이 사진을 추가한 뒤 사용할 흐름

이번 단계에서는 실제 사진을 추가하지 않습니다. 향후 검수할 사진을 새 root의 inbox에 넣은 뒤:

```sh
./.venv/bin/python experiments/prepare_eval_dataset.py \
  --dataset /Users/dev/Datasets/eden-vision-open-set-eval --taxonomy eden-v2 import

./.venv/bin/python experiments/labeling_app.py \
  --dataset /Users/dev/Datasets/eden-vision-open-set-eval --host 127.0.0.1 --port 8091
# http://127.0.0.1:8091 — 수동 라벨링 후 Ctrl+C로 서버 종료

./.venv/bin/python experiments/prepare_eval_dataset.py \
  --dataset /Users/dev/Datasets/eden-vision-open-set-eval --taxonomy eden-v2 promote

./.venv/bin/python experiments/evaluate_models.py \
  --dataset /Users/dev/Datasets/eden-vision-open-set-eval --taxonomy eden-v2 --validate-only
```

라벨링 UI는 draft 전용이며 0개일 때 `draft 항목이 없습니다.`를 표시합니다.
Promote 후 canonical 항목은 이 UI 목록에서 제외됩니다. UI와 import/promote CLI를
동시에 실행하지 마세요. CLI는 UI의 저장 lock을 사용하지 않습니다.

## Signal collector

다음 명령은 향후 사람의 GT가 준비된 뒤 사용할 실행 예시입니다. **이번 단계에는 실행하지 않습니다.**

```sh
/Users/dev/.venvs/eden-vision-ai-model-eval/bin/python \
  experiments/collect_open_set_signals.py \
  --dataset /Users/dev/Datasets/eden-vision-open-set-eval \
  --model clip --taxonomy eden-v2 \
  --output /Users/dev/Datasets/eden-vision-open-set-eval/reports/clip-signals-001.json

/Users/dev/.venvs/eden-vision-ai-model-eval/bin/python \
  experiments/collect_open_set_signals.py \
  --dataset /Users/dev/Datasets/eden-vision-open-set-eval \
  --model siglip2 --taxonomy eden-v2 \
  --output /Users/dev/Datasets/eden-vision-open-set-eval/reports/siglip2-signals-001.json
```

한 output에는 한 모델만 포함합니다. 기존 output은 항상 거부하며 overwrite 옵션이 없습니다.
JSON은 같은 디렉터리의 임시 파일에 기록·fsync한 뒤, 기존 목적지가 있으면 실패하는
atomic link로 공개합니다. 수집 실패 시 불완전 report를 공개하지 않습니다.
수집 중 manifest/metadata/image 변경도 검사해 provenance가 달라지면 output을 거부합니다.

`evaluate_models.make_plan/evaluate_condition`과 `model_runtime.create_runtime`을 재사용합니다.
고정 revision, 로컬 snapshot, `local_files_only=True`, MPS 필수, CPU fallback 금지 정책이
그대로 적용됩니다. CLIP text padding과 수정된 SigLIP2 max-length padding도 유지합니다.
캐시가 없으면 실패하며 다운로드하거나 대체 모델을 사용하지 않습니다.

빈 canonical dataset은 모델을 생성/로드하지 않고 `status=empty_dataset`, `model_loaded=false`,
빈 images 배열과 0-count summary를 가진 JSON을 생성합니다. 이 경우 main `.venv`에서도 실행 가능합니다.

## JSON signal 및 summary 정의

- 최상위: schema_version, UTC 생성 시각, dataset/model ID·revision, annotation/candidate taxonomy,
  candidate count, runtime metadata, prompt/cache 정보, dataset/manifest/image SHA provenance.
- 각 image의 `provenance`: dataset ID, image ID·경로·SHA-256, model·model ID·revision,
  taxonomy_version, candidate_count.
- `ground_truth`: expected(concept와 acceptable_ancestors), unknown, tags, notes.
- `ranking`: top_1/top_2의 concept_id·raw_score, `top1_top2_raw_margin = score1 - score2`,
  raw score가 포함된 top_5. Raw score는 `raw_image_text_logit`이며 calibration되지 않았습니다.
- `diagnostics.top1_top5_raw_spread = score1 - score5`.
- `diagnostics.top_k_hierarchy_coherence`: k=5, 기존 `_coherence` 정의를 재사용한
  related_pair_fraction(공통 ancestor가 있는 쌍의 비율), unique_broad_categories.
  공통 broad root만 있어도 related로 계산되므로 이 값을 정답률이나 certainty로 해석하지 않습니다.
- `summary`: image_count, known_count, unknown_count, broad_only_count, ambiguous_count.
  known은 unknown=false 전체이며 broad_only/ambiguous는 해당 태그 포함 여부입니다.
  그룹들은 겹칠 수 있어 합계가 전체 수와 같을 필요가 없습니다.
- 그룹별 top1_raw_score와 top1_top2_raw_margin에 count/min/median/p95/max를 제공합니다.
  기존 percentile helper의 `(n - 1) * q` 선형 보간을 사용합니다.
  빈 그룹은 status=no_samples, count=0, 나머지 통계=null입니다.
  UNKNOWN이 없으면 unknown_status=no_unknown_samples로 명시합니다.

Softmax, probability/confidence 필드, threshold, 자동 broad fallback/UNKNOWN 결정은 없습니다.
특히 known으로 오분류되는 UNKNOWN과 안전한 broad fallback을 판단하는 운영 규칙은
후속 calibration 작업으로 남깁니다. 이번 단계는 fine-tuning·production 통합을 포함하지 않습니다.

## 검증

```sh
./.venv/bin/python -m pytest -q
```

테스트는 tmp_path의 합성 fixture와 fake runtime만 사용합니다. CLI empty 경로의 AI import 금지,
CLIP/SigLIP2 runtime 선택, canonical 검증, 그룹 통계, margin/coherence, output 보존,
빈 UI 및 import → UI label → promote 흐름을 검증합니다. 실제 사진 inference는 실행하지 않습니다.
