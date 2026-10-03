# Eden Vision AI

Project Eden backend/frontend와 분리된 로컬 Vision AI 서비스입니다.
기본 `eden_vision.main:app`은 legacy dummy classification을 유지합니다.
실제 CLIP ranking은 별도 `eden_vision.production:app`으로 명시적으로 실행합니다.
실행·cache·readiness 계약은 [CLIP 운영 문서](CLIP_OPERATIONS.md)를 참고하세요.

## 환경 및 설치

macOS Apple Silicon, Python 3.12 이상을 사용합니다. Python 3.12를 권장합니다.
Homebrew Python 3.12가 설치된 환경에서:

```sh
cd ~/Developer/eden-vision-ai
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

모든 패키지는 `.venv`에만 설치합니다. 위 API-only 설치에는 ML 패키지가 포함되지 않습니다.
CLIP 실행 시에는 운영 문서의 별도 pinned runtime dependencies가 필요합니다.
선택적으로 `.env.example`을 `.env`로 복사할 수 있으며 `.env`는 Git에서 제외됩니다.
기본 설정은 `APP_NAME=Eden Vision AI`, `HOST=127.0.0.1`, `PORT=8090`,
`LOG_LEVEL=INFO`입니다. HOST는 로컬 주소 `127.0.0.1`만 허용합니다.
설정 파일은 프로젝트 디렉터리에서 읽으므로 아래 명령은 해당 위치에서 실행합니다.

## 서버 실행

```sh
source .venv/bin/activate
eden-vision-ai
```

기본 바인딩: `127.0.0.1:8090`. 8090이 이미 사용 중이면 기존 프로세스를 종료하지 말고 확인합니다.
종료는 실행 터미널에서 Ctrl+C를 누릅니다.

- `/`: 서비스 이름과 API 상태
- `/health`: `{"status":"ok","service":"eden-vision-ai"}`
- Swagger: http://127.0.0.1:8090/docs

```sh
curl -i http://127.0.0.1:8090/health
lsof -nP -iTCP:8090 -sTCP:LISTEN
```

시작 로그에는 서비스 이름, 호스트, 포트만 애플리케이션 정보로 기록합니다.
외부 포트 공개, Docker 및 기존 인프라 설정 변경은 이 프로젝트 범위에 포함하지 않습니다.

## 테스트

```sh
source .venv/bin/activate
pytest
```

## 로컬 Ground Truth 라벨링 UI

production API와 분리된 human-only 앱입니다. 기존 `.venv`만 사용하며
모델·torch·transformers·외부 서비스·CDN은 사용하지 않습니다.

```sh
cd /Users/dev/Developer/eden-vision-ai
./.venv/bin/python experiments/labeling_app.py \
  --dataset /Users/dev/Datasets/eden-vision-eval \
  --host 127.0.0.1 \
  --port 8091
```

브라우저에서 http://127.0.0.1:8091 을 엽니다. 기본값도 위와 같으며 loopback
주소만 허용합니다. 종료는 실행 터미널에서 Ctrl+C를 누릅니다.

- 첫 미완료 draft를 열며, 항목 선택 메뉴에서 완료된 라벨도 다시 수정할 수 있습니다.
- 78개 eden-v2 concept를 ID/display name으로 검색하고 복수 선택합니다.
  `acceptable_ancestors`는 기존 `HierarchyResolver`로 생성합니다.
- 태그는 `evaluation_core.ALLOWED_TAGS`의 7개만 선택 가능합니다:
  `single_object`, `multi_object`, `fine_grained`, `broad_only`, `ambiguous`, `unknown`, `scene`.
  자유 입력은 없으며 각 버튼의 tooltip과 작은 도움말에 의미를 표시합니다.
  `single_object`와 `multi_object`는 UI에서 교체 선택하며 API는 동시 요청을 거부합니다.
- Unknown은 expected를 비우고 `unknown` 태그를 추가하며, 해제 시 태그도 제거합니다.
  API는 known 항목의 `unknown` 태그를 거부합니다. 알려진 항목은 concept 1개 이상,
  모든 항목은 최종 tag 1개 이상이 필요합니다. 저장 직전 기존 canonical validator로 검증합니다.
- 기존 태그와 메모를 그대로 표시하고 이미지에서 라벨이나 태그를 추론하지 않습니다.
- Save / Save & Next는 즉시 `manifest.draft.jsonl`에 atomic write합니다.
  `manifest.jsonl`과 `dataset.json`은 변경하지 않으며 시작 시 초기화하지 않습니다.
- ←/→로 이동하고 ⌘ Enter 또는 Ctrl Enter로 저장 후 이동합니다.
  입력창에서 방향키는 텍스트 편집용으로 유지됩니다. 저장 전 이동은 확인합니다.
- 같은 항목을 여러 탭에서 수정하면 revision 충돌로 오래된 저장을 거부합니다.
  충돌 시 변경 내용을 별도로 보관한 뒤 새로고침해 최신 라벨을 확인하세요.
  UI 인스턴스 간 저장은 `.labeling.lock`으로 직렬화합니다. 기존 CLI는 이 lock을
  사용하지 않으므로 라벨링 중 import/label/promote CLI나 직접 manifest 편집을
  동시에 실행하지 마세요.

이 UI는 기존 평가기나 taxonomy를 수정하지 않습니다.
현재 JPEG 데이터셋은 Safari/Chrome에서 표시 가능하며,
추후 HEIF/HEIC를 등록하면 브라우저의 해당 이미지 형식 지원 여부에 따라 표시가 제한됩니다.

이미지 입력은 JPEG, MPO primary still, PNG, static WebP, HEIC, HEIF primary still을 지원합니다.
Pillow가 MPO로 식별한 경우 frame 0만 decode하고 내부 source format은 JPEG로 처리합니다.
MIME도 JPEG와 동일하며 다른 이미지 MIME은 거부합니다. 후속 frame은 decode하거나 별도로
저장하지 않습니다. Import는 전체 원본 container bytes를 그대로 `.jpg`에 복사하므로
재인코딩 없이 원본 SHA-256을 보존합니다. 일반 animated PNG/WebP 거부 정책은 유지됩니다.
HEIF의 thumbnail, depth, auxiliary image는 분류 대상에서 제외하며, 모든 입력은
orientation 보정 후 RGB8 pixel로 정규화합니다. 업로드와 평가 dataset import에는
10 MiB 파일 및 25 MP decoded pixel 제한이 적용됩니다.

## 향후 계획

- Phase 2: Vision model baseline
- Phase 3: hierarchical classification
- Phase 4: open-set / UNKNOWN handling
- Phase 5: Project Eden Backend integration

## 참고 문서

- [Open-set 데이터셋·라벨 정책·signal collector](OPEN_SET_EVAL.md)
- [FastAPI](https://fastapi.tiangolo.com/tutorial/)
- [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
