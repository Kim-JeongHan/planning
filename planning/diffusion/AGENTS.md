# AGENTS.md (planning/diffusion)

## 1) Submodule Summary

이 디렉토리는 경로 계획을 위한 확산 기반 샘플러의 학습/추론 파이프라인을 담당한다.

- 주요 언어: Python 3.10+
- 핵심 의존성: PyTorch, numpy, pydantic
- 선택/조건 의존성:
  - `torchvision`/`tensorboard`/`matplotlib`/`viser`(옵션/분석)
  - CUDA 가속(환경에 따라 선택)
- 핵심 책임:
  - 노이즈 스케줄링과 확산 단계 관리
  - 데이터셋 구성(샘플/조건부 정보/정규화)
  - 학습 루프, 체크포인트/로그/평가
  - 샘플 생성 및 후처리

## 2) 핵심 운영 원칙

1. 명세 우선: 기능 구현 전 요구사항, 비요구사항, 경계조건, 실패 모드, 성능 목표를 정리한다.
2. 작은 단위 변경: public API 호환성을 우선 유지하며 PR 단위로 분해한다.
3. 재현성 우선: seed, step 수, 하이퍼파라미터, 데이터 split/버전 정보를 항상 기록한다.
4. 안전한 변경: 데이터 경로, 체크포인트, 설정값은 검증 후 사용한다.
5. 테스트 필수: 새 동작/버그 수정은 테스트를 동반한다(성공/실패 케이스 포함).

## 3) 코딩 및 품질 규칙

- `black + ruff + isort` 스타일 준수, 라인 길이 100.
- 타입 힌트 적극 사용 (특히 public API, 모델 config, 데이터 구조).
- 공개 API는 최소화, 변경 시 하위 호환 정책 또는 deprecation 로그를 남긴다.
- 함수/클래스에 docstring 작성(목적, 인자, 반환값, 예외).
- `print` 대신 `logger` 사용(로그 레벨/구조화 필드 정의).
- 하드코딩 금지: 설정은 config/CLI/env에서 주입.
- 실패 모드별 예외 클래스 또는 명확한 `ValueError`/`TypeError` 메시지 사용.
- `path` 유효성(존재/접근권한/확장자/범위) 검증 후 파일 I/O 수행.
- 경로/체크포인트 로딩 시 안전하지 않은 역직렬화/실행 패턴 배제.

## 4) 모듈 구조(권장)

### A) Core
`planning/diffusion/models`, `planning/diffusion/sampling`
- 모델/샘플러의 수학적 핵심 로직 유지.

### B) Data
`planning/diffusion/training/dataset.py`
- 데이터셋 로딩/검증/정규화/배치 생성.

### C) Training
`planning/diffusion/training/trainer.py`, `planning/diffusion/training/noise.py`
- 학습 스텝, 옵티마이저/스케줄러, mixed precision, 체크포인트 관리.

### D) Runtime/Utils
`planning/diffusion/utils.py`
- 공통 유틸(시드, 로그, I/O 유틸, 설정 변환).

레이어 간 의존은 `Core <- Data <- Training <- CLI` 방향을 유지하고, 상위 레이어에서 하위 유틸로의 과도한 역참조를 피한다.

## 5) 문제 처리 프로세스 (요청 대응 형식)

A. 문제 이해: 요구사항(필수/선택), 제약조건, 입력/출력, 실패 케이스를 우선 정리한다.  
B. 설계 제안: 모듈 구조, public API, 데이터 구조, 예외·로그·설정 전략, 테스트 전략을 제시한다.  
C. 구현 계획: 작은 작업 단위의 체크리스트/커밋 계획을 작성한다.  
D. 코드 작성: 변경 파일 목록과 패치 또는 코드 블록을 제시하고, 호환성 영향을 명시한다.  
E. 검증: 단위/통합 테스트 및 추가 엣지 케이스 점검 계획을 제시한다.  
F. 문서화: README/CHANGELOG/ADR 업데이트 항목을 반영한다.

## 6) 문서/테스트/체크리스트

- 새 기능 추가 시 최소:
  - 하나 이상의 단위 테스트
  - 하나 이상의 실패 경로 테스트
  - 재현성 테스트(시드 고정, 기대치 고정)
- 권장 실행:
  - `uv run pytest tests/test_diffusion_trajectory_one_shot_example.py -v`
  - `uv run pytest tests/test_diffuser_sampling.py -v`
  - `uv run ruff check planning/diffusion/`
  - `uv run black --check planning/diffusion/`
  - `uv run mypy planning/diffusion/`

## 7) 금지사항

- 불명확한 요구를 그대로 구현하지 말고, 가정/확신/미확정 항목을 분리해 명시한다.
- 경로 traversal, 임의 코드 실행, pickle 무검증 로드, 광범위한 파일 쓰기 권한 암묵 허용을 기본 동작으로 두지 않는다.
- 성능 개선 주장은 벤치마크 설계 또는 측정 근거 없이 단정하지 않는다.
- 기존 동작과 충돌이 예상되면, 대안(비파괴/단계적 마이그레이션)을 먼저 제시한다.

## 8) 참고

`planning` 루트 AGENTS와 상충하는 규칙이 있으면 루트 규칙이 우선한다.  
단, `planning/diffusion`의 추가 규칙은 위 내용으로 보완한다.
