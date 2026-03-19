# pepbind06.py 코드 수정 가이드

> **목적**: 이 문서를 읽고 현재 코드 상태와 수정 사항을 파악한 뒤, 코드 수정 작업을 진행합니다.
> **코딩 규칙**: `RULES.md` 참조 (한국어 주석, snake_case, Google Style docstring)
> **환경**: conda env `pepbind_openmm`, Python 3.11, WSL2 Ubuntu 22.04.5 LTS

---

## 1. 프로젝트 개요

AI 기반 단백질 결합 펩타이드 후보 예측 시스템의 메인 파이프라인 코드입니다.

```
[타겟 서열] → PepMLM → ColabFold → OpenMM → Vina → PLIP → PRODIGY → 재시도 → 최종 출력
```

## 2. pepbind05.py → pepbind06.py 변경 내역

pepbind06.py는 pepbind05.py에서 아래 3가지를 수정한 버전입니다.

### 변경 1: Vina score_only 모드 전환

- **변경 전**: Vina가 도킹(포즈 탐색)을 수행하고 최적 포즈의 점수를 사용
- **변경 후**: `--score_only` 옵션으로 기존 구조를 재점수화 (포즈 탐색 없음)
- **이유**: 복합체 구조는 이미 ColabFold + OpenMM으로 예측/안정화 완료. Vina는 해당 구조의 결합 안정성을 점수화하는 역할만 수행
- **수정 위치**:
  - `parse_vina_score_from_stdout()` 함수: score_only 출력 형식(`Affinity: -7.5 (kcal/mol)`) 파싱 우선 추가, 기존 도킹 모드 파싱은 폴백으로 유지
  - `run_vina_on_rank1()` 함수 내 vina_cmd: `--center_x/y/z`, `--size_x/y/z`, `--out` 제거, `--score_only` 추가
  - docstring 및 출력 메시지: "도킹" → "스코어링 (score_only)"

### 변경 2: PLIP 최종점수(FinalScore) 제외

- **변경 전**: `FinalScore = 0.50×PRODIGY + 0.25×Vina + 0.15×PLIP + 0.10×ipTM`
- **변경 후**: `FinalScore = 0.60×PRODIGY + 0.30×Vina + 0.10×ipTM` (PLIP 제외)
- **근거**: PLIP은 결합 패턴 탐지(정성 도구)로 설계됨, 정량 점수화 선행 사례 없음 (2026.03.10 교수님 승인)
- **PLIP 값 자체는 엑셀에 그대로 기록됨** (보조 지표 참고용)
- **수정 위치**:
  - 가중치 상수: `W_PLIP = 0.15` 삭제, `W_PRODIGY`를 0.60, `W_VINA`를 0.30으로 변경
  - FinalScore 계산 로직: `w_plip * plip_norm` 제거, 계산 조건에서 `plip_ok` 제거
  - NORM_DEBUG_HEADERS: PLIP 관련 컬럼명에 `(참고)`, `(미적용)` 표시 추가
  - 디버그 시트: `c_plip = None`, `w_PLIP(미적용) = 0.0`

### 변경 3: ADCP (AutoDock CrankPep) STEP 4b 추가

- **위치**: STEP 4 (Vina)와 STEP 5 (PLIP) 사이
- **목적**: 펩타이드 전용 도킹 도구로, ADCP 랭킹을 기준(ground truth)으로 삼아 FinalScore 가중치를 최적화하기 위한 비교 평가용
- **FinalScore에 미반영** (보조 지표)
- **기본 비활성화**: `RUN_ADCP = 0` (ADCP 미설치 상태. 설치 후 `RUN_ADCP=1`로 변경)
- **수정 위치**:
  - 설정 상수 추가: `ADCP_CMD`, `RUN_ADCP`, `ADCP_NUM_STEPS`, `ADCP_NB_RUNS`, `ADCP_MAX_CORES`
  - 함수 추가: `parse_adcp_best_energy()`, `run_adcp_on_rank1()`
  - 헤더 상수: `RANK_TABLE_HEADERS`, `ALL_METRICS_HEADERS`에 `ADCP_score(kcal/mol)` 컬럼 추가
  - 메인 실행부: STEP 4 뒤에 STEP 4b 호출 코드 추가
  - 최종 엑셀: adcp_summary.xlsx에서 읽어 rows에 병합하는 로직 추가
  - ADCP 실행 조건: `.trg` 타겟 파일 필요 (AGFR로 사전 준비)

---

## 3. 현재 파이프라인 단계별 구성

| 단계 | 도구/모델 | 역할 | 비고 |
|------|---------|------|------|
| STEP 1 | - | 입력/경로 설정 | 타겟 서열, 펩타이드 길이, 생성 개수 |
| STEP 2 | PepMLM (ESM-2) | 펩타이드 후보 서열 생성 | GPU 필수 |
| STEP 3 | ColabFold (AF-Multimer) | 복합체 3D 구조 예측 | GPU 우선, CPU 폴백 |
| STEP 3b | OpenMM | 에너지 최소화 + GBSA 산출 | GPU 우선, CPU 폴백 |
| STEP 4 | AutoDock Vina | 결합 안정성 점수 (score_only) | CPU only |
| STEP 4b | ADCP (CrankPep) | 펩타이드 전용 도킹 | 비교 평가용, 기본 비활성화 |
| STEP 5 | PLIP | 상호작용 분석 | 보조 지표 (FinalScore 미반영) |
| STEP 6 | PRODIGY | 친화도 예측 (ΔG, Kd) | FinalScore 60% |
| STEP 7 | - | 실패 복합체 재시도 | GBSA>100 시 ColabFold부터 재실행, 최대 3회 |
| STEP 8 | - | 최종 Excel + PDB zip 출력 | FinalScore 랭킹 |

## 4. 현재 FinalScore 수식

```
FinalScore = 0.60 × norm(PRODIGY_dG) + 0.30 × norm(Vina_score) + 0.10 × norm(ipTM)
```

- PLIP: 엑셀에 값 기록은 유지하되, FinalScore 계산에서 제외
- ADCP: 엑셀에 값 기록은 유지하되, FinalScore 계산에서 제외
- 정규화: 각 점수를 사전 정의된 고정 범위(min/max)로 0~1 스케일링

### 정규화 범위

| 지표 | 범위 | 방향 |
|------|------|------|
| ipTM | [0.0, 1.0] | 클수록 좋음 |
| Vina | [-15.0, 0.0] | 음수일수록 좋음 |
| PRODIGY ΔG | [-20.0, 0.0] | 음수일수록 좋음 |

## 5. ADCP 활성화 방법 (설치 후)

1. ADCP 설치: https://ccsb.scripps.edu/adcp/
2. AGFR로 타겟 단백질의 `.trg` 파일 준비
3. `.trg` 파일을 프로젝트 base 폴더에 `target.trg`로 배치
4. 환경변수 `RUN_ADCP=1` 설정 또는 코드 상단 `RUN_ADCP` 값 변경

## 6. 주요 설정 상수 (코드 상단)

| 상수 | 현재 값 | 설명 |
|------|--------|------|
| W_PRODIGY | 0.60 | PRODIGY 가중치 |
| W_VINA | 0.30 | Vina 가중치 |
| W_IPTM | 0.10 | ipTM 가중치 |
| RUN_ADCP | 0 (비활성화) | ADCP 실행 여부 |
| ADCP_NUM_STEPS | 2,500,000 | ADCP MC 시뮬레이션 단계 수 |
| ADCP_NB_RUNS | 50 | ADCP 도킹 실행 횟수 |
| MAX_RETRY_ROUNDS | 3 | 실패 복합체 최대 재시도 횟수 |
| GBSA_FAILURE_THRESHOLD | 100.0 | GBSA 이상치 판정 기준 (kcal/mol) |

## 7. 이후 작업 예정

- ADCP 설치 및 `.trg` 파일 준비 후 ADCP 실행 테스트
- ADCP 랭킹 vs FinalScore 랭킹 비교 후 가중치 최적화
- MOE GBVI 값과 ADCP 값 상관성 분석
