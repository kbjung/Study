# pepbind05.py 구현 완료 Walkthrough

## 개요
`pepbind04.py`를 기반으로 **실패 복합체 자동 재시도 기능**을 추가한 `pepbind05.py`를 구현했습니다.

## 새로운 기능

### STEP 8: 실패 복합체 자동 재시도
1. **실패 판정 기준**:
   - GBSA > 100 kcal/mol
   - OpenMM 정제 실패 (원본 ColabFold PDB 사용)
   - 원자 충돌 감지 (최소 원자간 거리 < 1.5Å)

2. **재시도 프로세스**:
   - 다른 random seed로 ColabFold 재실행
   - OpenMM 정제 수행
   - Vina/PLIP/PRODIGY 재평가
   - 개선된 결과만 원본에 병합

3. **최대 3회까지 재시도** (설정 가능)

## 추가된 설정 변수

| 변수명 | 기본값 | 설명 |
|--------|--------|------|
| `MAX_RETRY_ROUNDS` | 3 | 최대 재시도 횟수 |
| `GBSA_FAILURE_THRESHOLD` | 100.0 | GBSA 실패 판정 임계값 |
| `RETRY_RANDOM_SEED_OFFSET` | 100 | 재시도 시 seed 오프셋 |
| `RUN_RETRY` | True | 재시도 기능 활성화 여부 |

## 추가된 함수

| 함수명 | 역할 |
|--------|------|
| [identify_failed_complexes()](file:///wsl.localhost/Ubuntu/home/aisys/work/pipeline/pepbind05.py#L3420) | GBSA/OpenMM 기준으로 실패 복합체 식별 |
| [_check_min_interchain_distance()](file:///wsl.localhost/Ubuntu/home/aisys/work/pipeline/pepbind05.py#L3502) | 원자 충돌 감지용 거리 계산 |
| [run_colabfold_for_subset()](file:///wsl.localhost/Ubuntu/home/aisys/work/pipeline/pepbind05.py#L3532) | 특정 펩타이드만 ColabFold 재실행 |
| [process_retry_complexes_pipeline()](file:///wsl.localhost/Ubuntu/home/aisys/work/pipeline/pepbind05.py#L3601) | 재시도 복합체 전체 파이프라인 실행 |
| [merge_retry_results()](file:///wsl.localhost/Ubuntu/home/aisys/work/pipeline/pepbind05.py#L3667) | 재시도 결과를 원본에 병합 |

## 파이프라인 흐름 변경

```
기존 (pepbind04):
STEP 1-6 → STEP 7(zip+Excel) → 종료

신규 (pepbind05):
STEP 1-6 → STEP 7(zip) → STEP 8(재시도 루프) → STEP 9(최종 Excel) → 종료
```

## 검증 결과
- ✅ Python 구문 검사 통과 (`py_compile`)
- ✅ 기존 기능 호환성 유지

## 실행 방법
```bash
cd /home/aisys/work/pipeline
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pepbind_openmm
python3 pepbind05.py
```
