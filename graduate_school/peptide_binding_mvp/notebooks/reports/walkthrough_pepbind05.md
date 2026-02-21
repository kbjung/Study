# pepbind05.py 구현 완료 Walkthrough

## 개요
`pepbind04.py`를 기반으로 **실패 복합체 자동 재시도 기능**을 추가한 `pepbind05.py`를 구현했습니다.

## 파이프라인 흐름

```
STEP 1: 워크스페이스 생성
STEP 2: PepMLM 펩타이드 생성
STEP 3: ColabFold 구조 예측
STEP 3b: OpenMM 정제 (minimize + MD)
STEP 4: AutoDock Vina 도킹
STEP 5: PLIP 상호작용 분석
STEP 6: PRODIGY 결합 친화도 평가
STEP 7: PDB zip 생성

★ STEP 8: 실패 복합체 자동 재시도 (최대 3회)
   └─ GBSA > 100 또는 OpenMM 실패 시
   └─ 다른 random seed로 ColabFold 재실행
   └─ OpenMM → GBSA 계산 → Vina/PLIP/PRODIGY 실행
   └─ GBSA <= 100 인 경우에만 결과 병합

STEP 9: 최종 엑셀 파일 생성 (재시도 결과 포함)
```

## STEP 8 실패 복합체 재시도 기능

### 실패 판정 기준
1. GBSA > 100 kcal/mol
2. OpenMM 정제 실패 (원본 ColabFold PDB 사용)
3. 원자 충돌 감지 (최소 원자간 거리 < 1.5Å)

### 재시도 프로세스
1. 다른 random seed로 ColabFold 재실행
2. OpenMM 정제 수행
3. GBSA 계산 (정제 성공 시)
4. GBSA <= 100 인 경우에만:
   - Vina/PLIP/PRODIGY 재평가 (메인 디렉토리에 저장)
   - 결과 병합
5. GBSA > 100 이면 다음 재시도에서 계속 처리

### 최대 3회까지 재시도 (설정 변경 가능)

## 설정 변수

| 변수명 | 기본값 | 설명 |
|--------|--------|------|
| `MAX_RETRY_ROUNDS` | 3 | 최대 재시도 횟수 |
| `GBSA_FAILURE_THRESHOLD` | 100.0 | GBSA 실패 판정 임계값 |
| `RETRY_RANDOM_SEED_OFFSET` | 100 | 재시도 시 seed 오프셋 |
| `RUN_RETRY` | True | 재시도 기능 활성화 여부 |

## 수정된 엑셀 출력 형식

### rank 시트
- `complex_pdb` 컬럼 추가 (peptide_seq 다음 위치)
- 컬럼 순서: rank, candidate_id, peptide_seq, **complex_pdb**, AlphaFold_status, ...

### all_metrics 시트
- `rank` 컬럼 맨 앞으로 이동
- 컬럼 순서: **rank**, candidate_id, peptide_seq, complex_pdb, ...

## 주요 함수

| 함수명 | 역할 |
|--------|------|
| `identify_failed_complexes()` | Excel/CSV에서 GBSA 값 읽어 실패 복합체 식별 |
| `run_colabfold_for_subset()` | 특정 펩타이드만 ColabFold 재실행 |
| `process_retry_complexes_pipeline()` | 재시도 복합체 전체 파이프라인 실행 |
| `merge_retry_results()` | GBSA 확인 후 결과 병합 |

## 실행 방법
```bash
cd /home/aisys/work/pipeline
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pepbind_openmm
python3 pepbind05.py
```

## 검증 결과
- ✅ Python 구문 검사 통과
- ✅ 기존 기능 호환성 유지
