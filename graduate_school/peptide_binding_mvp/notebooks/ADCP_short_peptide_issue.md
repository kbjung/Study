# ADCP 짧은 펩타이드(≤4-mer) 문제 정리 및 우회 전략

작성일: 2026-04-23
관련 코드: `pepbind07.py`
검증 실행: `PDP_20260423_144946` (4-mer 실패), `PDP_20260423_180318` (1.1 업그레이드 후에도 4-mer 실패), `PDP_20260423_193614` (5-mer 정상)

---

## 1. 요약

- **현상**: ADCP(AutoDock CrankPep)로 4-mer 이하 펩타이드를 도킹하면 crankpep 바이너리가 MC(Monte Carlo) 시뮬레이션 초기 단계에서 **segmentation fault**로 죽음
- **범위**: ADFRsuite 1.0(Python 2.7), 1.1(Python 3) 양쪽 모두 동일하게 발생
- **결과**: 50번 반복 실행 전부 실패 → 후속 클러스터링 단계에서 `AttributeError: Molecule instance has no attribute '_ag'` → ADCP 스코어 파싱 불가
- **확실한 해결법**: 펩타이드 길이 ≥ 5로 설정 (`PEPTIDE_LENGTH = 5`)
- **4-mer가 꼭 필요한 경우 우회책**: glycine 패딩, Vina 점수 대체, 전용 서버(HPEPDOCK) 사용 등

---

## 2. ADCP란?

ADCP(AutoDock CrankPep)는 Scripps Research CCSB에서 개발한 **단백질-펩타이드 도킹 전용 툴**입니다.

**핵심 개념:**
- AutoDock Vina가 작은 분자(small molecule) 도킹에 최적화된 반면, ADCP는 **유연한 펩타이드(3~20 amino acids) 도킹**에 특화
- 내부적으로 `crankpep`라는 C++ 바이너리로 **Replica Exchange Monte Carlo (REMC)** + **Backbone crankshaft moves** 알고리즘 사용
- 펩타이드의 backbone dihedral 각도(φ, ψ)를 Ramachandran 확률 분포에 따라 샘플링

**왜 특별한가?**
1. **Backbone flexibility**: 펩타이드는 small molecule보다 conformational space가 훨씬 크고 유연
2. **Ramachandran 제약**: 물리적으로 허용된 backbone 각도만 샘플링
3. **그리드 기반 scoring**: AGFR로 미리 계산한 AutoDock grid map 사용 → 빠른 에너지 평가
4. **대규모 샘플링**: 기본 50 runs × 2.5M MC steps = 1.25억 step

공식 논문: Zhang & Sanner (2019) *Bioinformatics* 35:5121

---

## 3. 4-mer 이하에서 실패하는 원인

### 3.1 관찰된 에러

```
Building protein from sequence: TSEE (4 amino acids, 1 chains).
...
swap in best curr 143.23 swap 9999 best 99999
Segmentation fault (core dumped)
```

이후 Python wrapper(`runADCP.py`)의 clustering 단계에서:
```python
File "clusterADCP.py", line 165, in __call__
    models = Read(syst)
  ...
AttributeError: Molecule instance has no attribute '_ag'
```

### 3.2 기술적 원인 (추정)

crankpep 바이너리는 비공개 C++ 코드라 정확한 원인 파악은 불가능하지만, 세그폴트가 **첫 MC swap 직후**에 발생하는 점을 근거로 다음 원인들이 추정됩니다.

**(a) 버퍼 오버플로우 / 언더플로우**
- crankpep이 펩타이드 길이를 기반으로 내부 배열을 할당할 때, N≤4에서 **최소 크기 가정(보통 5)** 을 위반해 배열 범위 밖 접근
- `len 4 bonds, Nchains 1:*0 x1 2+0 != (5-1+(1-3)*1` 같은 로그는 bond list 초기화에서 `(chain_length - 1 + (1 - residue_index))` 류의 인덱스 계산이 맞지 않음을 시사

**(b) Crankshaft move 기하학적 제약**
- Crankshaft move는 **2개의 pivot residue 사이의 3~4개 atoms**을 회전시킴
- 4-mer에선 회전 대상 원자가 거의 없거나 0개여서 0으로 나누기 / 빈 배열 접근 발생 가능

**(c) Replica Exchange 통계 샘플 부족**
- REMC는 replica 간 에너지 분포의 통계적 겹침이 필요
- 짧은 펩타이드는 conformational space가 작아 replica들이 거의 동일 상태에 수렴 → NaN 분산 → 세그폴트 유발 가능

**(d) Secondary Structure / Ramachandran 상호작용**
- `SS Energy diag offdiag 832.485 ...` 에서 보이듯 ADCP는 secondary structure bias를 사용
- 4-mer은 α-helix/β-sheet 형성 최소 길이에 못 미쳐, bias term 계산에서 특이점(singularity) 발생 가능성

**(e) 모던 glibc malloc 동작 변화**
- 오래된 코드가 glibc ≥ 2.26 이후의 tcache/arena 변경에 민감할 수 있음 (특히 작은 할당량에서)

### 3.3 버전 간 차이

| 버전 | Python | 동작 |
|------|--------|------|
| ADFRsuite 1.0 | 2.7.18 | 4-mer segfault (MC 몇 스텝 후) |
| ADFRsuite 1.1 | 3.x | 4-mer segfault (첫 swap 직후) — **동일 버그** |

→ **버전 업그레이드만으로는 해결 불가**. crankpep 바이너리 자체의 한계.

### 3.4 공식 ADCP 권장 최소 길이

ADCP 공식 문서 및 튜토리얼 예시는 모두 **5-mer 이상**을 사용합니다 (e.g., `GaRyMiChEL` 8-mer in help). 4-mer 이하는 **공식적으로 지원되지 않는 범위**로 봐야 합니다.

---

## 4. 우회 전략

### 전략 A. Glycine 패딩 (권장)

4-mer 펩타이드의 N- 또는 C-말단에 glycine(G)을 추가해 5-mer로 만든 후 ADCP 실행.

**장점:**
- 코드로 자동화 쉬움
- G는 side chain이 H 하나뿐이라 결합 contribution이 거의 0 (통계적으로 ~0.0 ± 0.5 kcal/mol)
- crankpep가 정상 작동

**단점:**
- 엄밀히는 "4-mer의 결합 친화도" 가 아니라 "5-mer(padded)의 결합 친화도"
- 추가된 G의 backbone이 binding site 내에서 공간을 차지해 약간의 스코어 변동 가능
- N-말단 vs C-말단 패딩에 따라 결과 약간 다름

**구현 예시:**
```python
def pad_peptide_for_adcp(pep_seq: str, min_len: int = 5,
                         side: str = "C") -> tuple[str, int]:
    """
    Returns (padded_seq, n_pads_added). side='C' 면 C-말단에 G 추가.
    """
    if len(pep_seq) >= min_len:
        return pep_seq, 0
    n_pad = min_len - len(pep_seq)
    if side == "C":
        return pep_seq + "G" * n_pad, n_pad
    else:
        return "G" * n_pad + pep_seq, n_pad
```

**보정 방법 (선택):**
- 테스트 세트에서 동일 4-mer × {N-padded, C-padded} 평균 → 패딩 영향 최소화
- 또는 벤치마크: 알려진 5-mer을 4-mer로 자르고 N/C 패딩 스코어와 비교해 보정계수 추정

### 전략 B. ADCP 대체: Vina + prodigy + PLIP 조합

4-mer의 경우 ADCP를 완전히 스킵하고 이미 파이프라인에 있는 다른 지표로 평가.

| 지표 | 역할 | 4-mer 지원 |
|------|------|------------|
| AutoDock Vina `--score_only` | 결합 에너지 (kcal/mol) | ✅ 길이 무관 |
| PRODIGY | ΔG, KD 예측 | ✅ 길이 무관 |
| PLIP | 수소결합/salt bridge 상호작용 수 | ✅ 길이 무관 |

**장점:** 추가 구현 불필요, 이미 검증된 도구들
**단점:** ADCP의 backbone sampling 정보는 얻을 수 없음

### 전략 C. 전용 펩타이드 도킹 서버/툴 사용

| 툴 | 최소 펩타이드 길이 | 비고 |
|----|------|------|
| **HPEPDOCK 2.0** | 3-mer | 중국 Huang Lab, 웹 서버 + 로컬 |
| **pepATTRACT** | 3-mer | 로컬 실행, 유연함 |
| **Rosetta FlexPepDock** | 2-mer 이상 | 매우 정확하지만 계산 비용 큼 |
| **AutoDock CrankPep (upstream)** | 5-mer 실무 | 현재 툴 |
| **Glide Peptide Docking** | 3-mer | 상용(Schrödinger) |

**권장:** 4-mer 전용이면 **HPEPDOCK 2.0** 로컬 설치가 가장 실용적.

### 전략 D. crankpep 바이너리 패치 (비권장)

ADFRsuite 소스 코드에 접근할 수 없으므로(C++ 코어는 precompiled), 패치 불가능. GitHub Issue 또는 메일링 리스트로 Sanner Lab에 버그 리포트는 가능합니다.

---

## 5. 권장 사항

### 5.1 일반 사용 시

```python
# pepbind07.py
PEPTIDE_LENGTH = 5  # 또는 그 이상
```

ADCP 성능이 가장 좋은 길이는 **6-10 amino acids**입니다. 5-mer도 정상 작동하지만 도킹 pocket이 제한되는 경우 더 긴 펩타이드가 더 잘 수렴합니다.

### 5.2 4-mer가 꼭 필요한 경우

**우선순위:**

1. **1순위 (간단)**: Vina + PRODIGY + PLIP 만으로 평가 (`RUN_ADCP=0`)
2. **2순위 (정확도 유지)**: Glycine 패딩 (C-말단) + 결과에 `padded=True` 플래그 기록
3. **3순위 (다른 도구)**: HPEPDOCK 로컬 설치해서 4-mer 도킹

### 5.3 파이프라인 코드 개선안 (선택적)

`pepbind07.py`의 `run_adcp_on_rank1` 함수에 다음 옵션 추가 가능:

```python
# 환경변수로 우회 전략 선택
ADCP_SHORT_PEP_STRATEGY = os.environ.get("ADCP_SHORT_PEP_STRATEGY", "skip")
#   "skip":   4-mer 이하는 ADCP 스킵 (현재 기본값, 안전)
#   "pad_c":  C-말단에 G 패딩하여 5-mer로 만든 후 실행
#   "pad_n":  N-말단에 G 패딩
```

---

## 6. 참고 자료

- ADCP 공식 튜토리얼: https://ccsb.scripps.edu/adcp/tutorial/
- Zhang, Y. & Sanner, M.F. (2019). "AutoDock CrankPep: combining folding and docking to predict protein-peptide complexes." *Bioinformatics* 35(24):5121–5127. DOI: 10.1093/bioinformatics/btz459
- ADFRsuite 다운로드: https://ccsb.scripps.edu/adfrsuite/downloads/
- HPEPDOCK: http://huanglab.phys.hust.edu.cn/hpepdock/
- pepATTRACT: https://bianca.science.uu.nl/pepattract/
- 본 프로젝트 실험 결과 폴더:
  - `PDP_20260423_144946` — ADFRsuite 1.0, 4-mer, 실패
  - `PDP_20260423_180318` — ADFRsuite 1.1, 4-mer, 실패 (버전 업그레이드로 해결 안 됨 확인)
  - `PDP_20260423_193614` — ADFRsuite 1.1, 5-mer, 정상 작동 (-10.4 ~ -13.7 kcal/mol)

---

## 7. 실행 환경 요약

- OS: WSL Ubuntu
- ADFRsuite: 1.1 (`/home/aisys/ADFRsuite_1.1/`), Python 3 번들
- Conda env: `pepbind_openmm`
- 코드: `pepbind07.py` (main branch + `claude/gracious-panini-a47e9e` worktree 수정)
- 검증된 정상 동작 조건: **ADFRsuite 1.1 + PEPTIDE_LENGTH ≥ 5**
