<!-- lint:disable F01,B01,D01,S01 -->
# AI 펩타이드-단백질 결합 후보 예측 파이프라인 (peptide_binding_mvp)

> 이 파일은 Claude Code 세션 시작 시 자동 로드되는 요약 컨텍스트다 (작성 2026-11-08, 제이 + Claude).
> 상세 근거 문서는 아래 "근거 문서 위치"를 필요할 때만 읽는다. 200줄을 넘기지 않는다.

## 1. 목적
- 타깃 단백질에 결합하는 펩타이드 후보를 자동 생성, 평가, 순위화한다. 기존 수작업(MOE) 대비 자동화가 목표.
- 1차 타깃: PD-L1 (서열 227 aa, `pepbind08.py`의 `TARGET_SEQUENCE`). 2차 타깃: IL-13Rα2 (PDB 3LB6에서 수용체 체인 추출, 3BPO 참조, 시작 대기 중).
- 사용자: 제이(김범중, 대학원생). 지도교수 호칭은 "제이콥". 문서에서 "선생님/교수님" 호칭 금지.

## 2. 메인 코드와 실행 환경
- 메인 코드: `notebooks/pepbind08.py` (약 5,400줄, 2026-11-09 생성). `pepbind07.py` 기반이며 FinalScore에서 PLIP을 제외(0.60/0.30/0.10), 기본값을 100/9/RUN_ADCP=0으로 되돌리고, RUN_ADCP=0이면 최종 Excel에서 ADCP 컬럼(`ADCP_score`, `ADCP_padded`, `ADCP_seq_used`)을 제외한다(`filter_adcp_headers()` 한 곳에서 처리).
- 이전 버전: `pepbind06.py`(약 4,900줄)와 `pepbind07.py`(ADCP 짧은 펩타이드 패딩, .trg 자동 생성, 부분 결과 파싱 추가)는 수정하지 않고 보존한다. 06, 07, 08 통합 방침은 **제이가 결정 중**. 08은 아직 공용 PC에서 실행 검증 전이다.
- 코드는 이 PC(`C:\Users\kbjoo\Documents\GitHub\Study\graduate_school\peptide_binding_mvp`)에 있고, **실제 실행은 연구실 공용 PC(Windows 10) 안의 WSL2**에서 한다. 이 PC에서는 파이프라인을 돌리지 않는다.
- 공용 PC: i7-12700, RTX 3070 8GB, RAM 32GB, WSL2 Ubuntu 22.04.5, conda env `pep_o` (Python 3.11.14, 2026-04 생성, pepbind08 실행 기준). 같은 PC에 구 env `pepbind_openmm`, `pepbind`도 남아 있다.
- Jupyter 커널 이름: `python3` (pep_o env 내장 커널, 경로 `~/miniconda3/envs/pep_o/share/jupyter/kernels/python3`). 사용자 등록 커널은 `pepbind`, `pepbind_openmm` 두 개이며 pep_o용 별도 커널은 없다. 노트북에서 구 커널 `pepbind_openmm`을 고르지 않도록 주의.
- 기본 작업 폴더: 환경변수 `PEPBIND_BASE_DIR` (기본 `~/work/pipeline`).
- 실행 예: `cd ~/work/pipeline; source ~/miniconda3/etc/profile.d/conda.sh; conda activate pep_o; python3 pepbind08.py`
- 환경 정의: `notebooks/pepbind_full.yml`, `notebooks/pepbind_min.yml`
- 외부 툴: ColabFold(AF-Multimer), OpenMM, AutoDock Vina, PLIP, PRODIGY, OpenBabel, PyTorch+Transformers. ADCP는 미설치(RUN_ADCP=0).
- 오프라인 MSA DB(UniRef30 + PDB70, 약 120GB) 구축은 계획 단계.

## 3. 파이프라인 단계 (pepbind06~08 공통)
| 단계 | 도구 | 역할 |
|---|---|---|
| 2 | PepMLM (ESM-2) | 펩타이드 서열 생성 (GPU 필수) |
| 3 | ColabFold Multimer | 복합체 구조 예측, ipTM 산출 |
| 3b | OpenMM | 에너지 최소화 + GBSA |
| 4 | AutoDock Vina | 결합 점수 (score_only 모드) |
| 4b | ADCP | 펩타이드 전용 도킹, 비교용, 기본 꺼짐 |
| 5 | PLIP | 상호작용 분석, 보조 지표 |
| 6 | PRODIGY | 친화도 ΔG 예측 |
| 7 | 재시도 | 실패 복합체를 ColabFold부터 재실행 |
| 8 | 출력 | 최종 Excel + PDB zip, FinalScore 순위 |

- 재시도: GBSA > 100 kcal/mol 또는 OpenMM 실패 또는 체인 간 최소거리 < 1.5 Å이면 실패 판정. 최대 3회(`MAX_RETRY_ROUNDS`), 시드 오프셋 100(`RETRY_RANDOM_SEED_OFFSET`), 개선된 결과만 병합.
- 정규화 범위(고정): ipTM [0, 1], Vina [-15, 0], PRODIGY ΔG [-20, 0].

## 4. FinalScore 현황 (pepbind08.py에서 문서와 코드 일치)
| 구분 | 수식 | 출처 |
|---|---|---|
| **pepbind08.py 코드 (현행)** | 0.60 x norm(PRODIGY) + 0.30 x norm(Vina) + 0.10 x norm(ipTM). 상수 W_PRODIGY 0.60, W_VINA 0.30, W_IPTM 0.10, W_PLIP 삭제, 계산 조건에서 PLIP 제외 | `pepbind08.py`, `GUIDE_pepbind06.md` 변경 2 |
| 구 안 (pepbind06, 07에 잔존) | 0.50 PRODIGY + 0.25 Vina + 0.15 PLIP + 0.10 ipTM | PRD 2026-02-20, 06/07 코드 |

- `pepbind06.py`, `pepbind07.py`는 여전히 구 안(PLIP 포함)이다. 최신 FinalScore가 필요하면 반드시 08을 쓴다.
- PLIP 값은 Excel(rank, all_metrics)에 계속 기록한다. norm_debug 시트는 PLIP 컬럼을 `norm_PLIP_weighted_total(참고)`, `w_PLIP(미적용)`(0.0), `contrib_PLIP(미적용)`(빈 값)으로 표시한다.
- 기본값: `pepbind08.py`와 `pepbind06.py`는 NUM_PEPTIDES=100, PEPTIDE_LENGTH=9, RUN_ADCP=0. `pepbind07.py`는 5, 5, RUN_ADCP=1.
- 08에서 RUN_ADCP=0이면 rank, all_metrics 시트에 ADCP 컬럼이 없고, RUN_ADCP=1이면 3개 컬럼이 유지된다.
- 08 헤더 docstring은 갱신했다. 06, 07의 docstring은 아직 "pepbind05.py" 설명이다.

## 5. 확정 결정 (제이/제이콥 승인, 임의 변경 금지)
- PLIP은 FinalScore에서 제외하고 보조 지표로만 기록 (2026.03.10 교수님 승인). 값은 Excel에 계속 기록.
- ADCP는 현재 구조와 정합성이 맞지 않아 부적합 판단 (2026.05.08). 4-mer 이하는 crankpep segfault, 5-mer 이상 사용 (`notebooks/ADCP_short_peptide_issue.md`).
- MM-GBSA는 중장기 후보. RTX 3070 8GB 한계와 Colab Pro+ ColabFold 불안정 때문에 연구실 오프라인 PC로 이전했다.
- 가중치 최적화 기준(ground truth)은 MOE GBVI 값. 목표는 GBVI 순위와 FinalScore 순위의 Spearman 상관 최대화.
- GBSA 이상값은 코드 버그가 아니라 ColabFold 예측 실패 또는 불리한 형태 때문. 최종 순위에서 GBSA_bind > 100 필터링 권장 (조사보고서 2차 분석).

## 6. 진행 중, 예정 작업
- (완료, 실행 검증 대기) 코드와 가이드의 FinalScore 불일치 해소: pepbind08.py에 반영. 공용 PC에서 소규모(NUM_PEPTIDES=2) 실행 후 FinalScore 수동 계산 대조 필요
- 결합 부위 지정 기능 (PRD상 우선순위 높음)
- ML 가중치 최적화 모듈 (GBVI 기준)
- GBSA를 GBVI 계열로 대체하는 방안 검토
- 오프라인 MSA DB 구축
- IL-13Rα2 타깃 적용 (동료의 비만 약물 과제 종료 후 시작, ED-B/2FNB 과제와 독립)
- 현재 진행 상태 및 다음 작업: **(제이가 추후 기입)**

## 7. 폴더 지도
- `notebooks/`: pepbind01~08.py, `GUIDE_pepbind06.md`, `RULES.md`, `ADCP_short_peptide_issue.md`, 환경 yml, 실행 결과 `PDP_*/`, `Reference code/`
- `scripts/`: ZDOCK 후처리 (clean_pdb, batch_clean_pdb, parse_zdock_scores)
- `data/`, `results/`: 샘플 PDB, 2025-07 모델 결과, 최종 순위 xlsx
- `README.md`: 파이프라인 01~04 이력 요약 (Colab 단계)
- 무시: `notebooks/.claude/worktrees/` (과거 워크트리)

## 8. 코딩 규칙 (notebooks/RULES.md 요약)
- 응답, 주석, 문서, 커밋 메시지는 한국어. 커밋은 Conventional Commits (`feat: ...`).
- 변수, 함수 snake_case, 클래스 PascalCase, 상수 UPPER_SNAKE_CASE, docstring은 Google Style.
- 파일 삭제 금지 대신 `_to_delete/`로 이동. 이전 버전은 `99 이전 버전/`.
- 코드 수정 전 현재 값을 먼저 읽고, 수정 후 파일을 다시 열어 반영을 확인한 뒤 보고한다.
- 문서(md)에는 엠대시와 화살표 기호를 쓰지 않는다. 시각은 KST.

## 9. 보안
- `HuggingFace Access token.txt`는 비밀 키 파일이다. **절대 읽거나 출력하거나 문서, 커밋에 넣지 않는다.** (권장: `.claude/settings.json`의 deny 규칙 + `.gitignore`)
- git 저장소 루트는 `C:\Users\kbjoo\Documents\GitHub\Study` (원격 github.com/kbjung/Study). 이 폴더는 그 하위 `graduate_school/peptide_binding_mvp`이다. 토큰 파일이 과거에 커밋됐는지는 아직 **미확인**.
- 이 폴더의 `.claude/settings.json`에 토큰 파일 읽기 금지 규칙, `.gitignore`에 제외 규칙을 넣었다.

## 10. 근거 문서 위치 (필요할 때만 읽기)
기준 폴더: `C:\Work\대학원\01_연구과제\AI 기반 펩타이드-단백질 결합 후보 예측\`
| 내용 | 위치 |
|---|---|
| 보고서 시리즈 컨텍스트 (확정 결정 4-3, 환경 4-4) | `01 파이프라인 설명서\보고서\CLAUDE.md` |
| PRD (2026-02-20, 가중치는 구버전) | `10 PRD with Claude\AI_펩타이드_파이프라인_PRD_20260220.docx` |
| 재시도 기능 구현 walkthrough | `07 클로드 인수인계 문서\pepbind05.py_구현_완료_walkthrough.pdf` |
| GBSA 이상값 조사 2차 | `07 클로드 인수인계 문서\GBSA_이상값_조사_보고서_(2차_분석).pdf` |
| IL-13Rα2 타깃 요약 | `IL-13Rα2 타겟 펩타이드 개발\IL13Ra2_target_summary.md` |
| 마스터 규칙 | `C:\Work\CLAUDE.md`, `C:\Work\대학원\CLAUDE.md` |
- 보고서 CLAUDE.md의 `generate_v7.js` 빌드 경로(`/sessions/...`)는 Cowork 전용이라 현재 재생성 불가.
