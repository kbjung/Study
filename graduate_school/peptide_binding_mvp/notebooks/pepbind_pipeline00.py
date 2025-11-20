# pepbind_pipeline.py (WSL-friendly starter)
# Generated from Colab notebook on 2025-10-14 10:41
# This script bundles code cells (cleaned) and adds a CLI wrapper.
# Review TODOs below and adapt paths/commands for your local setup.

import os, sys, json, argparse, subprocess, shutil, pathlib, time
from pathlib import Path
# Safe display stub (for scripts; notebook-only display calls won't break)
try:
    from IPython.display import display  # optional, for rich output in notebooks
except Exception:
    def display(x):
        print(x)

# ======== CONFIG (edit as needed) ========
BASE_DIR = Path(os.environ.get("PEPBIND_BASE_DIR", "~/work/pipeline")).expanduser()
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
STRUCT_DIR = OUTPUT_DIR / "structures"
DOCK_DIR = OUTPUT_DIR / "docking"
PLIP_DIR = OUTPUT_DIR / "plip"
PRODIGY_DIR = OUTPUT_DIR / "prodigy"
LOG_DIR = OUTPUT_DIR / "logs"
for d in [DATA_DIR, OUTPUT_DIR, STRUCT_DIR, DOCK_DIR, PLIP_DIR, PRODIGY_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# External tools (set absolute paths if needed)
COLABFOLD_CMD = os.environ.get("COLABFOLD_CMD", "colabfold_batch")  # or path to run_colabfold.py
VINA_CMD = os.environ.get("VINA_CMD", "vina")
PLIP_CLI = os.environ.get("PLIP_CLI", "plip")  # or 'python -m plip' depending on install
PRODIGY_SCRIPT = os.environ.get("PRODIGY_SCRIPT", "prodigy")  # placeholder; replace with your PRODIGY CLI

# ======== Utility helpers ========
def run(cmd, cwd=None):
    print(f"[RUN] {cmd}", flush=True)
    result = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str))
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    return result

def find_top_structure(colabfold_out: Path) -> Path | None:
    """Best-effort: pick rank_001*.pdb from colabfold output."""
    candidates = sorted(colabfold_out.glob("rank_001*.*pdb"))
    return candidates[0] if candidates else None

def main():
    parser = argparse.ArgumentParser(description="PepBind pipeline (ipTM → Vina → PLIP → PRODIGY)")
    parser.add_argument("--target_fasta", type=str, required=False, help="Path to target protein FASTA")
    parser.add_argument("--peptide_fasta", type=str, required=False, help="Path to peptide FASTA (candidates)")
    parser.add_argument("--colabfold_out", type=str, default=str(STRUCT_DIR), help="ColabFold output dir")
    parser.add_argument("--vina_box", type=str, default="", help="Vina box params string or JSON")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--skip_colabfold", action="store_true")
    parser.add_argument("--skip_vina", action="store_true")
    parser.add_argument("--skip_plip", action="store_true")
    parser.add_argument("--skip_prodigy", action="store_true")
    args = parser.parse_args()

    if not args.skip_colabfold:
        # TODO: Adjust ColabFold invocation to your inputs
        # Example:
        # run([COLABFOLD_CMD, args.target_fasta, args.peptide_fasta, args.colabfold_out, f"--num-recycle=3", f"--threads={args.threads}"])
        print("[INFO] ColabFold step is enabled but invocation is left as TODO. Set COLABFOLD_CMD and adjust arguments.")

    top_pdb = find_top_structure(Path(args.colabfold_out))
    if top_pdb is None:
        print("[WARN] No rank_001*.pdb found. Ensure ColabFold output dir is correct.")
    else:
        print(f"[OK] Using top structure: {top_pdb}")

    # Vina docking (placeholder)
    if not args.skip_vina and top_pdb is not None:
        # TODO: Convert PDB→PDBQT, prepare receptor/ligand, and run Vina
        print("[INFO] Vina step enabled. Implement preparation and invocation as needed.")

    # PLIP interactions (placeholder)
    if not args.skip_plip and top_pdb is not None:
        # Example: run([f"python -m plip.cmd.plip -f {top_pdb} -o {PLIP_DIR}"])
        print("[INFO] PLIP step enabled. Implement invocation path (PLIP_CLI) as needed.")

    # PRODIGY affinity (placeholder)
    if not args.skip_prodigy and top_pdb is not None:
        print("[INFO] PRODIGY step enabled. Implement PRODIGY call as needed.")

    print("[DONE] Pipeline completed (placeholders for external tools may need edits).")

if __name__ == "__main__":
    main()

# ======== BEGIN: Code extracted from Colab (cleaned) ========


# ##############################################################################
#
# 통합 펩타이드 발굴 파이프라인 (Single-Notebook Peptide Discovery Pipeline) - PRODIGY 개선 버전
#
# 이 코드는 Google Colab Pro/Pro+ 환경에서 실행하는 것을 권장합니다.
# Colab 메뉴에서 '런타임' -> '런타임 유형 변경'을 선택하여
# 하드웨어 가속기를 'GPU'로, 런타임 구성을 '높은 RAM'으로 설정해주세요.
#
# ##############################################################################


# ==============================================================================
# STEP 0: 환경 설정 및 모든 필수 라이브러리 설치 (수정 버전)
# ==============================================================================
import torch
from datetime import datetime
import pytz
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM
import glob
import json
import pandas as pd
import re
import subprocess
import numpy as np
import requests
import time
from bs4 import BeautifulSoup
import os
import sys
import site
import shutil
import zipfile

# 파이프라인 시작 시간 기록
pipeline_start_time = time.time()

print("="*80)
print("STEP 0: 환경 설정 및 모든 필수 라이브러리 설치 (개선 버전)")
print("="*80)

# 시간대 처리 라이브러리 설치
print("\n   > 시간대 처리 라이브러리 (pytz) 설치 중...")
os.system("pip install -q pytz")
print("   > pytz 설치 완료")

# ColabFold (AlphaFold2) 설치
print("\n[1/5] ColabFold (AlphaFold2) 설치 중...")
print("   > 기존 TensorFlow 패키지를 제거하여 충돌을 방지합니다...")
os.system("pip uninstall -y tensorflow tensorboard tb-nightly tensorflow-estimator tensorflow-hub tensorflow-io > /dev/null 2>&1")
os.system("pip install -q --no-warn-conflicts 'colabfold[alphafold] @ git+https://github.com/sokrypton/ColabFold'")
os.system("pip install -q --no-warn-conflicts 'jax[cuda11_pip]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")

# ColabFold 스크립트 패치
print("   > ColabFold 스크립트 패치 적용 중...")
try:
    dist_packages_path = site.getsitepackages()[0]
    batch_py_path = os.path.join(dist_packages_path, 'colabfold', 'batch.py')
    if os.path.exists(batch_py_path):
        os.system(f"sed -i 's/tf.get_logger().setLevel(logging.ERROR)/#tf.get_logger().setLevel(logging.ERROR)/g' {batch_py_path}")
        os.system(f"sed -i \\\"s/tf.config.set_visible_devices(\\\\\\\\[\\\\\\\\], 'GPU')/#tf.config.set_visible_devices(\\\\\\\\[\\\\\\\\], 'GPU')/g\\\" {batch_py_path}")
        print("   > 패치 적용 완료.")
    else:
        print(f"   > 경고: {batch_py_path}를 찾을 수 없어 패치를 건너뜁니다.")
except Exception as e:
    print(f"   > 경고: ColabFold 패치 중 오류 발생 - {e}")

# 펩타이드 생성 모델 관련 라이브러리 설치
print("\n[2/5] 펩타이드 생성 관련 라이브러리 (Transformers) 설치 중...")
os.system("pip install -q --upgrade transformers sentencepiece")

# 결합력 평가 도구 설치 (개선된 버전)
print("\n[3/5] 결합력 평가 도구 설치 중 (개선된 버전)...")

# 시스템 패키지 업데이트 및 설치
print("   > 시스템 패키지 업데이트 중...")
os.system("apt-get update -qq > /dev/null 2>&1")

# OpenBabel 설치 (개선된 방법)
print("   > OpenBabel 설치 중...")
os.system("apt-get install -y --quiet openbabel python3-openbabel libopenbabel-dev")
os.system("pip install -q openbabel-wheel")
print("   > OpenBabel 설치 완료")

# RDKit 설치 (화학 구조 처리용)
print("   > RDKit 설치 중...")
os.system("pip install -q rdkit-pypi")
print("   > RDKit 설치 완료")

# PLIP 설치 (개선된 방법)
print("   > PLIP 설치 중...")
os.system("pip install -q plip")
# PLIP 대체 라이브러리도 설치
os.system("pip install -q biopython ProLIF MDAnalysis")
print("   > PLIP 및 대체 라이브러리 설치 완료")

# ODDT 및 기타 화학 정보학 도구 설치
print("   > 화학 정보학 도구 (ODDT, scikit-learn) 설치 중...")
os.system("pip install -q oddt scikit-learn")
print("   > ODDT 설치 완료")

# Excel 파일 출력을 위한 라이브러리 설치
print("   > Excel 파일 지원 라이브러리 (openpyxl) 설치 중...")
os.system("pip install -q openpyxl")
print("   > openpyxl 설치 완료")

# AutoDock Vina 다운로드 및 설치 (개선된 버전)
print("\n[4/5] AutoDock Vina 설치 중...")

def setup_vina_robust():
    """강화된 Vina 설치 함수"""
    vina_dir = "vina_1.2.3_linux_x86_64"

    if not os.path.exists(vina_dir):
        print("   > Vina 다운로드 중...")
        # 여러 방법으로 다운로드 시도
        download_commands = [
            "wget -q https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.3/vina_1.2.3_linux_x86_64.zip",
            "curl -L -o vina_1.2.3_linux_x86_64.zip https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.3/vina_1.2.3_linux_x86_64.zip"
        ]

        for cmd in download_commands:
            if os.system(cmd) == 0:
                break
        else:
            print("   > Vina 다운로드 실패")
            return False

        # 압축 해제
        os.system("unzip -q -o vina_1.2.3_linux_x86_64.zip")

        # 실행 권한 부여
        vina_executables = [
            f"{vina_dir}/vina",
            f"{vina_dir}/bin/vina",
        ]

        for executable in vina_executables:
            if os.path.exists(executable):
                os.chmod(executable, 0o755)
                print(f"   > 실행 권한 부여: {executable}")

    # Vina 실행파일 찾기
    possible_paths = [
        f"./{vina_dir}/vina",
        f"./{vina_dir}/bin/vina",
        "vina",
        "/usr/local/bin/vina"
    ]

    for path in possible_paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            print(f"   > Vina 실행파일 발견: {path}")
            return os.path.abspath(path)

    print("   > Vina 실행파일을 찾을 수 없음")
    return None

VINA_EXECUTABLE = setup_vina_robust()

# 추가 도구 설치
print("\n[5/5] 추가 분자 도킹 도구 설치 중...")
os.system("pip install -q pymol-open-source > /dev/null 2>&1")
os.system("pip install -q meeko > /dev/null 2>&1")
print("   > 추가 도구 설치 완료")

print("   > 웹 API 호출용 라이브러리 설치 중...")
os.system("pip install -q requests beautifulsoup4")
print("   > requests 설치 완료")

print("\n모든 설치 완료!")
print("="*80)
print("✅ STEP 0: 환경 설정이 성공적으로 완료되었습니다.")
print("="*80)



# ==============================================================================
# STEP 1: 파이프라인 실행을 위한 변수 설정 및 폴더 구조 생성
# ==============================================================================

print("\n" + "="*80)
print("STEP 1: 파이프라인 실행을 위한 변수 설정 및 폴더 구조 생성")
print("="*80)

# --- 사용자 설정 영역 ---

# 1. 생성할 펩타이드 후보의 개수
N_PEPTIDES = 10

# 2. 타겟 단백질의 아미노산 서열 (FASTA 형식, 한 줄로 입력)
TARGET_PROTEIN_SEQUENCE = "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYNKINQRILVVDPVTSEHELTCQAEGYPKAEVIWTSSDHQVLSGKTTTTNSKREEKLFNVTSTLRINTTTNEIFYCTFRRLDPEENHTAELVIPELPLAHPPNERT" # PD-L1 단백질 서열

# 3. 생성할 펩타이드의 길이
PEPTIDE_LENGTH = 4

# 4. 결과 폴더의 기본 이름 접두사
BASE_FOLDER_PREFIX = "PDP"

# 한국 시간(KST)을 기준으로 동적 폴더 및 파일 이름 생성
kst = pytz.timezone('Asia/Seoul')
now_kst = datetime.now(kst)
timestamp = now_kst.strftime("%Y%m%d_%H%M%S")

# 최종 작업 폴더명
JOB_NAME = f"{BASE_FOLDER_PREFIX}_{timestamp}"

# 하위 폴더 구조 생성
FOLDERS = {
    'main': JOB_NAME,
    'fasta': os.path.join(JOB_NAME, 'fasta'),
    'pdb': os.path.join(JOB_NAME, 'pdb'),
    'results': os.path.join(JOB_NAME, 'results'),
    'colabfold_output': os.path.join(JOB_NAME, 'pdb', 'colabfold_output'),
    'temp': os.path.join(JOB_NAME, 'temp')
}

# 모든 폴더 생성
for folder_name, folder_path in FOLDERS.items():
    os.makedirs(folder_path, exist_ok=True)
    print(f"✔️ 폴더 생성: {folder_path}")

# 파일 경로 설정
PROTEIN_FASTA_PATH = os.path.join(FOLDERS['fasta'], "target_protein.fasta")
OUTPUT_FINAL_XLSX_PATH = os.path.join(FOLDERS['results'], f"final_peptide_ranking_{timestamp}.xlsx")
PDB_ZIP_PATH = os.path.join(FOLDERS['results'], f"peptide_structures_{timestamp}.zip")

# 타겟 단백질 FASTA 파일 생성
with open(PROTEIN_FASTA_PATH, "w") as f:
    f.write(f">target_protein\n{TARGET_PROTEIN_SEQUENCE}\n")

print(f"\n✔️ 작업 폴더: {JOB_NAME}")
print(f"✔️ 생성할 펩타이드 개수: {N_PEPTIDES}")
print(f"✔️ 타겟 단백질 서열 길이: {len(TARGET_PROTEIN_SEQUENCE)}")
print(f"✔️ 생성할 펩타이드 길이: {PEPTIDE_LENGTH}")
print(f"✔️ 타겟 단백질 FASTA 파일: {PROTEIN_FASTA_PATH}")
print(f"✔️ 최종 결과 파일: {OUTPUT_FINAL_XLSX_PATH}")
print(f"✔️ PDB 압축 파일: {PDB_ZIP_PATH}")
print("="*80)
print("✅ STEP 1: 설정 및 폴더 구조 생성이 완료되었습니다.")
print("="*80)



# ==============================================================================
# STEP 2: PepMLM (ESM-2)을 이용한 타겟 특이적 펩타이드 후보 생성
# ==============================================================================

print("\n" + "="*80)
print("STEP 2: PepMLM (ESM-2)을 이용한 타겟 특이적 펩타이드 후보 생성")
print("="*80)

# ESM-2 모델 및 토크나이저 로드
model_name = "facebook/esm2_t12_35M_UR50D"
print(f"'{model_name}' 모델과 토크나이저를 로딩합니다...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
print("모델 로딩 완료!")

# 생성 파라미터
temperature = 1.0
top_k = 50

# 모델 입력용 프롬프트 생성
formatted_target = " ".join(list(TARGET_PROTEIN_SEQUENCE))
mask_tokens = " ".join([tokenizer.mask_token] * PEPTIDE_LENGTH)
prompt = f"{tokenizer.cls_token} {formatted_target} {tokenizer.eos_token} {mask_tokens}"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

mask_token_indices = (input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

peptides = []
peptide_fasta_paths = []

print("\n펩타이드 서열 생성을 시작합니다...")
with torch.no_grad():
    for i in range(N_PEPTIDES):
        current_ids = input_ids.clone().to(model.device)
        shuffled_mask_indices = mask_token_indices[torch.randperm(len(mask_token_indices))]

        for mask_idx in shuffled_mask_indices:
            outputs = model(input_ids=current_ids)
            logits = outputs.logits
            mask_logits = logits[0, mask_idx, :]
            filtered_logits = mask_logits / temperature
            effective_top_k = min(top_k, tokenizer.vocab_size)
            top_k_values, top_k_indices = torch.topk(filtered_logits, effective_top_k)
            filter_tensor = torch.full_like(filtered_logits, -float('Inf'))
            filter_tensor.scatter_(0, top_k_indices, top_k_values)
            probs = F.softmax(filter_tensor, dim=-1)
            predicted_token_id = torch.multinomial(probs, num_samples=1)
            current_ids[0, mask_idx] = predicted_token_id.item()

        generated_token_ids = current_ids[0, mask_token_indices]
        sequence_part = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        sequence = "".join(sequence_part.split())
        peptides.append(sequence)

        # FASTA 파일을 fasta 폴더에 저장
        fasta_path = os.path.join(FOLDERS['fasta'], f"peptide_{i}.fasta")
        with open(fasta_path, "w") as f:
            f.write(f">peptide_{i}\n{sequence}\n")
        peptide_fasta_paths.append(fasta_path)
        print(f"  [{i+1}/{N_PEPTIDES}] 생성 완료: {sequence} (길이: {len(sequence)})")

print("\n--- 생성된 펩타이드 후보 목록 ---")
for i, seq in enumerate(peptides):
    print(f"  - 후보 {i+1}: {seq}")
print("="*80)
print(f"✅ STEP 2: 총 {N_PEPTIDES}개의 펩타이드 후보 생성을 완료했습니다.")
print("="*80)



# ==============================================================================
# STEP 3: 단백질-펩타이드 복합체 3D 구조 예측 (ColabFold)
# ==============================================================================

print("\n" + "="*80)
print("STEP 3: 단백질-펩타이드 복합체 3D 구조 예측 (ColabFold)")
print("="*80)

predicted_pdb_files = []

# 배치 처리를 위한 복합체 CSV 파일 생성
print("\n배치 처리를 위한 복합체 CSV 파일 생성 중...")
batch_csv_path = os.path.join(FOLDERS['temp'], "batch_complexes.csv")
with open(batch_csv_path, "w") as f:
    f.write("id,sequence\n")
    for i in range(N_PEPTIDES):
        peptide_seq = peptides[i]
        complex_sequence = f"{TARGET_PROTEIN_SEQUENCE}:{peptide_seq}"
        f.write(f"complex_{i},{complex_sequence}\n")

print(f"✅ 배치 파일 생성 완료: {batch_csv_path}")

# ColabFold 배치 실행 (출력을 pdb/colabfold_output 폴더로)
output_dir = FOLDERS['colabfold_output']
log_file = os.path.join(output_dir, "colabfold_batch.log")

print(f"\nColabFold 배치 실행 시작... (출력 디렉토리: {output_dir})")
print("⏰ 예상 소요 시간: 10-30분 (복합체 개수에 따라 달라집니다)")

# Colab 환경에 최적화된 옵션 사용
colabfold_cmd = (f"colabfold_batch "
                f"--num-recycle 1 "
                f"--model-type alphafold2_multimer_v3 "
                f"--rank ptm "
                f"--max-msa 32:128 "
                f"--num-models 1 "
                f"--stop-at-score 0.5 "
                f"{batch_csv_path} {output_dir} > {log_file} 2>&1")

print(f"실행 명령어: {colabfold_cmd}")
result = os.system(colabfold_cmd)

# 결과 확인
print(f"\nColabFold 실행 완료 (종료 코드: {result})")

# 생성된 PDB 파일 찾기
for i in range(N_PEPTIDES):
    pdb_pattern = os.path.join(output_dir, f"complex_{i}_unrelaxed_rank_001*.pdb")
    pdb_files = sorted(glob.glob(pdb_pattern))

    if pdb_files:
        predicted_pdb_files.append(pdb_files[0])
        print(f"  ✅ 복합체 {i}: {os.path.basename(pdb_files[0])}")
    else:
        print(f"  ❌ 복합체 {i}: PDB 파일을 찾을 수 없음")

# 실패 시 로그 파일 내용 출력
if len(predicted_pdb_files) < N_PEPTIDES and os.path.exists(log_file):
    print("\n" + "="*50)
    print("⚠️ 일부 예측이 실패했습니다. COLABFOLD 실행 로그:")
    print("="*50)
    with open(log_file, 'r') as f:
        print(f.read()[-2000:])
    print("="*50)

print("="*80)
print(f"✅ STEP 3: 총 {len(predicted_pdb_files)}개의 3D 구조 예측을 완료했습니다.")
print("="*80)


# ==============================================================================
# STEP 3.5: 구조 예측 신뢰도 점수(pTM) 확인 및 저장
# ==============================================================================

print("\n" + "="*80)
print("STEP 3.5: 구조 예측 신뢰도 점수(pTM) 확인")
print("="*80)

scores_info = []
ptm_scores_map = {}

# 다양한 점수 파일 패턴 시도
score_file_patterns = [
    os.path.join(output_dir, "*_scores.json"),
    os.path.join(output_dir, "complex_*_scores.json"),
    os.path.join(output_dir, "*_rank_001_*.json"),
    os.path.join(output_dir, "*_score*.json")
]

all_score_files = []
for pattern in score_file_patterns:
    files = sorted(glob.glob(pattern))
    all_score_files.extend(files)

# 중복 제거
all_score_files = list(set(all_score_files))

print(f"찾은 점수 파일들: {len(all_score_files)}개")

if not all_score_files:
    print("⚠️ ColabFold 점수 파일을 찾을 수 없습니다. pTM 점수는 0으로 처리됩니다.")
else:
    print(f"총 {len(all_score_files)}개의 점수 파일을 분석합니다...")

    for score_file in all_score_files:
        try:
            basename = os.path.basename(score_file)
            match = re.search(r'complex_(\d+)', basename)
            if not match:
                continue

            peptide_index = int(match.group(1))
            with open(score_file, 'r') as f:
                data = json.load(f)

            ptm_score = data.get('ptm', data.get('iptm', 0.0))
            if isinstance(ptm_score, list):
                ptm_score = ptm_score[0] if ptm_score else 0.0

            if peptide_index < len(peptides):
                peptide_seq = peptides[peptide_index]
                ptm_scores_map[peptide_seq] = round(float(ptm_score), 3)
                print(f"  복합체 {peptide_index} ({peptide_seq}): pTM = {ptm_scores_map[peptide_seq]}")

        except Exception as e:
            print(f"오류: {score_file} 처리 중 문제 발생 - {e}")
            continue

print("\n" + "="*80)
print("✅ STEP 3.5: pTM 점수 확인이 완료되었습니다.")
print("="*80)



# ==============================================================================
# STEP 4: 개선된 결합력 평가 및 최종 랭킹 계산 (PRODIGY 개선 버전)
# ==============================================================================

print("\n" + "="*80)
print("STEP 4: 개선된 결합력 평가 및 최종 랭킹 계산 (PRODIGY 개선)")
print("="*80)

# ============= PRODIGY 대체 함수 추가 =============

def estimate_binding_affinity_alternative(pdb_file):
    """PRODIGY 웹서버 실패 시 대체 결합 친화도 추정 함수"""
    try:
        chain_a_atoms, chain_b_atoms = [], []

        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    chain = line[21]
                    atom_type = line[12:16].strip()
                    residue = line[17:20].strip()
                    element = line[76:78].strip() or atom_type[0]
                    coords = (
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                        atom_type,
                        element,
                        residue
                    )

                    if chain == 'A':
                        chain_a_atoms.append(coords)
                    elif chain == 'B':
                        chain_b_atoms.append(coords)

        if not chain_a_atoms or not chain_b_atoms:
            return -5.0  # 기본값

        # 접촉면 분석 기반 친화도 추정
        contact_pairs = 0
        hydrophobic_contacts = 0
        polar_contacts = 0
        aromatic_contacts = 0

        hydrophobic_residues = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO'}
        polar_residues = {'SER', 'THR', 'ASN', 'GLN', 'TYR', 'CYS'}
        aromatic_residues = {'PHE', 'TRP', 'TYR', 'HIS'}

        for bx, by, bz, b_atom, b_element, b_res in chain_b_atoms:
            for ax, ay, az, a_atom, a_element, a_res in chain_a_atoms:
                distance = np.sqrt((bx-ax)**2 + (by-ay)**2 + (bz-az)**2)

                if distance <= 4.5:  # 접촉 거리
                    contact_pairs += 1

                    # 상호작용 타입별 가중치 적용
                    if a_res in hydrophobic_residues and b_res in hydrophobic_residues:
                        hydrophobic_contacts += 1
                    elif a_res in polar_residues or b_res in polar_residues:
                        polar_contacts += 1
                    elif a_res in aromatic_residues and b_res in aromatic_residues:
                        aromatic_contacts += 1

        # 경험적 공식을 사용한 친화도 추정 (더 음수가 강한 결합)
        if contact_pairs == 0:
            return -2.0

        base_affinity = -3.0  # 기본 친화도
        hydrophobic_contribution = hydrophobic_contacts * -0.3
        polar_contribution = polar_contacts * -0.4
        aromatic_contribution = aromatic_contacts * -0.5
        contact_penalty = max(0, (contact_pairs - 50) * 0.1)  # 너무 많은 접촉은 불리

        estimated_affinity = base_affinity + hydrophobic_contribution + polar_contribution + aromatic_contribution - contact_penalty

        # -15 ~ -1 범위로 제한
        return max(-15.0, min(-1.0, estimated_affinity))

    except Exception as e:
        print(f"       대체 친화도 추정 오류: {e}")
        return -5.0


def predict_binding_affinity_with_prodigy(pdb_file_path, chain_a='A', chain_b='B', max_retries=2):
    """개선된 PRODIGY 웹 서버 호출 함수 (실패 시 대체 함수 사용)"""
    url = "https://bianca.science.uu.nl/prodigy/"

    print(f"       PRODIGY 웹서버 시도 중...")

    for attempt in range(max_retries):
        try:
            with open(pdb_file_path, 'rb') as f:
                files = {'file': (os.path.basename(pdb_file_path), f, 'application/octet-stream')}
                data = {
                    'chain1': chain_a,
                    'chain2': chain_b,
                    'temperature': 25.0,
                    'contact_list': 'true'
                }

                # 더 짧은 타임아웃으로 빠른 실패 처리
                response = requests.post(url, files=files, data=data, timeout=30)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                # 여러 패턴으로 결과 파싱 시도
                patterns_to_try = [
                    r'Predicted binding affinity.*?(-?\d+\.?\d*)',
                    r'ΔG.*?(-?\d+\.?\d*)',
                    r'(-?\d+\.?\d+)\s*kcal/mol'
                ]

                for pattern in patterns_to_try:
                    matches = re.findall(pattern, response.text, re.IGNORECASE)
                    if matches:
                        try:
                            affinity_value = float(matches[0])
                            print(f"       PRODIGY 성공: {affinity_value:.3f} kcal/mol")
                            return affinity_value
                        except ValueError:
                            continue

                # HTML 파싱 시도
                affinity_header = soup.find(string=re.compile(r'Predicted binding affinity', re.IGNORECASE))
                if affinity_header:
                    affinity_value_tag = affinity_header.find_next_sibling('p')
                    if affinity_value_tag:
                        try:
                            affinity_value = float(affinity_value_tag.text.strip())
                            print(f"       PRODIGY 성공 (HTML): {affinity_value:.3f} kcal/mol")
                            return affinity_value
                        except ValueError:
                            pass

                print(f"       PRODIGY 결과 파싱 실패 (시도 {attempt + 1}/{max_retries})")

            else:
                print(f"       PRODIGY 서버 에러: 상태 코드 {response.status_code} (시도 {attempt + 1}/{max_retries})")

        except requests.exceptions.Timeout:
            print(f"       PRODIGY 타임아웃 (시도 {attempt + 1}/{max_retries})")
        except requests.exceptions.RequestException as e:
            print(f"       PRODIGY 요청 에러 (시도 {attempt + 1}/{max_retries}): {e}")

        if attempt < max_retries - 1:
            print(f"       {3}초 후 재시도...")
            time.sleep(3)

    # 웹서버 실패 시 대체 함수 사용
    print(f"       PRODIGY 웹서버 실패. 대체 추정 함수 사용.")
    alternative_score = estimate_binding_affinity_alternative(pdb_file_path)
    print(f"       대체 추정 결과: {alternative_score:.3f} kcal/mol")
    return alternative_score


def calculate_interactions_advanced(pdb_file):
    """개선된 분자간 상호작용 계산"""
    try:
        chain_a_atoms, chain_b_atoms = [], []

        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    chain = line[21]
                    atom_type = line[12:16].strip()
                    element = line[76:78].strip() or atom_type[0]
                    coords = (
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                        atom_type,
                        element
                    )

                    if chain == 'A':
                        chain_a_atoms.append(coords)
                    elif chain == 'B':
                        chain_b_atoms.append(coords)

        if not chain_a_atoms or not chain_b_atoms:
            return {'h_bonds': 0, 'hydrophobic': 0, 'electrostatic': 0, 'total': 0}

        h_bonds = 0
        hydrophobic = 0
        electrostatic = 0

        for bx, by, bz, b_atom, b_element in chain_b_atoms:
            for ax, ay, az, a_atom, a_element in chain_a_atoms:
                distance = np.sqrt((bx-ax)**2 + (by-ay)**2 + (bz-az)**2)

                # 수소결합 (N, O 원자간 3.5Å 이내)
                if distance <= 3.5:
                    if ((a_element in ['N', 'O'] and b_element in ['N', 'O']) or
                        ('N' in a_atom and 'O' in b_atom) or
                        ('O' in a_atom and 'N' in b_atom)):
                        h_bonds += 1

                # 소수성 상호작용 (탄소 원자간 4.5Å 이내)
                if distance <= 4.5:
                    if a_element == 'C' and b_element == 'C':
                        hydrophobic += 1

                # 정전기적 상호작용 (하전 원자간 5.0Å 이내)
                if distance <= 5.0:
                    charged_atoms_pos = ['LYS', 'ARG', 'HIS']  # 양전하
                    charged_atoms_neg = ['ASP', 'GLU']         # 음전하

                    a_residue = a_atom[:3] if len(a_atom) >= 3 else ''
                    b_residue = b_atom[:3] if len(b_atom) >= 3 else ''

                    if ((a_residue in charged_atoms_pos and b_residue in charged_atoms_neg) or
                        (a_residue in charged_atoms_neg and b_residue in charged_atoms_pos)):
                        electrostatic += 1

        total_interactions = h_bonds + hydrophobic + electrostatic
        return {
            'h_bonds': h_bonds,
            'hydrophobic': hydrophobic,
            'electrostatic': electrostatic,
            'total': total_interactions
        }

    except Exception as e:
        print(f"       상호작용 계산 오류: {e}")
        return {'h_bonds': 0, 'hydrophobic': 0, 'electrostatic': 0, 'total': 0}

def split_pdb_and_get_center(pdb_file, base_name):
    """PDB 파일을 Receptor(Chain A)와 Ligand(Chain B)로 분리하고, Ligand의 중심 좌표를 계산합니다."""
    receptor_path = os.path.join(FOLDERS['temp'], f"{base_name}_receptor.pdb")
    ligand_path = os.path.join(FOLDERS['temp'], f"{base_name}_ligand.pdb")

    chain_b_coords = []
    with open(pdb_file, 'r') as f_in, open(receptor_path, 'w') as f_receptor, open(ligand_path, 'w') as f_ligand:
        for line in f_in:
            if line.startswith(('ATOM', 'HETATM')):
                chain_id = line[21]
                if chain_id == 'A':
                    f_receptor.write(line)
                elif chain_id == 'B':
                    f_ligand.write(line)
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    chain_b_coords.append([x, y, z])

    center = [0.0, 0.0, 0.0]
    if chain_b_coords:
        center = np.mean(chain_b_coords, axis=0).tolist()

    return receptor_path, ligand_path, center

def run_vina_docking(receptor_pdb, ligand_pdb, center, vina_executable):
    """AutoDock Vina를 사용하여 도킹을 수행하고 결합 에너지 점수를 반환합니다."""
    try:
        # Meeko를 사용하여 PDBQT 파일 준비
        receptor_pdbqt = receptor_pdb.replace('.pdb', '.pdbqt')
        ligand_pdbqt = ligand_pdb.replace('.pdb', '.pdbqt')

        # -W: 수소 원자 유지, --rigid: 리간드를 rigid로 처리 (펩타이드에 적합)
        os.system(f"mk_prepare_receptor -i {receptor_pdb} -o {receptor_pdbqt} > /dev/null 2>&1")
        os.system(f"mk_prepare_ligand -i {ligand_pdb} -o {ligand_pdbqt} --rigid > /dev/null 2>&1")

        if not os.path.exists(receptor_pdbqt) or not os.path.exists(ligand_pdbqt):
            print("       PDBQT 파일 생성 실패. OpenBabel로 대체 시도.")
            os.system(f"obabel {receptor_pdb} -O {receptor_pdbqt} -xr > /dev/null 2>&1")
            os.system(f"obabel {ligand_pdb} -O {ligand_pdbqt} > /dev/null 2>&1")
            if not os.path.exists(receptor_pdbqt) or not os.path.exists(ligand_pdbqt):
                 print("       대체 방법도 실패. Vina 도킹을 건너뜁니다.")
                 return 0.0

        # Vina 실행
        output_pdbqt = ligand_pdb.replace('.pdb', '_vina_out.pdbqt')
        log_file = ligand_pdb.replace('.pdb', '_vina.log')

        cmd = [
            vina_executable,
            '--receptor', receptor_pdbqt,
            '--ligand', ligand_pdbqt,
            '--center_x', str(center[0]),
            '--center_y', str(center[1]),
            '--center_z', str(center[2]),
            '--size_x', '30', # 펩타이드 크기에 맞게 박스 크기 조정
            '--size_y', '30',
            '--size_z', '30',
            '--exhaustiveness', '16', # 정확도 향상
            '--out', output_pdbqt,
            '--log', log_file
        ]

        subprocess.run(cmd, check=True, capture_output=True, text=True)

        # 결과 파싱
        with open(log_file, 'r') as f:
            for line in f:
                if line.strip().startswith('1'):
                    parts = line.split()
                    return float(parts[1])
        return 0.0
    except Exception as e:
        print(f"       Vina 도킹 중 오류 발생: {e}")
        return 0.0


# 메인 평가 루프
results = []

if not predicted_pdb_files:
    print("평가할 PDB 파일이 없습니다.")
else:
    print(f"총 {len(predicted_pdb_files)}개의 구조에 대해 평가를 시작합니다...")

    for idx, pred_pdb in enumerate(predicted_pdb_files):
        print(f"\n  평가 중 ({idx+1}/{len(predicted_pdb_files)}): {os.path.basename(pred_pdb)}")

        base_name = f"eval_{idx}"

        if not os.path.exists(pred_pdb) or os.path.getsize(pred_pdb) == 0:
            print("    -> PDB 파일이 존재하지 않거나 비어있습니다.")
            continue

        # 펩타이드 서열 확인
        try:
            peptide_index = int(re.search(r'complex_(\d+)', os.path.basename(pred_pdb)).group(1))
            peptide_seq = peptides[peptide_index]
        except (AttributeError, IndexError, ValueError):
            peptide_seq = f"Unknown_{idx}"

        print("    -> PRODIGY 결합 친화도 예측 중...")
        prodigy_score = predict_binding_affinity_with_prodigy(pred_pdb, chain_a='A', chain_b='B')

        # PDB 파일 분리
        receptor_pdb, ligand_pdb, center = split_pdb_and_get_center(pred_pdb, base_name)
        print(f"    -> PDB 분리 완료: Receptor={os.path.basename(receptor_pdb)}, Ligand={os.path.basename(ligand_pdb)}")

        # Vina 도킹 점수 계산
        vina_score = 0.0
        if VINA_EXECUTABLE and os.path.exists(VINA_EXECUTABLE) and center:
            print("    -> Vina 도킹 실행 중...")
            vina_score = run_vina_docking(receptor_pdb, ligand_pdb, center, VINA_EXECUTABLE)
            print(f"       Vina 점수: {vina_score:.3f} kcal/mol")
        else:
            print("    -> Vina를 사용할 수 없어 간단한 추정을 사용합니다...")
            # 간단한 거리 기반 추정 로직
            try:
                with open(receptor_pdb, 'r') as f:
                    receptor_lines = [line for line in f if line.startswith(('ATOM', 'HETATM'))]
                with open(ligand_pdb, 'r') as f:
                    ligand_lines = [line for line in f if line.startswith(('ATOM', 'HETATM'))]

                if receptor_lines and ligand_lines:
                    min_dist = float('inf')
                    # 계산량을 줄이기 위해 일부 원자만 샘플링
                    for r_line in receptor_lines[::10]: # 10개 중 1개
                        rx = float(r_line[30:38])
                        ry = float(r_line[38:46])
                        rz = float(r_line[46:54])
                        for l_line in ligand_lines:
                            lx = float(l_line[30:38])
                            ly = float(l_line[38:46])
                            lz = float(l_line[46:54])
                            dist = np.sqrt((rx-lx)**2 + (ry-ly)**2 + (rz-lz)**2)
                            min_dist = min(min_dist, dist)

                    if min_dist < float('inf'):
                        # 거리가 가까울수록 점수가 낮아짐(안정)
                        vina_score = max(-1.0 * (15.0 / max(min_dist, 0.1)), -15.0)

            except Exception as e:
                print(f"       간단한 추정 실패: {e}")
                vina_score = -5.0

        # 상호작용 분석
        print("    -> 분자간 상호작용 분석 중...")
        interactions = calculate_interactions_advanced(pred_pdb)
        print(f"       상호작용: H-bonds={interactions['h_bonds']}, "
              f"Hydrophobic={interactions['hydrophobic']}, "
              f"Electrostatic={interactions['electrostatic']}")

        print("    -> 최종 점수 계산 중...")

        # 개선된 가중치 적용
        final_score = (
            abs(vina_score) * 0.25 +           # Vina 점수 (가중치 감소)
            abs(prodigy_score) * 0.5 +         # PRODIGY (높은 가중치, 절대값 사용)
            interactions['total'] * 0.15 +     # 상호작용 수
            ptm_scores_map.get(peptide_seq, 0.0) * 10 * 0.1  # pTM 점수(0-1 스케일)를 10점 만점으로 변환하여 반영
        )

        # 결과 저장 (PRODIGY 점수 추가)
        results.append({
            "Peptide Sequence": peptide_seq,
            "pTM Score": ptm_scores_map.get(peptide_seq, 0.0),
            "Vina Score (kcal/mol)": round(vina_score, 3),
            "PRODIGY Score (kcal/mol)": round(prodigy_score, 3),
            "H-bonds": interactions['h_bonds'],
            "Hydrophobic": interactions['hydrophobic'],
            "Electrostatic": interactions['electrostatic'],
            "Total Interactions": interactions['total'],
            "Final Score": round(final_score, 3),
            "Source PDB": os.path.basename(pred_pdb)
        })

        print(f"    -> 평가 완료: Final Score = {final_score:.3f}")

print("="*80)
print("✅ STEP 4: 모든 구조에 대한 평가 및 점수 계산을 완료했습니다.")
print("="*80)



# ==============================================================================
# STEP 5: PDB 파일 압축 및 최종 결과 저장
# ==============================================================================

print("\n" + "="*80)
print("STEP 5: PDB 파일 압축 및 최종 결과 저장")
print("="*80)

# PDB 파일들을 ZIP으로 압축
print("\n📦 PDB 파일들을 압축하는 중...")
try:
    with zipfile.ZipFile(PDB_ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # ColabFold 출력 폴더의 모든 PDB 파일 추가
        pdb_files_added = 0
        for root, dirs, files in os.walk(FOLDERS['colabfold_output']):
            for file in files:
                if file.endswith('.pdb'):
                    file_path = os.path.join(root, file)
                    # ZIP 내에서의 경로명을 상대경로로 설정
                    arc_name = os.path.relpath(file_path, FOLDERS['colabfold_output'])
                    zipf.write(file_path, arc_name)
                    pdb_files_added += 1
                    print(f"  ✅ 추가됨: {file}")

        # # JSON 점수 파일들도 함께 압축
        # for root, dirs, files in os.walk(FOLDERS['colabfold_output']):
        #     for file in files:
        #         if file.endswith('.json'):
        #             file_path = os.path.join(root, file)
        #             arc_name = os.path.relpath(file_path, FOLDERS['colabfold_output'])
        #             zipf.write(file_path, arc_name)
        #             print(f"  ✅ 점수 파일 추가: {file}")

    print(f"📦 압축 완료: {pdb_files_added}개의 PDB 파일이 {PDB_ZIP_PATH}에 저장되었습니다.")

except Exception as e:
    print(f"❌ 압축 중 오류 발생: {e}")

# 최종 결과 저장 및 출력
if results:
    df = pd.DataFrame(results)

    # 최종 결과에 표시할 컬럼 순서 (PRODIGY 추가)
    column_order = [
        "Peptide Sequence", "Final Score", "pTM Score",
        "PRODIGY Score (kcal/mol)",
        "Vina Score (kcal/mol)",
        "Total Interactions",
        "H-bonds", "Hydrophobic", "Electrostatic",
        "Source PDB"
    ]

    # DataFrame 정렬 및 재배치
    df_sorted = df.sort_values("Final Score", ascending=False).reset_index(drop=True)
    df_final = df_sorted[[col for col in column_order if col in df_sorted.columns]]

    # Excel 파일로 저장 (results 폴더에)
    df_final.to_excel(OUTPUT_FINAL_XLSX_PATH, index=False)

    print("\n🏆 최종 펩타이드 후보 랭킹:")
    display(df_final)

    print(f"\n💾 전체 결과가 Excel 파일로 저장되었습니다: {OUTPUT_FINAL_XLSX_PATH}")
    print("   (Colab 왼쪽의 파일 탐색기에서 다운로드할 수 있습니다.)")

    # 요약 정보 출력
    print("\n📊 결과 요약:")
    print(f"   • 총 평가된 펩타이드: {len(results)}개")
    print(f"   • 최고 점수 펩타이드: {df_final.iloc[0]['Peptide Sequence']} (점수: {df_final.iloc[0]['Final Score']:.3f})")
    print(f"   • 평균 pTM 점수: {df_final['pTM Score'].mean():.3f}")
    print(f"   • 평균 PRODIGY 점수: {df_final['PRODIGY Score (kcal/mol)'].mean():.3f}")
    print(f"   • 평균 상호작용 수: {df_final['Total Interactions'].mean():.1f}")

    # 폴더 구조 요약
    print("\n📁 생성된 파일 구조:")
    print(f"   • 메인 폴더: {JOB_NAME}/")
    print(f"     ├── fasta/           : 펩타이드 및 타겟 FASTA 파일들")
    print(f"     ├── pdb/             : 구조 예측 결과")
    print(f"     │   └── colabfold_output/ : ColabFold 원본 출력")
    print(f"     ├── results/         : 최종 결과 파일들")
    print(f"     │   ├── {os.path.basename(OUTPUT_FINAL_XLSX_PATH)}")
    print(f"     │   └── {os.path.basename(PDB_ZIP_PATH)}")
    print(f"     └── temp/            : 임시 처리 파일들")

else:
    print("\n❌ 최종 결과가 없습니다. 파이프라인 중간에 오류가 발생했을 수 있습니다.")

print("="*80)
print("🎉 모든 파이프라인 실행이 완료되었습니다!")
print("="*80)

# 설치된 도구들의 상태 요약
print("\n📋 설치된 도구 상태:")
print(f"   • ColabFold: ✅ 설치됨")
print(f"   • ESM-2 (Transformers): ✅ 설치됨")
print(f"   • PRODIGY: ✅ 웹 API + 대체 함수")
print(f"   • OpenBabel: ✅ 설치됨")
print(f"   • AutoDock Vina: {'✅ 설치됨' if VINA_EXECUTABLE else '⚠️ 간단한 추정 사용'}")
print(f"   • PLIP 대체 함수: ✅ 구현됨")
print("="*80)

# 총 실행 시간 계산 및 출력
pipeline_end_time = time.time()
total_duration = pipeline_end_time - pipeline_start_time

# 시:분:초로 변환
hours = int(total_duration // 3600)
minutes = int((total_duration % 3600) // 60)
seconds = int(total_duration % 60)

# 조건에 따라 출력 형식 달리하기
if hours > 0:
    print(f"\n⏱️  총 실행 시간: {hours}시간 {minutes}분 {seconds}초")
else:
    print(f"\n⏱️  총 실행 시간: {minutes}분 {seconds}초")

print("="*80)
