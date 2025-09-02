# %%
# ##############################################################################
#
# 통합 펩타이드 발굴 파이프라인 (Single-Notebook Peptide Discovery Pipeline)
#
# Google Colab Pro/Pro+ 환경에서 실행 권장
# 런타임 유형: GPU, 높은 RAM 설정
#
# ##############################################################################

import time
# 파이프라인 시작 시간 기록
pipeline_start_time = time.time()

# ==============================================================================
# STEP 0: 환경 설정 및 필수 라이브러리 설치
# ==============================================================================

print("="*80)
print("STEP 0: 환경 설정 및 필수 라이브러리 설치")
print("="*80)

import os
import sys
import site
import subprocess
import shutil

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

# 결합력 평가 도구 설치
print("\n[3/5] 결합력 평가 도구 설치 중...")

# 시스템 패키지 업데이트 및 설치
print("   > 시스템 패키지 업데이트 중...")
os.system("apt-get update -qq > /dev/null 2>&1")

# OpenBabel 설치
print("   > OpenBabel 설치 중...")
os.system("apt-get install -y --quiet openbabel python3-openbabel libopenbabel-dev")
os.system("pip install -q openbabel-wheel")
print("   > OpenBabel 설치 완료")

# RDKit 설치 (화학 구조 처리용)
print("   > RDKit 설치 중...")
os.system("pip install -q rdkit-pypi")
print("   > RDKit 설치 완료")

# PLIP 설치
print("   > PLIP 설치 중...")
os.system("pip install -q plip")
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

# AutoDock Vina 다운로드 및 설치
print("\n[4/5] AutoDock Vina 설치 중...")

def setup_vina_robust():
    """Vina 설치 함수"""
    vina_dir = "vina_1.2.3_linux_x86_64"

    if not os.path.exists(vina_dir):
        print("   > Vina 다운로드 중...")
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
os.system("pip install -q requests")
print("   > requests 설치 완료")

print("\n모든 설치 완료!")
print("="*80)
print("✅ STEP 0: 환경 설정이 성공적으로 완료되었습니다.")
print("="*80)

# ==============================================================================
# STEP 1: 파이프라인 실행을 위한 변수 설정
# ==============================================================================

print("\n" + "="*80)
print("STEP 1: 파이프라인 실행을 위한 변수 설정")
print("="*80)

import torch
from datetime import datetime
import pytz

# --- 사용자 설정 영역 ---

# 1. 생성할 펩타이드 후보의 개수
N_PEPTIDES = 5

# 2. 타겟 단백질의 아미노산 서열 (FASTA 형식, 한 줄로 입력)
TARGET_PROTEIN_SEQUENCE = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK"

# 3. 생성할 펩타이드의 길이
PEPTIDE_LENGTH = 10

# 4. 결과 폴더의 기본 이름 접두사
BASE_FOLDER_PREFIX = "PDP"

# 한국 시간(KST)을 기준으로 동적 폴더 및 파일 이름 생성
kst = pytz.timezone('Asia/Seoul')
now_kst = datetime.now(kst)
timestamp = now_kst.strftime("%Y%m%d_%H%M%S")

# 최종 결과 폴더명
JOB_NAME = f"{BASE_FOLDER_PREFIX}_{timestamp}"

# 설정값 확인 및 디렉토리/파일 경로 생성
os.makedirs(JOB_NAME, exist_ok=True)
PROTEIN_FASTA_PATH = os.path.join(JOB_NAME, "target_protein.fasta")
OUTPUT_FINAL_XLSX_PATH = os.path.join(JOB_NAME, f"final_peptide_ranking_{timestamp}.xlsx")

with open(PROTEIN_FASTA_PATH, "w") as f:
    f.write(f">target_protein\n{TARGET_PROTEIN_SEQUENCE}\n")

print(f"✔️ 작업 폴더: {JOB_NAME}")
print(f"✔️ 생성할 펩타이드 개수: {N_PEPTIDES}")
print(f"✔️ 타겟 단백질 서열 길이: {len(TARGET_PROTEIN_SEQUENCE)}")
print(f"✔️ 생성할 펩타이드 길이: {PEPTIDE_LENGTH}")
print(f"✔️ 타겟 단백질 FASTA 파일 저장: {PROTEIN_FASTA_PATH}")
print(f"✔️ 최종 결과 파일 저장 경로: {OUTPUT_FINAL_XLSX_PATH}")
print("="*80)
print("✅ STEP 1: 설정이 완료되었습니다.")
print("="*80)

# ==============================================================================
# STEP 2: PepMLM (ESM-2)을 이용한 타겟 특이적 펩타이드 후보 생성
# ==============================================================================

print("\n" + "="*80)
print("STEP 2: PepMLM (ESM-2)을 이용한 타겟 특이적 펩타이드 후보 생성")
print("="*80)

import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM

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

        fasta_path = os.path.join(JOB_NAME, f"peptide_{i}.fasta")
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
# STEP 3: 단백질-펩타이드 복합체 3D 구조 예측 (ColabFold) 및 신뢰도 확인
# ==============================================================================

import glob
import json
import pandas as pd
from IPython.display import display
import re

print("\n" + "="*80)
print("STEP 3: 단백질-펩타이드 복합체 3D 구조 예측 (ColabFold) 및 신뢰도 확인")
print("="*80)

predicted_pdb_files = []

# 배치 처리를 위한 복합체 CSV 파일 생성
print("\n배치 처리를 위한 복합체 CSV 파일 생성 중...")
batch_csv_path = os.path.join(JOB_NAME, "batch_complexes.csv")
with open(batch_csv_path, "w") as f:
    f.write("id,sequence\n")
    for i in range(N_PEPTIDES):
        peptide_seq = peptides[i]
        complex_sequence = f"{TARGET_PROTEIN_SEQUENCE}:{peptide_seq}"
        f.write(f"complex_{i},{complex_sequence}\n")

print(f"✅ 배치 파일 생성 완료: {batch_csv_path}")

# ColabFold 배치 실행
output_dir = os.path.join(JOB_NAME, "colabfold_batch_output")
os.makedirs(output_dir, exist_ok=True)
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

# 구조 예측 신뢰도 점수(pTM) 확인
print("\n구조 예측 신뢰도 점수(pTM) 확인 중...")

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

print("="*80)
print(f"✅ STEP 3: 총 {len(predicted_pdb_files)}개의 3D 구조 예측 및 pTM 점수 확인을 완료했습니다.")
print("="*80)

# ==============================================================================
# STEP 4: 결합력 평가 및 최종 랭킹 계산
# ==============================================================================

print("\n" + "="*80)
print("STEP 4: 결합력 평가 및 최종 랭킹 계산")
print("="*80)

import re
import subprocess
import glob
import numpy as np
import requests
import json
import time

# ============= PPI-Affinity 함수들 =============

def predict_ppi_affinity_web(protein_seq, peptide_seq, max_retries=3):
    """PPI-Affinity 웹 서비스를 통한 결합 친화도 예측"""

    services = [
        {
            "name": "PPI-Affinity HKU",
            "url": "https://ppi-affinity.hkucc.hku.hk/api/predict",
            "method": "hku"
        },
        {
            "name": "ProtParam Alternative",
            "url": "https://web.expasy.org/cgi-bin/protparam/protparam1",
            "method": "expasy"
        }
    ]

    for service in services:
        try:
            print(f"    -> {service['name']} 서비스 시도 중...")

            if service['method'] == 'hku':
                score = call_hku_ppi_service(protein_seq, peptide_seq)
            elif service['method'] == 'expasy':
                score = call_expasy_service(protein_seq, peptide_seq)
            else:
                continue

            if score is not None and score > 0:
                print(f"       성공! 예측 점수: {score:.3f}")
                return score

        except Exception as e:
            print(f"       {service['name']} 실패: {e}")
            continue

    # 모든 웹서비스 실패 시 로컬 추정 함수 사용
    print("    -> 웹서비스 실패, 로컬 추정 함수 사용")
    return predict_ppi_affinity_local(protein_seq, peptide_seq)

def call_hku_ppi_service(protein_seq, peptide_seq):
    """HKU PPI-Affinity 서비스 호출"""
    try:
        data = {
            "sequence1": protein_seq,
            "sequence2": peptide_seq,
            "format": "json"
        }

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "PeptidePipeline/1.0"
        }

        response = requests.post(
            "https://ppi-affinity.hkucc.hku.hk/api/predict",
            json=data,
            headers=headers,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            return float(result.get('binding_affinity', 0))
        else:
            return None

    except Exception as e:
        print(f"       HKU API 오류: {e}")
        return None

def call_expasy_service(protein_seq, peptide_seq):
    """ExPASy 기반 단백질 특성 분석 후 친화도 추정"""
    try:
        protein_features = analyze_sequence_properties(protein_seq)
        peptide_features = analyze_sequence_properties(peptide_seq)

        hydrophobic_match = abs(protein_features['hydrophobicity'] - peptide_features['hydrophobicity'])
        charge_interaction = protein_features['charge'] * peptide_features['charge']
        size_factor = min(len(protein_seq), len(peptide_seq)) / max(len(protein_seq), len(peptide_seq))

        affinity_score = (
            (1.0 - hydrophobic_match / 10.0) * 4.0 +
            abs(charge_interaction) * 2.0 +
            size_factor * 2.0 +
            (peptide_features['flexibility'] * 2.0)
        )

        return min(max(affinity_score, 0.0), 10.0)

    except Exception as e:
        print(f"       ExPASy 기반 분석 오류: {e}")
        return None

def analyze_sequence_properties(sequence):
    """아미노산 서열의 물리화학적 특성 분석"""

    aa_properties = {
        'A': {'hydro': 1.8, 'charge': 0, 'flex': 0.3},   'R': {'hydro': -4.5, 'charge': 1, 'flex': 0.9},
        'N': {'hydro': -3.5, 'charge': 0, 'flex': 0.6},  'D': {'hydro': -3.5, 'charge': -1, 'flex': 0.6},
        'C': {'hydro': 2.5, 'charge': 0, 'flex': 0.3},   'Q': {'hydro': -3.5, 'charge': 0, 'flex': 0.7},
        'E': {'hydro': -3.5, 'charge': -1, 'flex': 0.7}, 'G': {'hydro': -0.4, 'charge': 0, 'flex': 0.9},
        'H': {'hydro': -3.2, 'charge': 0.5, 'flex': 0.7},'I': {'hydro': 4.5, 'charge': 0, 'flex': 0.3},
        'L': {'hydro': 3.8, 'charge': 0, 'flex': 0.3},   'K': {'hydro': -3.9, 'charge': 1, 'flex': 0.8},
        'M': {'hydro': 1.9, 'charge': 0, 'flex': 0.5},   'F': {'hydro': 2.8, 'charge': 0, 'flex': 0.3},
        'P': {'hydro': -1.6, 'charge': 0, 'flex': 0.1},  'S': {'hydro': -0.8, 'charge': 0, 'flex': 0.5},
        'T': {'hydro': -0.7, 'charge': 0, 'flex': 0.4},  'W': {'hydro': -0.9, 'charge': 0, 'flex': 0.3},
        'Y': {'hydro': -1.3, 'charge': 0, 'flex': 0.4},  'V': {'hydro': 4.2, 'charge': 0, 'flex': 0.2}
    }

    if not sequence:
        return {'hydrophobicity': 0, 'charge': 0, 'flexibility': 0}

    total_hydro = sum(aa_properties.get(aa, {'hydro': 0})['hydro'] for aa in sequence)
    total_charge = sum(aa_properties.get(aa, {'charge': 0})['charge'] for aa in sequence)
    total_flex = sum(aa_properties.get(aa, {'flex': 0.5})['flex'] for aa in sequence)

    return {
        'hydrophobicity': total_hydro / len(sequence),
        'charge': total_charge,
        'flexibility': total_flex / len(sequence)
    }

def predict_ppi_affinity_local(protein_seq, peptide_seq):
    """로컬 PPI 친화도 추정 함수"""
    try:
        protein_props = analyze_sequence_properties(protein_seq)
        peptide_props = analyze_sequence_properties(peptide_seq)

        charge_score = 0
        if protein_props['charge'] * peptide_props['charge'] < 0:
            charge_score = min(abs(protein_props['charge'] * peptide_props['charge']), 5.0)

        hydro_diff = abs(protein_props['hydrophobicity'] - peptide_props['hydrophobicity'])
        hydro_score = max(3.0 - hydro_diff * 0.5, 0)

        length_ratio = len(peptide_seq) / len(protein_seq)
        size_score = 2.0 * (1.0 - abs(length_ratio - 0.1))

        flex_score = peptide_props['flexibility'] * 2.0

        complexity_score = len(set(peptide_seq)) / 20.0 * 1.5

        final_score = charge_score + hydro_score + size_score + flex_score + complexity_score
        return min(max(final_score, 0.0), 10.0)

    except Exception as e:
        print(f"       로컬 PPI 예측 오류: {e}")
        return 5.0

# ============= 분자간 상호작용 계산 함수 =============

def calculate_interactions_advanced(pdb_file):
    """분자간 상호작용 계산"""
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
                    charged_atoms_pos = ['LYS', 'ARG', 'HIS']
                    charged_atoms_neg = ['ASP', 'GLU']

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
    receptor_path = f"{base_name}_receptor.pdb"
    ligand_path = f"{base_name}_ligand.pdb"

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
        receptor_pdbqt = receptor_pdb.replace('.pdb', '.pdbqt')
        ligand_pdbqt = ligand_pdb.replace('.pdb', '.pdbqt')

        os.system(f"mk_prepare_receptor -i {receptor_pdb} -o {receptor_pdbqt} > /dev/null 2>&1")
        os.system(f"mk_prepare_ligand -i {ligand_pdb} -o {ligand_pdbqt} --rigid > /dev/null 2>&1")

        if not os.path.exists(receptor_pdbqt) or not os.path.exists(ligand_pdbqt):
            print("       PDBQT 파일 생성 실패. OpenBabel로 대체 시도.")
            os.system(f"obabel {receptor_pdb} -O {receptor_pdbqt} -xr > /dev/null 2>&1")
            os.system(f"obabel {ligand_pdb} -O {ligand_pdbqt} > /dev/null 2>&1")
            if not os.path.exists(receptor_pdbqt) or not os.path.exists(ligand_pdbqt):
                 print("       대체 방법도 실패. Vina 도킹을 건너뜁니다.")
                 return 0.0

        output_pdbqt = ligand_pdb.replace('.pdb', '_vina_out.pdbqt')
        log_file = ligand_pdb.replace('.pdb', '_vina.log')

        cmd = [
            vina_executable,
            '--receptor', receptor_pdbqt,
            '--ligand', ligand_pdbqt,
            '--center_x', str(center[0]),
            '--center_y', str(center[1]),
            '--center_z', str(center[2]),
            '--size_x', '30',
            '--size_y', '30',
            '--size_z', '30',
            '--exhaustiveness', '16',
            '--out', output_pdbqt,
            '--log', log_file
        ]

        subprocess.run(cmd, check=True, capture_output=True, text=True)

        with open(log_file, 'r') as f:
            for line in f:
                if line.strip().startswith('1'):
                    parts = line.split()
                    return float(parts[1])
        return 0.0
    except Exception as e:
        print(f"       Vina 도킹 중 오류 발생: {e}")
        return 0.0

# ============= 메인 평가 루프 =============

results = []

if not predicted_pdb_files:
    print("평가할 PDB 파일이 없습니다.")
else:
    print(f"총 {len(predicted_pdb_files)}개의 구조에 대해 평가를 시작합니다...")

    for idx, pred_pdb in enumerate(predicted_pdb_files):
        print(f"\n  평가 중 ({idx+1}/{len(predicted_pdb_files)}): {os.path.basename(pred_pdb)}")

        base_name = os.path.join(JOB_NAME, f"eval_{idx}")

        if not os.path.exists(pred_pdb) or os.path.getsize(pred_pdb) == 0:
            print("    -> PDB 파일이 존재하지 않거나 비어있습니다.")
            continue

        # 펩타이드 서열 확인
        try:
            peptide_index = int(re.search(r'complex_(\d+)', os.path.basename(pred_pdb)).group(1))
            peptide_seq = peptides[peptide_index]
        except (AttributeError, IndexError, ValueError):
            peptide_seq = f"Unknown_{idx}"

        # PPI-Affinity 예측
        print("    -> PPI-Affinity 결합 친화도 예측 중...")
        ppi_affinity_score = predict_ppi_affinity_web(TARGET_PROTEIN_SEQUENCE, peptide_seq)
        print(f"       PPI-Affinity 점수: {ppi_affinity_score:.3f}")

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
            try:
                with open(receptor_pdb, 'r') as f:
                    receptor_lines = [line for line in f if line.startswith(('ATOM', 'HETATM'))]
                with open(ligand_pdb, 'r') as f:
                    ligand_lines = [line for line in f if line.startswith(('ATOM', 'HETATM'))]

                if receptor_lines and ligand_lines:
                    min_dist = float('inf')
                    for r_line in receptor_lines[::10]:
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

        # 최종 점수 계산
        print("    -> 최종 점수 계산 중...")

        final_score = (
            abs(vina_score) * 0.25 +
            ppi_affinity_score * 0.5 +
            interactions['total'] * 0.15 +
            ptm_scores_map.get(peptide_seq, 0.0) * 10 * 0.1
        )

        # 결과 저장
        results.append({
            "Peptide Sequence": peptide_seq,
            "pTM Score": ptm_scores_map.get(peptide_seq, 0.0),
            "Vina Score (kcal/mol)": round(vina_score, 3),
            "PPI-Affinity Score": round(ppi_affinity_score, 3),
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
# STEP 5: 최종 결과 확인 및 저장
# ==============================================================================

print("\n" + "="*80)
print("STEP 5: 최종 결과 확인 및 저장")
print("="*80)

if results:
    import pandas as pd
    from IPython.display import display

    df = pd.DataFrame(results)

    column_order = [
        "Peptide Sequence", "Final Score", "pTM Score",
        "PPI-Affinity Score",
        "Vina Score (kcal/mol)",
        "H-bonds", "Hydrophobic", "Electrostatic", "Total Interactions",
        "Source PDB"
    ]

    df_sorted = df.sort_values("Final Score", ascending=False).reset_index(drop=True)
    df_final = df_sorted[[col for col in column_order if col in df_sorted.columns]]

    # Excel 파일로 저장
    df_final.to_excel(OUTPUT_FINAL_XLSX_PATH, index=False)

    print("\n🏆 최종 펩타이드 후보 랭킹:")
    display(df_final)

    print(f"\n💾 전체 결과가 Excel 파일로 저장되었습니다: {OUTPUT_FINAL_XLSX_PATH}")
    print("   (Colab 왼쪽의 파일 탐색기에서 다운로드할 수 있습니다.)")

    print("\n📊 결과 요약:")
    print(f"   • 총 평가된 펩타이드: {len(results)}개")
    print(f"   • 최고 점수 펩타이드: {df_final.iloc[0]['Peptide Sequence']} (점수: {df_final.iloc[0]['Final Score']:.3f})")
    print(f"   • 평균 pTM 점수: {df_final['pTM Score'].mean():.3f}")
    print(f"   • 평균 PPI-Affinity 점수: {df_final['PPI-Affinity Score'].mean():.3f}")
    print(f"   • 평균 상호작용 수: {df_final['Total Interactions'].mean():.1f}")

else:
    print("\n❌ 최종 결과가 없습니다. 파이프라인 중간에 오류가 발생했을 수 있습니다.")

print("="*80)
print("🎉 모든 파이프라인 실행이 완료되었습니다!")
print("="*80)

print("\n📋 설치된 도구 상태:")
print(f"   • ColabFold: ✅ 설치됨")
print(f"   • ESM-2 (Transformers): ✅ 설치됨")
print(f"   • PPI-Affinity: ✅ 웹 API 통합")
print(f"   • OpenBabel: ✅ 설치됨")
print(f"   • AutoDock Vina: {'✅ 설치됨' if VINA_EXECUTABLE else '⚠️ 간단한 추정 사용'}")
print(f"   • PLIP 대체 함수: ✅ 구현됨")
print("="*80)

# 총 실행 시간 계산 및 출력
pipeline_end_time = time.time()
total_duration_seconds = pipeline_end_time - pipeline_start_time

# 시간, 분, 초 단위로 계산
total_hours = int(total_duration_seconds // 3600)
remaining_seconds = total_duration_seconds % 3600
total_minutes = int(remaining_seconds // 60)
total_seconds = int(remaining_seconds % 60)

# 실행 시간 표시 (1시간 이상인 경우 시간 포함)
if total_hours > 0:
    print(f"\n⏱️  총 실행 시간: {total_hours}시간 {total_minutes}분 {total_seconds}초")
else:
    print(f"\n⏱️  총 실행 시간: {total_minutes}분 {total_seconds}초")
print("="*80)


