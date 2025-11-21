#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pepbind_pipeline.py - WSL/오프라인 환경용 통합 파이프라인

구성:
1) PepMLM(ESM-2)로 타깃 기반 펩타이드 후보 생성 (GPU 사용)
2) ColabFold 멀티머로 타깃-펩타이드 복합체 구조 예측 (진행 상황 출력)
3) AutoDock Vina 도킹 (CPU) - 기본 골격 구현, 리셉터/리간드 분리 및 박스 자동 설정
4) PLIP 상호작용 분석 (CPU)
5) PRODIGY 결합 친화도 평가 (CPU)

실행:
    cd ~/work/pipeline
    conda activate pepbind
    python pepbind_pipeline.py
"""
import re
import json
import zipfile
from openpyxl import Workbook

import os
import sys
import time
import csv
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM
from Bio.PDB import PDBParser, PDBIO, Select

# =====================================================================
# === 사용자 설정 영역: 여기만 수정해서 사용하면 됩니다 ===================
# =====================================================================

# 1) 타깃 단백질 서열 (FASTA의 sequence 부분만)
#    → 실제 타깃 서열을 아래 문자열에 그대로 붙여넣어 사용하세요.
TARGET_SEQUENCE = (
    "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYNKINQRILVVDPVTSEHELTCQAEGYPKAEVIWTSSDHQVLSGKTTTTNSKREEKLFNVTSTLRINTTTNEIFYCTFRRLDPEENHTAELVIPELPLAHPPNERT"
    # TODO: 여기에 실제 타겟 단백질 서열을 붙여넣으세요.
)

# 2) 생성할 펩타이드 설정
NUM_PEPTIDES   = 10   # 생성할 펩타이드 후보 개수
PEPTIDE_LENGTH = 4    # 각 펩타이드 길이 (아미노산 개수)

# 3) ColabFold / 평가 단계 사용 여부
RUN_COLABFOLD  = True   # ColabFold 구조 예측 실행 여부
RUN_VINA       = True   # AutoDock Vina 도킹 실행 여부
RUN_PLIP       = True   # PLIP 상호작용 분석 실행 여부
RUN_PRODIGY    = True   # PRODIGY 결합 친화도 평가 실행 여부

# 4) 쓰레드 수 (ColabFold 등에 사용)
NUM_THREADS = 8

# 5) 작업 기본 디렉토리 (워크스페이스가 이 아래에 생성됨)
BASE_DIR = Path(os.environ.get("PEPBIND_BASE_DIR", "~/work/pipeline")).expanduser()

# 6) 외부 도구 경로 (환경에 맞게 수정 가능)
#    - COLABFOLD_CMD: colabfold_batch 가 PATH에 있으면 그대로 "colabfold_batch"
#    - VINA_CMD: vina 설치되어 있으면 "vina"
#    - PLIP_CMD: 보통 "python -m plip.cmd.plip" 형태로 설정
#    - PRODIGY_SCRIPT: 로컬 PRODIGY CLI 스크립트 경로
COLABFOLD_CMD   = os.environ.get("COLABFOLD_CMD", "colabfold_batch")
VINA_CMD        = os.environ.get("VINA_CMD", "vina")
PLIP_CMD        = os.environ.get("PLIP_CMD", "")            # 예: "python -m plip.cmd.plip"
PRODIGY_SCRIPT  = os.environ.get("PRODIGY_SCRIPT", "")      # 예: "/home/aisys/tools/prodigy_cli.py"

# =====================================================================
# === 공통 설정 / 유틸 =================================================
# =====================================================================

BASE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] PyTorch device: {DEVICE}")

def run(cmd, cwd=None):
    """서브프로세스 실행 래퍼 (오류 시 예외 발생)."""
    print(f"[RUN] {cmd}")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        shell=isinstance(cmd, str),
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (code={result.returncode}): {cmd}")
    return result

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def init_workspace():
    """PDP_YYYYMMDD_HHMMSS 형태 워크스페이스 및 하위 폴더 생성."""
    ws_name = f"PDP_{timestamp()}"
    ws_root = BASE_DIR / ws_name

    folders = {
        "root": ws_root,
        "fasta": ws_root / "fasta",
        "pdb": ws_root / "pdb",
        "colabfold_out": ws_root / "pdb" / "colabfold_output",
        "results": ws_root / "results",
        "vina": ws_root / "results" / "vina",
        "plip": ws_root / "results" / "plip",
        "prodigy": ws_root / "results" / "prodigy",
        "temp": ws_root / "temp",
    }
    for d in folders.values():
        d.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("STEP 1: 워크스페이스 / 폴더 구조 생성")
    print("=" * 80)
    for k, v in folders.items():
        print(f"✔️ {k:12s}: {v}")
    print("=" * 80)

    return folders

# =====================================================================
# === STEP 2: PepMLM (ESM-2) 기반 펩타이드 생성 =======================
# =====================================================================

def load_esm_mlm(model_name: str = "facebook/esm2_t12_35M_UR50D"):
    print("\n" + "=" * 80)
    print("STEP 2: PepMLM (ESM-2) 모델 로딩")
    print("=" * 80)
    print(f"모델 로딩: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(DEVICE)
    model.eval()
    print("✅ PepMLM 모델 로딩 완료")
    return tokenizer, model

def generate_peptides_with_mlm(
    tokenizer,
    model,
    target_sequence: str,
    num_peptides: int = NUM_PEPTIDES,
    peptide_len: int = PEPTIDE_LENGTH,
    top_k: int = 10,
    temperature: float = 1.0,
):
    """
    PepMLM(ESM-2) 기반 펩타이드 생성 (샘플링 버전)

    - "[PEP] [MASK] [MASK] ..." 형태로 입력
    - 각 MASK 위치에서 top-k 확률 분포에서 랜덤 샘플링
    - special token (PAD, CLS, SEP, MASK, UNK)는 제외
    - 마지막 peptide_len 글자를 펩타이드로 사용
    """
    print("\n펩타이드 서열 생성을 시작합니다...")

    mask_token = tokenizer.mask_token
    if mask_token is None:
        raise ValueError("토크나이저에 [MASK] 토큰이 없습니다.")

    # 제외할 토큰 아이디들
    bad_ids = set()
    for tid in [
        tokenizer.pad_token_id,
        tokenizer.cls_token_id if hasattr(tokenizer, "cls_token_id") else None,
        tokenizer.sep_token_id if hasattr(tokenizer, "sep_token_id") else None,
        tokenizer.mask_token_id,
        tokenizer.unk_token_id if hasattr(tokenizer, "unk_token_id") else None,
    ]:
        if tid is not None:
            bad_ids.add(tid)

    # 단순 프롬프트 (필요하면 target_sequence 일부를 섞어 넣어도 됨)
    prompt = "[PEP] " + " ".join([mask_token] * peptide_len)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

    peptides = []
    seen = set()

    with torch.no_grad():
        attempt = 0
        while len(peptides) < num_peptides and attempt < num_peptides * 5:
            attempt += 1
            ids = input_ids.clone()

            for pos in range(ids.size(1)):
                if ids[0, pos].item() == tokenizer.mask_token_id:
                    outputs = model(ids)
                    logits = outputs.logits[0, pos] / temperature
                    probs = F.softmax(logits, dim=-1)

                    # special token 확률 0으로
                    for bid in bad_ids:
                        probs[bid] = 0.0

                    # 정규화
                    probs = probs / probs.sum()

                    # top-k 샘플링
                    k = min(top_k, probs.size(0))
                    top_vals, top_idx = torch.topk(probs, k=k)
                    top_vals = top_vals / top_vals.sum()  # 다시 정규화
                    sampled_local = torch.multinomial(top_vals, num_samples=1)
                    sampled_id = top_idx[sampled_local]
                    ids[0, pos] = sampled_id

            seq = tokenizer.decode(ids[0], skip_special_tokens=True).replace(" ", "")
            pep = seq[-peptide_len:]

            # 너무 이상하게 짧/긴 경우 방어
            if len(pep) != peptide_len:
                continue

            if pep in seen:
                continue
            seen.add(pep)
            peptides.append(pep)
            print(f"  [{len(peptides)}/{num_peptides}] 생성 완료: {pep} (길이: {len(pep)})")

    print("\n--- 생성된 펩타이드 후보 목록 ---")
    for i, p in enumerate(peptides, 1):
        print(f"  - 후보 {i}: {p}")
    print("=" * 80)
    print(f"✅ STEP 2: 총 {len(peptides)}개 펩타이드 후보 생성 완료")
    print("=" * 80)
    return peptides


def write_target_fasta(fasta_dir: Path, target_sequence: str) -> Path:
    fasta_path = fasta_dir / "target_protein.fasta"
    with open(fasta_path, "w") as f:
        f.write(">target_protein\n")
        f.write(target_sequence.strip() + "\n")
    return fasta_path

def write_peptide_fasta(fasta_dir: Path, peptides) -> Path:
    pep_fa = fasta_dir / "peptides.fasta"
    with open(pep_fa, "w") as f:
        for i, pep in enumerate(peptides):
            f.write(f">pep_{i}\n{pep}\n")
    return pep_fa

# =====================================================================
# === STEP 3: ColabFold 배치 (멀티머) + 진행 상황 표시 =================
# =====================================================================

def prepare_colabfold_batch_csv(temp_dir: Path, target_sequence: str, peptides) -> Path:
    """
    ColabFold 1.5.5용 multimer CSV 입력 생성.

    - CSV 컬럼: id, sequence
    - sequence 형식: "타깃서열:펩타이드서열"
      예) AAAAAA...AAAA:PPPP

    ColabFold는 sequence 컬럼에서 ':'로 체인을 구분해
    alphafold2_multimer_v3 모델에 넣어서 복합체 구조를 예측한다.
    """
    csv_path = temp_dir / "batch_complexes.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "sequence"])
        for i, pep in enumerate(peptides):
            complex_id = f"complex_{i}"
            complex_seq = f"{target_sequence}:{pep}"
            writer.writerow([complex_id, complex_seq])
    print(f"✅ ColabFold 배치 CSV 생성 (id,sequence 형식): {csv_path}")
    return csv_path


def run_colabfold_batch_with_progress(csv_path: Path, out_dir: Path, total_complexes: int):
    """
    colabfold_batch 실행 + 진행 상황 출력:
    - rank_001*.pdb 개수를 주기적으로 세어
    - "완료된 구조 개수 / 전체 복합체 개수" 형태로 출력

    현재 설치된 colabfold_batch 버전은 --num-threads 옵션을 지원하지 않으므로
    해당 옵션은 사용하지 않는다.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "colabfold_batch.log"

    cmd = [
        COLABFOLD_CMD,
        "--num-recycle", "1",
        "--model-type", "alphafold2_multimer_v3",
        "--rank", "ptm",
        "--max-msa", "32:128",
        "--num-models", "1",
        "--stop-at-score", "0.5",
        str(csv_path),
        str(out_dir),
    ]

    print("\n" + "=" * 80)
    print("STEP 3: ColabFold 배치 실행")
    print("=" * 80)
    print("[INFO] 실행 명령어:")
    print(" ", " ".join(cmd))
    print(f"[INFO] 로그 파일: {log_file}")

    with open(log_file, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)

    last_done = -1
    while True:
        ret = proc.poll()

        rank1_files = list(out_dir.glob("*rank_001*.*pdb"))
        done = len(rank1_files)
        if done != last_done:
            print(f"\r[ColabFold 진행 상황] {done}/{total_complexes} 구조 완료", end="", flush=True)
            last_done = done

        if ret is not None:
            break
        time.sleep(30)

    print()
    if proc.returncode != 0:
        print(f"[ERROR] ColabFold 실행 실패 (returncode={proc.returncode}). 마지막 40줄 로그:")
        try:
            with open(log_file) as f:
                lines = f.readlines()
            for line in lines[-40:]:
                print(line.rstrip())
        except Exception as e:
            print(f"[WARN] 로그 파일을 읽는 중 오류 발생: {e}")
        raise RuntimeError(f"ColabFold 실행 실패, 로그 확인: {log_file}")

    print("[INFO] ColabFold 실행 완료")
    rank1_files = sorted(out_dir.glob("*rank_001*.*pdb"))
    print(f"[INFO] rank_001 PDB 개수: {len(rank1_files)}")
    return rank1_files

def zip_rank1_pdbs(rank1_pdbs, results_dir: Path):
    """
    ColabFold에서 생성된 rank_001 PDB들을 하나의 zip 파일로 압축.
    """
    if not rank1_pdbs:
        print("[INFO] zip으로 묶을 rank_001 PDB가 없습니다.")
        return None

    zip_path = results_dir / f"peptide_structures_{timestamp()}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for pdb_path in rank1_pdbs:
            # zip 안에는 파일명만 들어가도록
            zf.write(pdb_path, arcname=pdb_path.name)

    print(f"✅ rank_001 PDB 압축 저장: {zip_path}")
    return zip_path

# =====================================================================
# === STEP 4: AutoDock Vina 도킹 (기본 골격) ==========================
# =====================================================================

class ChainSelect(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id
    def accept_chain(self, chain):
        return chain.get_id() == self.chain_id

def split_complex_to_receptor_ligand(complex_pdb: Path, out_dir: Path, receptor_chain: str = "A", ligand_chain: str = "B"):
    """
    간단 가정:
    - ColabFold 멀티머 출력에서 체인 A: 타깃 단백질
    - 체인 B: 펩타이드
    필요하면 receptor_chain, ligand_chain을 바꿔서 호출 가능.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", str(complex_pdb))
    model = next(structure.get_models())

    io = PDBIO()
    rec_pdb = out_dir / f"{complex_pdb.stem}_receptor_{receptor_chain}.pdb"
    lig_pdb = out_dir / f"{complex_pdb.stem}_ligand_{ligand_chain}.pdb"

    io.set_structure(model)
    io.save(str(rec_pdb), ChainSelect(receptor_chain))
    io.set_structure(model)
    io.save(str(lig_pdb), ChainSelect(ligand_chain))

    return rec_pdb, lig_pdb

def compute_box_from_ligand(lig_pdb: Path, padding: float = 10.0):
    """
    리간드 좌표를 기반으로 박스 중심/크기를 자동 설정.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("ligand", str(lig_pdb))
    model = next(structure.get_models())
    coords = []
    for atom in model.get_atoms():
        coord = atom.get_coord()
        coords.append(coord)
    if not coords:
        raise ValueError(f"리간드 PDB에서 원자 좌표를 찾지 못했습니다: {lig_pdb}")

    import numpy as np
    coords = np.array(coords)
    center = coords.mean(axis=0)
    minc = coords.min(axis=0)
    maxc = coords.max(axis=0)
    size = (maxc - minc) + padding

    box = {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "center_z": float(center[2]),
        "size_x": float(size[0]),
        "size_y": float(size[1]),
        "size_z": float(size[2]),
    }
    return box

def prepare_pdbqt(rec_pdb: Path, lig_pdb: Path, out_dir: Path):
    """
    PDB → PDBQT 변환.
    우선순위:
    1) AutoDockTools 스크립트(prepare_receptor4.py, prepare_ligand4.py)가 있으면 그것 사용
    2) 없으면 obabel이 있으면 obabel 사용
    3) 둘 다 없으면 예외 발생
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_pdbqt = out_dir / f"{rec_pdb.stem}.pdbqt"
    lig_pdbqt = out_dir / f"{lig_pdb.stem}.pdbqt"

    prep_rec = shutil.which("prepare_receptor4.py")
    prep_lig = shutil.which("prepare_ligand4.py")
    obabel   = shutil.which("obabel")

    if prep_rec and prep_lig:
        run(f"{prep_rec} -r {rec_pdb} -o {rec_pdbqt} -A hydrogens")
        run(f"{prep_lig} -l {lig_pdb} -o {lig_pdbqt} -A hydrogens")
    elif obabel:
        # 간단 obabel 변환 (세부 옵션은 환경에 맞게 조정 가능)
        run(f"{obabel} -ipdb {rec_pdb} -xr -opdbqt -O {rec_pdbqt}")
        run(f"{obabel} -ipdb {lig_pdb}      -opdbqt -O {lig_pdbqt}")
    else:
        raise RuntimeError("PDBQT 변환 도구(prepare_* 또는 obabel)를 찾을 수 없습니다.")

    return rec_pdbqt, lig_pdbqt

def parse_vina_score_from_stdout(stdout: str):
    """
    AutoDock Vina stdout에서 best score(대개 kcal/mol, 음수)를 추정.

    우선순위:
    1) 'REMARK VINA RESULT:' 형식이 있으면 그 줄에서 float 추출
    2) 없으면 전체에서 -30~0 범위의 실수들을 모아 가장 작은 값(가장 낮은 에너지)을 선택
    """
    best = None

    # 1) REMARK VINA RESULT 스타일
    for line in stdout.splitlines():
        if "REMARK VINA RESULT" in line:
            for token in line.split():
                try:
                    val = float(token)
                    best = val
                    return best
                except ValueError:
                    continue

    # 2) fallback: 전체에서 -30~0 사이의 실수 후보를 모음
    candidates = []
    for line in stdout.splitlines():
        for m in re.finditer(r"[-+]?\d+\.\d+", line):
            val = float(m.group(0))
            if -30.0 <= val <= 0.0:
                candidates.append(val)

    if candidates:
        # 가장 에너지가 낮은(가장 음수) 값을 best로 사용
        best = min(candidates)

    return best

def run_vina_on_rank1(rank1_pdbs, vina_dir: Path):
    """
    AutoDock Vina 도킹.
    - GPU 지원은 기본적으로 없고 CPU에서 동작.
    - 여기서는 단순히 체인 A(리셉터)/B(펩타이드)로 가정하고 박스 자동 설정 후 도킹 실행.
    - 설치된 Vina 버전(23d1252-mod)은 --log 옵션이 없으므로 stdout을 캡처해서 점수를 파싱한다.
    """
    print("\n" + "=" * 80)
    print("STEP 4: AutoDock Vina 도킹")
    print("=" * 80)

    if not rank1_pdbs:
        print("[WARN] Vina 실행할 rank_001 PDB가 없습니다.")
        return
    if not shutil.which(VINA_CMD):
        print(f"[WARN] VINA_CMD='{VINA_CMD}' 실행 파일을 찾을 수 없습니다. (PATH 확인 필요)")
        return

    vina_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = [["complex", "vina_score", "receptor_pdbqt", "ligand_pdbqt", "log_file"]]

    for complex_pdb in rank1_pdbs:
        base = complex_pdb.stem
        complex_out_dir = vina_dir / base
        complex_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[INFO] Vina 준비: {complex_pdb.name}")
        # 1) 체인 분리 (A: 타깃, B: 펩타이드 가정)
        rec_pdb, lig_pdb = split_complex_to_receptor_ligand(
            complex_pdb,
            complex_out_dir,
            receptor_chain="A",
            ligand_chain="B",
        )

        # 2) PDBQT 준비
        rec_pdbqt, lig_pdbqt = prepare_pdbqt(rec_pdb, lig_pdb, complex_out_dir)

        # 3) 박스 설정 (리간드 기반)
        box = compute_box_from_ligand(lig_pdb)

        # 4) Vina 실행 (stdout 캡처)
        out_pdbqt = complex_out_dir / f"{base}_vina_out.pdbqt"
        log_file  = complex_out_dir / f"{base}_vina_stdout.txt"

        vina_cmd = (
            f"{VINA_CMD} "
            f"--receptor {rec_pdbqt} "
            f"--ligand {lig_pdbqt} "
            f"--center_x {box['center_x']:.3f} --center_y {box['center_y']:.3f} --center_z {box['center_z']:.3f} "
            f"--size_x {box['size_x']:.3f} --size_y {box['size_y']:.3f} --size_z {box['size_z']:.3f} "
            f"--out {out_pdbqt}"
        )

        print(f"[RUN] {vina_cmd}")
        result = subprocess.run(
            vina_cmd,
            shell=True,
            capture_output=True,
            text=True,
        )

        # stdout/stderr를 파일로 저장 (나중에 확인용)
        with open(log_file, "w") as lf:
            lf.write("=== STDOUT ===\n")
            lf.write(result.stdout or "")
            lf.write("\n\n=== STDERR ===\n")
            lf.write(result.stderr or "")

        if result.returncode != 0:
            print(f"[ERROR] Vina 실행 실패 (code={result.returncode}). 로그 파일: {log_file}")
            print(result.stdout)
            print(result.stderr)
            best_score = None
        else:
            best_score = parse_vina_score_from_stdout(result.stdout)
            print(f"[INFO] {complex_pdb.name} Vina best score: {best_score}")

        summary_rows.append([base, best_score, rec_pdbqt.name, lig_pdbqt.name, log_file.name])

    # 전체 요약 저장
    summary_csv = vina_dir / "vina_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(summary_rows)
    print(f"\n✅ Vina 요약 저장: {summary_csv}")
    print("=" * 80)

def load_vina_scores(vina_dir: Path):
    """
    vina_summary.csv 에서 complex별 Vina score 로딩.
    """
    scores = {}
    summary_csv = vina_dir / "vina_summary.csv"
    if not summary_csv.exists():
        print("[WARN] Vina summary CSV가 존재하지 않습니다:", summary_csv)
        return scores

    with open(summary_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            base = row.get("complex")
            val = row.get("vina_score")
            if base is None:
                continue
            try:
                scores[base] = float(val)
            except (TypeError, ValueError):
                scores[base] = None
    return scores


def load_prodigy_scores(prodigy_dir: Path):
    """
    PRODIGY 출력 텍스트(*_prodigy.txt)에서 ΔG(kcal/mol) 값을 추정.
    """
    scores = {}
    if not prodigy_dir.exists():
        return scores

    for txt in prodigy_dir.glob("*_prodigy.txt"):
        base = txt.stem.replace("_prodigy", "")
        try:
            with open(txt) as f:
                text = f.read()
        except Exception:
            continue

        # -50 ~ 0 범위 내 실수 중 가장 작은 값을 ΔG로 사용
        vals = []
        for m in re.finditer(r"[-+]?\d+\.\d+", text):
            v = float(m.group(0))
            if -50.0 <= v <= 0.0:
                vals.append(v)
        scores[base] = min(vals) if vals else None

    return scores


def load_iptm_scores(colabfold_out_dir: Path):
    """
    ColabFold 출력 폴더의 *_scores.json 파일들에서 ipTM 값 로딩.

    실제 파일명/구조는 colabfold 버전에 따라 다를 수 있으므로,
    없으면 그냥 빈 dict를 반환.
    """
    iptms = {}
    if not colabfold_out_dir.exists():
        return iptms

    for js in colabfold_out_dir.glob("*_scores.json"):
        base = js.stem.replace("_scores", "")
        try:
            with open(js) as f:
                data = json.load(f)
        except Exception:
            continue
        if isinstance(data, dict):
            if "iptm" in data:
                iptms[base] = data["iptm"]
            elif "iptm+ptm" in data:
                iptms[base] = data["iptm+ptm"]
    return iptms


def load_plip_scores(plip_dir: Path):
    """
    PLIP 결과 폴더들에서 간단한 상호작용 스코어 추출.
    - 각 complex별 report.txt 에서 Hydrogen bonds / Hydrophobic / Salt bridges 등 숫자를 추정.
    """
    scores = {}
    if not plip_dir.exists():
        return scores

    for subdir in plip_dir.iterdir():
        if not subdir.is_dir():
            continue
        base = subdir.name
        report = subdir / "report.txt"
        if not report.exists():
            continue

        try:
            with open(report) as f:
                text = f.read()
        except Exception:
            continue

        hbond = hydrophobic = saltbridge = 0
        for line in text.splitlines():
            lower = line.lower()
            nums = re.findall(r"\b\d+\b", line)
            if not nums:
                continue
            last_num = int(nums[-1])
            if "hydrogen bond" in lower:
                hbond = last_num
            elif "hydrophobic" in lower:
                hydrophobic = last_num
            elif "salt bridge" in lower:
                saltbridge = last_num

        total = hbond + hydrophobic + saltbridge
        scores[base] = {
            "total": total,
            "hbond": hbond,
            "hydrophobic": hydrophobic,
            "saltbridge": saltbridge,
        }

    return scores

def minmax_norm(value_dict, higher_is_better=True):
    """
    dict(base -> value) 형태를 받아 0~1 범위로 min-max 정규화.
    higher_is_better=True  이면 값이 클수록 1에 가깝게,
    higher_is_better=False 이면 값이 작을수록(예: 에너지가 더 낮을수록) 1에 가깝게.
    """
    vals = [v for v in value_dict.values() if v is not None]
    if not vals:
        return {}

    vmin, vmax = min(vals), max(vals)
    if abs(vmax - vmin) < 1e-8:
        # 전부 같은 값이면 1.0으로 처리
        return {k: 1.0 for k, v in value_dict.items() if v is not None}

    out = {}
    for k, v in value_dict.items():
        if v is None:
            continue
        if higher_is_better:
            s = (v - vmin) / (vmax - vmin)
        else:
            s = (vmax - v) / (vmax - vmin)
        out[k] = s
    return out

def build_and_save_final_table(folders, peptides, rank1_pdbs):
    """
    ColabFold / Vina / PLIP / PRODIGY / ipTM 결과를 모아서
    A안 가중치로 FinalScore_A를 계산하고 엑셀로 저장.

    A안:
      PRODIGY 0.35
      Vina    0.20
      PLIP    0.25
      ipTM    0.20
    """
    results_dir     = folders["results"]
    colabfold_out   = folders["colabfold_out"]
    vina_dir        = folders["vina"]
    plip_dir        = folders["plip"]
    prodigy_dir     = folders["prodigy"]

    # 1) raw metric 로딩
    vina_vals    = load_vina_scores(vina_dir)          # base -> vina_score
    prodigy_vals = load_prodigy_scores(prodigy_dir)    # base -> ΔG (kcal/mol)
    iptm_vals    = load_iptm_scores(colabfold_out)     # base -> ipTM
    plip_metrics = load_plip_scores(plip_dir)          # base -> dict(total, hbond, ...)

    plip_total_vals = {b: d["total"] for b, d in plip_metrics.items()}

    # 2) 정규화 (0~1)
    iptm_norm    = minmax_norm(iptm_vals, higher_is_better=True)
    vina_norm    = minmax_norm(vina_vals, higher_is_better=False)
    prodigy_norm = minmax_norm(prodigy_vals, higher_is_better=False)
    plip_norm    = minmax_norm(plip_total_vals, higher_is_better=True)

    # 3) complex_0 → 펩타이드 서열 매핑
    id_to_pep = {f"complex_{i}": pep for i, pep in enumerate(peptides)}

    # 4) 각 rank_001 PDB마다 row 구성
    rows = []
    for pdb_path in rank1_pdbs:
        base = pdb_path.stem  # complex_0_unrelaxed_rank_001_...
        candidate_id = base.split("_unrelaxed")[0]  # complex_0
        pep_seq = id_to_pep.get(candidate_id, "")

        vina    = vina_vals.get(base)
        prodigy = prodigy_vals.get(base)
        iptm    = iptm_vals.get(base)

        plip_data   = plip_metrics.get(base, {})
        plip_total  = plip_data.get("total")
        plip_hbond  = plip_data.get("hbond")
        plip_hphob  = plip_data.get("hydrophobic")
        plip_salt   = plip_data.get("saltbridge")

        # A안 가중치
        w_prodigy = 0.35
        w_vina    = 0.20
        w_plip    = 0.25
        w_iptm    = 0.20

        final_score = (
            w_prodigy * prodigy_norm.get(base, 0.0) +
            w_vina    * vina_norm.get(base, 0.0) +
            w_plip    * plip_norm.get(base, 0.0) +
            w_iptm    * iptm_norm.get(base, 0.0)
        )

        rows.append({
            "candidate_id": candidate_id,
            "peptide_seq":  pep_seq,
            "complex_pdb":  pdb_path.name,
            "final_score":  final_score,
            "prodigy_dG":   prodigy,
            "vina_score":   vina,
            "plip_total":   plip_total,
            "plip_hbond":   plip_hbond,
            "plip_hphob":   plip_hphob,
            "plip_salt":    plip_salt,
            "iptm":         iptm,
        })

    # 5) FinalScore_A 내림차순 정렬
    rows.sort(key=lambda r: (r["final_score"] if r["final_score"] is not None else -1e9), reverse=True)

    # 6) 엑셀로 저장
    wb = Workbook()
    ws = wb.active
    ws.title = "pepbind_ranking_A"

    headers = [
        "rank",
        "candidate_id",
        "peptide_seq",
        "complex_pdb",
        "FinalScore_A",
        "PRODIGY_dG(kcal/mol)",
        "Vina_score(kcal/mol)",
        "PLIP_total_interactions",
        "PLIP_hbond",
        "PLIP_hydrophobic",
        "PLIP_saltbridge",
        "ipTM",
    ]
    ws.append(headers)

    for idx, r in enumerate(rows, start=1):
        ws.append([
            idx,
            r["candidate_id"],
            r["peptide_seq"],
            r["complex_pdb"],
            round(r["final_score"], 4) if r["final_score"] is not None else None,
            r["prodigy_dG"],
            r["vina_score"],
            r["plip_total"],
            r["plip_hbond"],
            r["plip_hphob"],
            r["plip_salt"],
            r["iptm"],
        ])

    out_xlsx = results_dir / f"final_peptide_ranking_A_{timestamp()}.xlsx"
    wb.save(out_xlsx)
    print(f"✅ 최종 결과 엑셀 저장: {out_xlsx}")
    return out_xlsx

# =====================================================================
# === STEP 5: PLIP 상호작용 분석 =====================================
# =====================================================================

def run_plip_on_rank1(rank1_pdbs, plip_dir: Path):
    """
    PLIP 상호작용 분석.
    - PLIP 역시 CPU 기반 툴.
    - 각 복합체 PDB에 대해 PLIP CLI를 실행해서 결과 폴더만 만들어 둔다.
      (상세 파싱/요약은 필요 시 추가)
    """
    print("\n" + "=" * 80)
    print("STEP 5: PLIP 상호작용 분석")
    print("=" * 80)
    if not rank1_pdbs:
        print("[WARN] PLIP 실행할 rank_001 PDB가 없습니다.")
        return
    if not PLIP_CMD:
        print("[WARN] PLIP_CMD 가 비어 있습니다. 예: export PLIP_CMD='python -m plip.cmd.plip'")
        return

    plip_dir.mkdir(parents=True, exist_ok=True)
    for pdb in rank1_pdbs:
        base = pdb.stem
        out_subdir = plip_dir / base
        out_subdir.mkdir(exist_ok=True)
        cmd = f"{PLIP_CMD} -f {pdb} -o {out_subdir}"
        run(cmd)
        print(f"✔️ PLIP 완료: {pdb.name} → {out_subdir}")
    print("=" * 80)

# =====================================================================
# === STEP 6: PRODIGY 결합 친화도 평가 ===============================
# =====================================================================

def run_prodigy_on_rank1(rank1_pdbs, prodigy_dir: Path):
    """
    PRODIGY 결합 친화도 평가.
    - PRODIGY_SCRIPT 경로(로컬 CLI)를 이용해 호출.
    - 각 복합체 PDB에 대해 출력 파일(*_prodigy.txt) 생성.
    """
    print("\n" + "=" * 80)
    print("STEP 6: PRODIGY 결합 친화도 평가")
    print("=" * 80)
    if not rank1_pdbs:
        print("[WARN] PRODIGY 실행할 rank_001 PDB가 없습니다.")
        return
    if not PRODIGY_SCRIPT:
        print("[WARN] PRODIGY_SCRIPT 환경변수가 설정되어 있지 않습니다.")
        print("       예: export PRODIGY_SCRIPT='/home/aisys/tools/prodigy_cli.py'")
        return

    prodigy_dir.mkdir(parents=True, exist_ok=True)
    for pdb in rank1_pdbs:
        out_txt = prodigy_dir / f"{pdb.stem}_prodigy.txt"
        cmd = f"{sys.executable} {PRODIGY_SCRIPT} -i {pdb} -o {out_txt}"
        run(cmd)
        print(f"✔️ PRODIGY 완료: {pdb.name} → {out_txt}")
    print("=" * 80)

# =====================================================================
# === MAIN ============================================================
# =====================================================================

def main():
    # 1) 워크스페이스 생성
    folders = init_workspace()

    # 2) 타깃 서열 FASTA 저장
    target_seq = TARGET_SEQUENCE.strip()
    target_fasta = write_target_fasta(folders["fasta"], target_seq)
    print(f"✔️ 타깃 단백질 길이: {len(target_seq)}")
    print(f"✔️ 타깃 FASTA: {target_fasta}")

    # 3) PepMLM(ESM-2) 기반 펩타이드 생성
    tokenizer, model = load_esm_mlm()
    peptides = generate_peptides_with_mlm(
        tokenizer,
        model,
        target_seq,
        num_peptides=NUM_PEPTIDES,
        peptide_len=PEPTIDE_LENGTH,
    )
    pep_fasta = write_peptide_fasta(folders["fasta"], peptides)
    print(f"✔️ PepMLM 결과 저장: {pep_fasta}")

    # 4) ColabFold 구조 예측
    rank1_pdbs = []
    if RUN_COLABFOLD and peptides:
        csv_path = prepare_colabfold_batch_csv(
            folders["temp"],
            target_seq,
            peptides,
        )
        rank1_pdbs = run_colabfold_batch_with_progress(
            csv_path,
            folders["colabfold_out"],
            total_complexes=len(peptides),
        )
    else:
        print("\n[INFO] RUN_COLABFOLD=False 또는 펩타이드 없음 → ColabFold 단계 스킵")

    # 5) Vina / PLIP / PRODIGY
    if RUN_VINA:
        run_vina_on_rank1(rank1_pdbs, folders["vina"])
    else:
        print("\n[INFO] RUN_VINA=False → Vina 단계 스킵")

    if RUN_PLIP:
        run_plip_on_rank1(rank1_pdbs, folders["plip"])
    else:
        print("[INFO] RUN_PLIP=False → PLIP 단계 스킵")

    if RUN_PRODIGY:
        run_prodigy_on_rank1(rank1_pdbs, folders["prodigy"])
    else:
        print("[INFO] RUN_PRODIGY=False → PRODIGY 단계 스킵")

    print("\n" + "=" * 80)
    print("🎉 파이프라인 실행 종료")
    print("=" * 80)
    print(f"[INFO] 워크스페이스: {folders['root']}")
    print("=" * 80)

    # 5) Vina / PLIP / PRODIGY
    if RUN_VINA:
        run_vina_on_rank1(rank1_pdbs, folders["vina"])
    else:
        print("\n[INFO] RUN_VINA=False → Vina 단계 스킵")

    if RUN_PLIP:
        run_plip_on_rank1(rank1_pdbs, folders["plip"])
    else:
        print("[INFO] RUN_PLIP=False → PLIP 단계 스킵")

    if RUN_PRODIGY:
        run_prodigy_on_rank1(rank1_pdbs, folders["prodigy"])
    else:
        print("[INFO] RUN_PRODIGY=False → PRODIGY 단계 스킵")

    # 6) rank_001 PDB zip 압축 + A안 최종 엑셀
    pdb_zip = None
    final_xlsx = None
    if rank1_pdbs:
        pdb_zip   = zip_rank1_pdbs(rank1_pdbs, folders["results"])
        final_xlsx = build_and_save_final_table(folders, peptides, rank1_pdbs)
    else:
        print("[INFO] rank_001 PDB가 없어 zip/엑셀 생성을 생략합니다.")

    print("\n" + "=" * 80)
    print("🎉 파이프라인 실행 종료")
    print("=" * 80)
    print(f"[INFO] 워크스페이스: {folders['root']}")
    if pdb_zip:
        print(f"[INFO] PDB zip: {pdb_zip}")
    if final_xlsx:
        print(f"[INFO] 최종 엑셀: {final_xlsx}")
    print("=" * 80)

if __name__ == "__main__":
    main()
