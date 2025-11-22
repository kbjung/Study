#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pepbind_pipeline.py - WSL/오프라인 환경용 통합 파이프라인 (정리 버전)

구성:
- STEP 2: PepMLM(ESM-2)로 펩타이드 후보 생성 (GPU 사용)
- STEP 3: ColabFold 멀티머로 타깃-펩타이드 복합체 구조 예측 (진행 상황 표시)
- STEP 4: AutoDock Vina 도킹 (CPU, stdout 파싱)
- STEP 5: PLIP 상호작용 분석
- STEP 6: PRODIGY 결합 자유에너지 평가
- STEP 7: 최종 평가(A안 가중치) + 엑셀 파일 생성 + rank_001 PDB zip 압축

A안 가중치:
  PRODIGY 0.35
  Vina    0.20
  PLIP    0.25
  ipTM    0.20
"""

import os
import time
import csv
import re
import json
import zipfile
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM
from Bio.PDB import PDBParser, PDBIO, Select
from openpyxl import Workbook
import pandas as pd

import xml.etree.ElementTree as ET

START_TIME = datetime.now()

# =====================================================================
# === 사용자 설정 영역: 여기만 수정해서 사용 ==========================
# =====================================================================

# 1) 타깃 단백질 서열 (FASTA의 sequence 부분만)
TARGET_SEQUENCE = (
    "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYNKINQRILVVDPVTSEHELTCQAEGYPKAEVIWTSSDHQVLSGKTTTTNSKREEKLFNVTSTLRINTTTNEIFYCTFRRLDPEENHTAELVIPELPLAHPPNERT" # PD-L1 단백질 서열
)

# 2) 생성할 펩타이드 설정
NUM_PEPTIDES   = 10   # 생성할 펩타이드 후보 개수
PEPTIDE_LENGTH = 4    # 각 펩타이드 길이 (아미노산 개수)

# 3) ColabFold / 평가 단계 사용 여부
RUN_COLABFOLD  = True   # ColabFold 구조 예측 실행 여부
RUN_VINA       = True   # AutoDock Vina 도킹 실행 여부
RUN_PLIP       = True   # PLIP 상호작용 분석 실행 여부
RUN_PRODIGY    = True   # PRODIGY 결합 친화도 평가 실행 여부

# 4) 작업 기본 디렉토리
BASE_DIR = Path(os.environ.get("PEPBIND_BASE_DIR", "~/work/pipeline")).expanduser()

# 5) 외부 도구 경로 (환경에 맞게 수정 가능)
COLABFOLD_CMD   = os.environ.get("COLABFOLD_CMD", "colabfold_batch").strip()
VINA_CMD        = os.environ.get("VINA_CMD", "vina").strip()
PLIP_CMD        = os.environ.get("PLIP_CMD", "plip").strip()          # 기본값도 plip으로
PRODIGY_SCRIPT  = os.environ.get("PRODIGY_SCRIPT", "prodigy").strip()


# =====================================================================
# === 공통 설정 / 유틸 =================================================
# =====================================================================

BASE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] PyTorch device: {DEVICE}")


def run(cmd, cwd=None):
    """서브프로세스 실행 래퍼 (간단 버전)."""
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


def parse_prodigy_dg_from_stdout(stdout: str):
    """
    PRODIGY stdout에서 ΔG(또는 binding energy)를 추출하는 헬퍼 함수.
    형식 예시:
      Binding energy: -12.3 kcal/mol
      Predicted ΔG: -10.5 kcal/mol

    여러 줄 중 첫 번째 매칭값만 사용.
    실패하면 None 리턴.
    """
    if not stdout:
        return None

    # 1) 가장 자주 나오는 패턴들 우선 시도
    patterns = [
        r"Binding energy\s*[:=]\s*([\-+]?\d+(?:\.\d+)?)",   # Binding energy: -12.3
        r"Predicted\s*Δ?G\s*[:=]\s*([\-+]?\d+(?:\.\d+)?)",  # Predicted ΔG: -10.5
        r"\bΔG\s*[:=]\s*([\-+]?\d+(?:\.\d+)?)",             # ΔG: -9.87
    ]

    for pat in patterns:
        m = re.search(pat, stdout, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass

    # 2) 백업: stdout 전체에서 "소수점이 있는 첫 번째 실수"
    m = re.search(r"([\-+]?\d+\.\d+)", stdout)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None

    return None


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
        getattr(tokenizer, "cls_token_id", None),
        getattr(tokenizer, "sep_token_id", None),
        tokenizer.mask_token_id,
        getattr(tokenizer, "unk_token_id", None),
    ]:
        if tid is not None:
            bad_ids.add(tid)

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

                    for bid in bad_ids:
                        probs[bid] = 0.0

                    probs = probs / probs.sum()

                    k = min(top_k, probs.size(0))
                    top_vals, top_idx = torch.topk(probs, k=k)
                    top_vals = top_vals / top_vals.sum()
                    sampled_local = torch.multinomial(top_vals, num_samples=1)
                    sampled_id = top_idx[sampled_local]
                    ids[0, pos] = sampled_id

            seq = tokenizer.decode(ids[0], skip_special_tokens=True).replace(" ", "")
            pep = seq[-peptide_len:]

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
# === STEP 3: ColabFold 배치 (멀티머) ================================
# =====================================================================

def prepare_colabfold_batch_csv(temp_dir: Path, target_sequence: str, peptides) -> Path:
    """
    ColabFold 1.5.5용 multimer CSV 입력 생성.

    - CSV 컬럼: id, sequence
    - sequence 형식: "타깃서열:펩타이드서열"
      예) AAAAA...AAAA:PPPP
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


# =====================================================================
# === STEP 4: AutoDock Vina 도킹 =====================================
# =====================================================================

class ChainSelect(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.get_id() == self.chain_id


def split_complex_to_receptor_ligand(
    complex_pdb: Path,
    out_dir: Path,
    receptor_chain: str = "A",
    ligand_chain: str = "B",
):
    """
    간단 가정:
    - ColabFold 멀티머 출력에서 체인 A: 타깃 단백질
    - 체인 B: 펩타이드
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
    1) AutoDockTools 스크립트(prepare_receptor4.py, prepare_ligand4.py)가 있으면 그것 사용
    2) 없으면 obabel 사용
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
        run(f"{obabel} -ipdb {rec_pdb} -xr -opdbqt -O {rec_pdbqt}")
        run(f"{obabel} -ipdb {lig_pdb}      -opdbqt -O {lig_pdbqt}")
    else:
        raise RuntimeError("PDBQT 변환 도구(prepare_* 또는 obabel)를 찾을 수 없습니다.")

    return rec_pdbqt, lig_pdbqt


def parse_vina_score_from_stdout(stdout: str):
    """
    AutoDock Vina stdout에서 best score(affinity, kcal/mol)를 파싱.

    우선순위:
    1) mode 테이블 (mode | affinity | ...)에서 affinity 열 파싱
    2) fallback: 'REMARK VINA RESULT' 형식이 있으면 그 줄에서 float 추출
    """
    energies = []

    # 1) mode 테이블 파싱
    for line in stdout.splitlines():
        s = line.strip()
        if not s:
            continue
        # 헤더/구분선은 건너뛰기
        if s.startswith("mode") or set(s) <= {"-", "+"}:
            continue

        parts = s.split()
        # "1  -7.5  ..." 이런 형식일 때
        if parts and parts[0].isdigit() and len(parts) >= 2:
            try:
                val = float(parts[1])
            except ValueError:
                continue
            energies.append(val)

    if energies:
        # 가장 낮은 에너지(가장 좋은 포즈)를 반환
        return min(energies)

    # 2) fallback: 예전 스타일 'REMARK VINA RESULT:' 줄
    for line in stdout.splitlines():
        if "REMARK VINA RESULT" in line:
            for token in line.split():
                try:
                    return float(token)
                except ValueError:
                    continue

    # 아무 것도 못 찾으면 None
    return None


def run_vina_on_rank1(rank1_pdbs, vina_dir: Path):
    """
    AutoDock Vina 도킹 실행 + 요약/상태 로그 기록.

    - vina_summary.csv 컬럼:
        complex, vina_score, vina_status, receptor_pdbqt, ligand_pdbqt, log_file
    - vina_status 예시:
        '정상', '정상(점수=0.0)',
        '실패: Vina 실행 에러(code=...)',
        '실패: Vina 실행 에러(code=...)(PDBQT parsing error: ...)',
        '파싱실패: stdout에서 점수 패턴 없음'
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
    summary_rows = [["complex", "vina_score", "vina_status", "receptor_pdbqt", "ligand_pdbqt", "log_file"]]
    debug_lines = []

    for complex_pdb in rank1_pdbs:
        base = complex_pdb.stem
        complex_out_dir = vina_dir / base
        complex_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[INFO] Vina 준비: {complex_pdb.name}")
        rec_pdb, lig_pdb = split_complex_to_receptor_ligand(
            complex_pdb,
            complex_out_dir,
            receptor_chain="A",
            ligand_chain="B",
        )

        rec_pdbqt, lig_pdbqt = prepare_pdbqt(rec_pdb, lig_pdb, complex_out_dir)
        box = compute_box_from_ligand(lig_pdb)

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

        with open(log_file, "w", encoding="utf-8") as lf:
            lf.write("=== STDOUT ===\n")
            lf.write(result.stdout or "")
            lf.write("\n\n=== STDERR ===\n")
            lf.write(result.stderr or "")

        best_score = None
        status = ""

        if result.returncode != 0:
            # 실행 자체 실패
            status = f"실패: Vina 실행 에러(code={result.returncode})"
            if "PDBQT parsing error" in (result.stdout or "") or "PDBQT parsing error" in (result.stderr or ""):
                status += " (PDBQT parsing error: flex residue/ligand 태그 문제)"
            print(f"[ERROR] {status}. 로그 파일: {log_file}")
            print(result.stdout or "")
            print(result.stderr or "")
            debug_lines.append(f"{base}\t{status}\tlog={log_file.name}")
        else:
            # 실행은 성공 → score 파싱
            best_score = parse_vina_score_from_stdout(result.stdout or "")
            if best_score is None:
                status = "파싱실패: stdout에서 점수 패턴 없음"
                print(f"[WARN] {complex_pdb.name} Vina 점수 파싱 실패. 로그 파일 확인: {log_file}")
                debug_lines.append(f"{base}\t{status}\tlog={log_file.name}")
            else:
                if best_score == 0.0:
                    status = "정상(점수=0.0)"
                else:
                    status = "정상"
                print(f"[INFO] {complex_pdb.name} Vina best score: {best_score}")
                debug_lines.append(f"{base}\t{status}\t{best_score}")

        summary_rows.append([base, best_score, status, rec_pdbqt.name, lig_pdbqt.name, log_file.name])

    summary_csv = vina_dir / "vina_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(summary_rows)
    print(f"\n✅ Vina 요약 저장: {summary_csv}")

    if debug_lines:
        debug_file = vina_dir / "vina_debug.txt"
        debug_file.write_text("\n".join(debug_lines), encoding="utf-8")
        print(f"[INFO] Vina 디버그 로그: {debug_file}")

    print("=" * 80)


# =====================================================================
# === STEP 5: PLIP 상호작용 분석 =====================================
# =====================================================================

def run_plip_on_rank1(rank1_pdbs, plip_dir: Path):
    """
    PLIP 상호작용 분석.
    - 각 complex별 stdout/stderr 를 로그 파일로 저장.
    """
    print("\n" + "=" * 80)
    print("STEP 5: PLIP 상호작용 분석")
    print("=" * 80)
    if not rank1_pdbs:
        print("[WARN] PLIP 실행할 rank_001 PDB가 없습니다.")
        return
    if not PLIP_CMD:
        print("[WARN] PLIP_CMD 가 비어 있습니다.")
        return

    plip_dir.mkdir(parents=True, exist_ok=True)
    debug_lines = []

    for pdb in rank1_pdbs:
        base = pdb.stem
        out_subdir = plip_dir / base
        out_subdir.mkdir(exist_ok=True)
        cmd = f"{PLIP_CMD} -f {pdb} -o {out_subdir}"
        print(f"[RUN] {cmd}")
        log_file = out_subdir / f"{base}_plip.log"

        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        with open(log_file, "w", encoding="utf-8") as lf:
            lf.write("=== STDOUT ===\n")
            lf.write(result.stdout or "")
            lf.write("\n\n=== STDERR ===\n")
            lf.write(result.stderr or "")

        if result.returncode != 0:
            print(f"[ERROR] PLIP 실패: {pdb.name} (code={result.returncode}). 로그: {log_file}")
            print((result.stderr or "")[:300])
            debug_lines.append(f"{base}\tERROR_returncode_{result.returncode}\tlog={log_file.name}")
        else:
            print(f"✔️ PLIP 완료: {pdb.name} → {out_subdir}")
            debug_lines.append(f"{base}\tOK\tlog={log_file.name}")

    if debug_lines:
        debug_file = plip_dir / "plip_run_debug.txt"
        debug_file.write_text("\n".join(debug_lines), encoding="utf-8")
        print(f"[INFO] PLIP 실행 디버그 로그: {debug_file}")
    print("=" * 80)


# =====================================================================
# === STEP 6: PRODIGY 결합 친화도 평가 ===============================
# =====================================================================

def run_prodigy_on_rank1(rank1_pdbs, out_dir: Path) -> pd.DataFrame:
    """
    PRODIGY 결합 친화도 평가.

    - prodigy_summary.csv 컬럼:
        complex, PRODIGY_dG, prodigy_status
    - prodigy_status 예시:
        '정상',
        '실패: PRODIGY 실행 에러(code=...)',
        '파싱실패: stdout에서 ΔG 패턴 없음'
    """
    print("\n" + "="*80)
    print("STEP 6: PRODIGY 결합 친화도 평가")
    print("="*80)

    if not PRODIGY_SCRIPT:
        print("[WARN] PRODIGY_SCRIPT 환경변수가 설정되어 있지 않습니다.")
        print("       예: export PRODIGY_SCRIPT='prodigy'")
        return pd.DataFrame()

    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    debug_lines = []

    for pdb_path in rank1_pdbs:
        complex_name = Path(pdb_path).stem
        out_txt = out_dir / f"{complex_name}_prodigy.txt"
        err_txt = out_dir / f"{complex_name}_prodigy.stderr.txt"

        cmd = [
            *PRODIGY_SCRIPT.split(),
            str(pdb_path),
            "--selection", "A", "B",
        ]
        print(f"[RUN] {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        # stdout / stderr 저장
        out_txt.write_text(result.stdout or "", encoding="utf-8")
        err_txt.write_text(result.stderr or "", encoding="utf-8")

        dg = None
        status = ""

        if result.returncode != 0:
            status = f"실패: PRODIGY 실행 에러(code={result.returncode})"
            print(f"[ERROR] {status}: {complex_name}. 로그: {out_txt.name}, {err_txt.name}")
            print((result.stderr or "")[:300])
        else:
            # stdout에서 ΔG 값 파싱
            dg = parse_prodigy_dg_from_stdout(result.stdout or "")
            if dg is None:
                status = "파싱실패: stdout에서 ΔG 패턴 없음"
                print(f"[WARN] PRODIGY ΔG 파싱 실패: {complex_name}. (로그: {out_txt.name})")
            else:
                status = "정상"

        records.append({
            "complex": complex_name,
            "PRODIGY_dG": dg,
            "prodigy_status": status,
        })
        debug_lines.append(f"{complex_name}\t{status}\t{dg}")

    if debug_lines:
        debug_file = out_dir / "prodigy_debug.txt"
        debug_file.write_text("\n".join(debug_lines), encoding="utf-8")
        print(f"[INFO] PRODIGY 디버그 로그: {debug_file}")

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "prodigy_summary.csv", index=False)
    print(f"✅ PRODIGY 요약 저장: {out_dir / 'prodigy_summary.csv'}")
    return df


# =====================================================================
# === STEP 7: 최종 평가(가중치 A안) + PDB zip + 엑셀 =================
# =====================================================================

def zip_rank1_pdbs(rank1_pdbs, results_dir: Path):
    """
    ColabFold에서 생성된 rank_001 PDB들을 하나의 zip 파일로 압축.
    (타깃 단백질 + 생성 펩타이드 복합체 구조)
    """
    if not rank1_pdbs:
        print("[INFO] zip으로 묶을 rank_001 PDB가 없습니다.")
        return None

    zip_path = results_dir / f"peptide_structures_{timestamp()}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for pdb_path in rank1_pdbs:
            zf.write(pdb_path, arcname=pdb_path.name)

    print(f"✅ rank_001 PDB 압축 저장: {zip_path}")
    return zip_path


def load_vina_scores(vina_dir: Path):
    """
    vina_summary.csv 에서 complex별 Vina score와 상태를 로딩.

    반환:
      scores   : dict[complex_id] = float 또는 None
      statuses : dict[complex_id] = 상태 문자열
    """
    scores = {}
    statuses = {}

    summary_csv = vina_dir / "vina_summary.csv"
    if not summary_csv.exists():
        print("[WARN] Vina summary CSV가 존재하지 않습니다:", summary_csv)
        return scores, statuses

    try:
        df = pd.read_csv(summary_csv)
    except Exception as e:
        print(f"[WARN] Vina summary CSV 로딩 실패: {e}")
        return scores, statuses

    if "complex" not in df.columns:
        print("[WARN] Vina summary CSV에 'complex' 컬럼이 없습니다.")
        return scores, statuses

    has_status = "vina_status" in df.columns

    for _, row in df.iterrows():
        base = row.get("complex")
        if isinstance(base, float) and pd.isna(base):
            continue
        base = str(base).strip()
        if not base:
            continue

        val = row.get("vina_score")
        try:
            scores[base] = float(val)
        except (TypeError, ValueError):
            scores[base] = None

        if has_status:
            s = row.get("vina_status")
            statuses[base] = "" if pd.isna(s) else str(s)
        else:
            # 구버전 summary에는 상태가 없으므로 점수 유무로만 대략 추정
            statuses[base] = "정상" if scores[base] is not None else "미기록"

    print(f"[INFO] Vina 점수를 읽어온 구조 수: {len(scores)}")
    return scores, statuses


def load_prodigy_scores(prodigy_dir: Path):
    """
    PRODIGY 결과 로딩.

    우선순위:
      1) prodigy_summary.csv (있다면)
      2) 없으면 *_prodigy.txt 백업 파싱

    반환:
      scores   : dict[complex_id] = float 또는 None
      statuses : dict[complex_id] = 상태 문자열
    """
    scores = {}
    statuses = {}

    if not prodigy_dir.exists():
        return scores, statuses

    summary_csv = prodigy_dir / "prodigy_summary.csv"
    if summary_csv.exists():
        try:
            df = pd.read_csv(summary_csv)
            if "complex" in df.columns:
                # 값 컬럼 찾기 (prodigy_dg, PRODGIGY_dG 등)
                val_col = None
                for c in df.columns:
                    cl = c.lower()
                    if cl.startswith("prodigy") and "status" not in cl:
                        val_col = c
                        break
                has_status = "prodigy_status" in df.columns

                for _, row in df.iterrows():
                    comp = row.get("complex")
                    if isinstance(comp, float) and pd.isna(comp):
                        continue
                    comp = str(comp).strip()
                    if not comp:
                        continue

                    val = row.get(val_col) if val_col is not None else None
                    try:
                        scores[comp] = float(val)
                    except (TypeError, ValueError):
                        scores[comp] = None

                    if has_status:
                        s = row.get("prodigy_status")
                        statuses[comp] = "" if pd.isna(s) else str(s)
                    else:
                        statuses[comp] = "정상" if scores[comp] is not None else "미기록"

            print(f"[INFO] PRODIGY ΔG를 요약 파일에서 불러옴: {len(scores)}개 구조")
        except Exception as e:
            print(f"[WARN] prodigy_summary.csv 로딩 실패: {e}")

    # summary에서 뭔가 읽어왔다면 그대로 사용
    if scores or statuses:
        return scores, statuses

    # 백업: *_prodigy.txt 직접 파싱
    for txt in prodigy_dir.glob("*_prodigy.txt"):
        base = txt.stem.replace("_prodigy", "")
        try:
            text = txt.read_text()
        except Exception:
            continue

        vals = []
        for m in re.finditer(r"[-+]?\d+\.\d+", text):
            v = float(m.group(0))
            # PRODIGY ΔG 대략적인 범위
            if -50.0 <= v <= 0.0:
                vals.append(v)

        if vals:
            scores[base] = min(vals)
            statuses[base] = "텍스트 파싱(백업)"
        else:
            scores[base] = None
            statuses[base] = "파싱실패: *_prodigy.txt에서 ΔG 후보값 없음"

    print(f"[INFO] PRODIGY ΔG를 텍스트에서 직접 파싱: {len(scores)}개 구조")
    return scores, statuses


def load_iptm_scores(colabfold_out_dir: Path, rank1_pdbs):
    """
    ColabFold 출력 폴더에서 ipTM 값을 최대한 유연하게 찾는다.

    - 각 rank_001 PDB의 stem(base)를 기준으로
      1) base*scores*.json
      2) base_prefix*scores*.json  (base에서 '_unrelaxed' 앞부분)
      3) base*_ranking_debug.json, base_prefix*_ranking_debug.json, ranking_debug.json
    에서 'iptm' 또는 'iptm+ptm' 키를 찾아본다.
    """
    iptms = {}
    if not colabfold_out_dir.exists():
        return iptms

    for pdb in rank1_pdbs:
        base = pdb.stem
        prefix = base.split("_unrelaxed")[0]

        found_val = None

        # 1) scores*.json 후보들
        candidates = list(colabfold_out_dir.glob(f"{base}*scores*.json"))
        if not candidates:
            candidates = list(colabfold_out_dir.glob(f"{prefix}*scores*.json"))

        for js in candidates:
            try:
                with open(js) as f:
                    data = json.load(f)
            except Exception:
                continue

            if isinstance(data, dict):
                v = data.get("iptm")
                if isinstance(v, (int, float)):
                    found_val = float(v)
                    break
                v = data.get("iptm+ptm")
                if isinstance(v, (int, float)):
                    found_val = float(v)
                    break

        # 2) ranking_debug 후보들
        if found_val is None:
            rd_candidates = [
                colabfold_out_dir / f"{base}_ranking_debug.json",
                colabfold_out_dir / f"{prefix}_ranking_debug.json",
                colabfold_out_dir / "ranking_debug.json",
            ]
            for rd in rd_candidates:
                if not rd.exists():
                    continue
                try:
                    with open(rd) as f:
                        data = json.load(f)
                except Exception:
                    continue

                if isinstance(data, dict):
                    v = data.get("iptm") or data.get("iptm+ptm")
                    if isinstance(v, (int, float)):
                        found_val = float(v)
                        break

        if found_val is not None:
            iptms[base] = found_val

    print(f"[INFO] ipTM 값을 읽어온 구조 수: {len(iptms)} / {len(rank1_pdbs)}")
    return iptms


def load_plip_scores(plip_dir: Path):
    """
    PLIP 결과 폴더들에서 상호작용 스코어 + 상태를 추출.

    - 각 complex별 서브폴더 이름이 complex_0_unrelaxed_... 형태라고 가정.
    - 우선 report.xml, 없으면 report.txt를 사용.
    - 아무 것도 없으면 plip_status에 이유를 기록하고 상호작용 수는 None으로 둔다.

    반환:
      metrics  : dict[base] = {
                    "total": total or None,
                    "hbond": ...,
                    "hydrophobic": ...,
                    "saltbridge": ...
                }
      statuses : dict[base] = 상태 문자열
    """
    metrics = {}
    statuses = {}

    if not plip_dir.exists():
        return metrics, statuses

    debug_lines = []

    for subdir in plip_dir.iterdir():
        if not subdir.is_dir():
            continue

        base = subdir.name
        xml_path = subdir / "report.xml"
        txt_report = subdir / "report.txt"

        hbond = 0
        hydrophobic = 0
        saltbridge = 0
        total = None

        source = None
        status = ""

        # 1) report.xml 우선
        if xml_path.exists():
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                hbond = sum(1 for _ in root.iter("hydrogen_bond"))
                hydrophobic = sum(1 for _ in root.iter("hydrophobic_interaction"))
                saltbridge = sum(1 for _ in root.iter("salt_bridge"))
                total = hbond + hydrophobic + saltbridge
                source = "xml"
                status = "정상(xml)"
            except Exception as e:
                status = f"실패: report.xml 파싱 에러({e})"
                debug_lines.append(f"{base}\treport_xml_parse_error\t{e}")

        # 2) report.txt 백업
        if source is None and txt_report.exists():
            try:
                text = txt_report.read_text()
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
                source = "report.txt"
                status = "정상(report.txt)"
            except Exception as e:
                status = f"실패: report.txt 파싱 에러({e})"
                debug_lines.append(f"{base}\treport_txt_parse_error\t{e}")

        # 3) 둘 다 없으면 → 실패 처리
        if source is None and not status:
            status = "실패: report.xml/report.txt 없음 (PLIP 출력에 상호작용 요약 파일이 없음)"

        # total이 없거나 실패 상태면 세부값을 None으로
        if total is None or status.startswith("실패"):
            metrics[base] = {
                "total": None,
                "hbond": None,
                "hydrophobic": None,
                "saltbridge": None,
            }
        else:
            metrics[base] = {
                "total": total,
                "hbond": hbond,
                "hydrophobic": hydrophobic,
                "saltbridge": saltbridge,
            }

        statuses[base] = status
        debug_lines.append(
            f"{base}\t{status}\t"
            f"total={metrics[base]['total']}\t"
            f"hbond={metrics[base]['hbond']}\t"
            f"hydrophobic={metrics[base]['hydrophobic']}\t"
            f"saltbridge={metrics[base]['saltbridge']}"
        )

    # 디버그 로그
    if debug_lines:
        debug_file = plip_dir / "plip_parse_debug.txt"
        debug_file.write_text("\n".join(debug_lines), encoding="utf-8")
        print(f"[INFO] PLIP 파싱 디버그 로그: {debug_file}")

    # 요약 CSV (각 complex별 한 줄)
    summary_rows = []
    for base, m in metrics.items():
        summary_rows.append({
            "complex": base,
            "plip_total_interactions": m["total"],
            "plip_hbond": m["hbond"],
            "plip_hydrophobic": m["hydrophobic"],
            "plip_saltbridge": m["saltbridge"],
            "plip_status": statuses.get(base, ""),
        })

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        df.to_csv(plip_dir / "plip_summary.csv", index=False)
        print(f"[INFO] PLIP 요약 CSV 저장: {plip_dir / 'plip_summary.csv'}")

    print(f"[INFO] PLIP 상호작용을 읽어온 구조 수: {len(metrics)}")
    return metrics, statuses


def minmax_norm(value_dict, higher_is_better=True):
    """
    dict(base -> value) 형태를 받아 0~1 범위로 min-max 정규화.
    higher_is_better=True  이면 값이 클수록 1에 가깝게,
    higher_is_better=False 이면 값이 작을수록(에너지가 더 낮을수록) 1에 가깝게.
    """
    vals = [v for v in value_dict.values() if v is not None]
    if not vals:
        return {}

    vmin, vmax = min(vals), max(vals)
    if abs(vmax - vmin) < 1e-8:
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
      PRODIGY 0.35  (ΔG, 더 작을수록 좋음)
      Vina    0.20  (에너지, 더 작을수록 좋음)
      PLIP    0.25  (총 상호작용 수, 많을수록 좋음)
      ipTM    0.20  (인터페이스 신뢰도, 높을수록 좋음)

    추가:
      - 각 평가 모델별 status 컬럼을 엑셀에 함께 기록
        (vina_status, prodigy_status, plip_status)
    """
    results_dir     = folders["results"]
    colabfold_out   = folders["colabfold_out"]
    vina_dir        = folders["vina"]
    plip_dir        = folders["plip"]
    prodigy_dir     = folders["prodigy"]

    # 각 평가 지표 + 상태 불러오기
    vina_vals, vina_status         = load_vina_scores(vina_dir)
    prodigy_vals, prodigy_status   = load_prodigy_scores(prodigy_dir)
    iptm_vals                      = load_iptm_scores(colabfold_out, rank1_pdbs)
    plip_metrics, plip_status      = load_plip_scores(plip_dir)

    # PLIP total 값만 따로 dict로 추출
    plip_total_vals = {b: d.get("total") for b, d in plip_metrics.items()}

    # 0~1 정규화
    iptm_norm    = minmax_norm(iptm_vals, higher_is_better=True)
    vina_norm    = minmax_norm(vina_vals, higher_is_better=False)
    prodigy_norm = minmax_norm(prodigy_vals, higher_is_better=False)
    plip_norm    = minmax_norm(plip_total_vals, higher_is_better=True)

    # candidate_id → peptide 매핑 (complex_0, complex_1 ...)
    id_to_pep = {f"complex_{i}": pep for i, pep in enumerate(peptides)}

    rows = []
    for pdb_path in rank1_pdbs:
        base = pdb_path.stem                                       # complex_0_unrelaxed_...
        candidate_id = base.split("_unrelaxed")[0]                  # complex_0
        pep_seq = id_to_pep.get(candidate_id, "")

        vina    = vina_vals.get(base)
        prodigy = prodigy_vals.get(base)
        iptm    = iptm_vals.get(base)

        plip_data   = plip_metrics.get(base, {})
        plip_total  = plip_data.get("total")
        plip_hbond  = plip_data.get("hbond")
        plip_hphob  = plip_data.get("hydrophobic")
        plip_salt   = plip_data.get("saltbridge")

        # 가중치
        w_prodigy = 0.35
        w_vina    = 0.20
        w_plip    = 0.25
        w_iptm    = 0.20

        # 정규화된 점수 조합
        final_score = (
            w_prodigy * prodigy_norm.get(base, 0.0) +
            w_vina    * vina_norm.get(base, 0.0) +
            w_plip    * plip_norm.get(base, 0.0) +
            w_iptm    * iptm_norm.get(base, 0.0)
        )

        rows.append({
            "candidate_id":     candidate_id,
            "peptide_seq":      pep_seq,
            "complex_pdb":      pdb_path.name,
            "final_score":      final_score,
            "prodigy_dG":       prodigy,
            "prodigy_status":   prodigy_status.get(base, "미기록"),
            "vina_score":       vina,
            "vina_status":      vina_status.get(base, "미기록"),
            "plip_total":       plip_total,
            "plip_hbond":       plip_hbond,
            "plip_hphob":       plip_hphob,
            "plip_salt":        plip_salt,
            "plip_status":      plip_status.get(base, "미기록"),
            "iptm":             iptm,
        })

    # FinalScore_A 기준으로 내림차순 정렬
    rows.sort(
        key=lambda r: (r["final_score"] if r["final_score"] is not None else -1e9),
        reverse=True,
    )

    # 엑셀 작성
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
        "PRODIGY_status",
        "Vina_score(kcal/mol)",
        "Vina_status",
        "PLIP_total_interactions",
        "PLIP_hbond",
        "PLIP_hydrophobic",
        "PLIP_saltbridge",
        "PLIP_status",
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
            r["prodigy_status"],
            r["vina_score"],
            r["vina_status"],
            r["plip_total"],
            r["plip_hbond"],
            r["plip_hphob"],
            r["plip_salt"],
            r["plip_status"],
            r["iptm"],
        ])

    out_xlsx = results_dir / f"final_peptide_ranking_A_{timestamp()}.xlsx"
    wb.save(out_xlsx)
    print(f"✅ 최종 결과 엑셀 저장: {out_xlsx}")
    return out_xlsx


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

    # 6) rank_001 PDB zip 압축 + A안 최종 엑셀
    pdb_zip = None
    final_xlsx = None
    if rank1_pdbs:
        pdb_zip   = zip_rank1_pdbs(rank1_pdbs, folders["results"])
        final_xlsx = build_and_save_final_table(folders, peptides, rank1_pdbs)
    else:
        print("[INFO] rank_001 PDB가 없어 zip/엑셀 생성을 생략합니다.")

    # 종료 시간 및 소요 시간 계산
    END_TIME = datetime.now()

    start_str = START_TIME.strftime("%Y.%m.%d %H:%M:%S")
    end_str   = END_TIME.strftime("%Y.%m.%d %H:%M:%S")

    elapsed = END_TIME - START_TIME
    total_seconds = int(elapsed.total_seconds())

    days = total_seconds // (24 * 3600)
    total_seconds %= (24 * 3600)
    hours = total_seconds // 3600
    total_seconds %= 3600
    minutes = total_seconds // 60
    seconds = total_seconds % 60

    # "00일 00시간 00분 00초" 형태에서
    # 일/시간은 0이면 생략
    parts = []
    if days > 0:
        parts.append(f"{days:02d}일")
    if days > 0 or hours > 0:
        parts.append(f"{hours:02d}시간")
    parts.append(f"{minutes:02d}분")
    parts.append(f"{seconds:02d}초")
    elapsed_str = " ".join(parts)

    print("\n" + "=" * 80)
    print("🎉 파이프라인 실행 종료")
    print("=" * 80)
    print(f"[INFO] 워크스페이스: {folders['root']}")
    if pdb_zip:
        print(f"[INFO] PDB zip: {pdb_zip}")
    if final_xlsx:
        print(f"[INFO] 최종 엑셀: {final_xlsx}")
    print(f"[INFO] 시작 시간: {start_str}")
    print(f"[INFO] 종료 시간: {end_str}")
    print(f"[INFO] 총 소요 시간: {elapsed_str}")
    print("=" * 80)


if __name__ == "__main__":
    main()
