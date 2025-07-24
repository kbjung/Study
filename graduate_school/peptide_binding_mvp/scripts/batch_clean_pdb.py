import os
from pathlib import Path
from clean_pdb import clean_pdb

# Input and output directories
input_dir = Path("results")
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

# File pattern: model_01.pdb ~ model_10.pdb
for i in range(1, 11):
    model_name = f"model_{i:02d}.pdb"
    input_path = input_dir / model_name
    output_path = output_dir / f"model_{i:02d}_cleaned.pdb"
    if input_path.exists():
        print(f"✔ Cleaning {model_name}...")
        clean_pdb(str(input_path), str(output_path))
    else:
        print(f"✘ {model_name} not found. Skipping.")