import os
import pandas as pd

def parse_zdock_scores(score_file_path, top_n=10, output_csv_path=None):
    with open(score_file_path, "r") as f:
        lines = f.readlines()

    # Skip ZDOCK metadata/header lines
    score_lines = lines[5:]

    # Extract scores
    parsed_scores = []
    for idx, line in enumerate(score_lines):
        parts = line.strip().split()
        if len(parts) == 7:
            phi, theta, psi, rec_res, lig_res, angle_id, score = parts
            parsed_scores.append({
                "Model": f"model_{idx+1:02d}",
                "Score": float(score)
            })

    df = pd.DataFrame(parsed_scores)
    df_sorted = df.sort_values(by="Score", ascending=False).head(top_n)

    if output_csv_path:
        df_sorted.to_csv(output_csv_path, index=False)
        print(f"✅ Top {top_n} model scores saved to: {output_csv_path}")
    else:
        print(f"✅ Top {top_n} ZDOCK model scores:")
        print(df_sorted)

    return df_sorted

if __name__ == "__main__":
    # Adjust paths according to your local setup
    score_file = "results/job.479551.zd3.0.2.out"
    output_csv = "results/top_zdock_scores.csv"

    parse_zdock_scores(score_file, top_n=10, output_csv_path=output_csv)