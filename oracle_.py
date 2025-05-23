# save this as evaluate_oracle.py

import os
import pandas as pd
import numpy as np
import argparse

def compute_metrics_for_file(file_path: str, k: int = 25):
    df = pd.read_csv(file_path)

    recalls = []
    mrrs = []
    ndcgs = []

    for idx, row in df.iterrows():
        candidates = eval(row['candidate_items'])
        scores = eval(row['scores'])
        true_item = row['true_item']

        # Sort candidates by descending score
        sorted_items = [x for _, x in sorted(zip(scores, candidates), key=lambda x: -x[0])]

        try:
            rank = sorted_items.index(true_item)
        except ValueError:
            rank = None

        if rank is not None:
            if rank < k:
                recalls.append(1.0)
                mrrs.append(1.0 / (rank + 1))
                ndcgs.append(1.0 / np.log2(rank + 2))
            else:
                recalls.append(0.0)
                mrrs.append(0.0)
                ndcgs.append(0.0)
        else:
            recalls.append(0.0)
            mrrs.append(0.0)
            ndcgs.append(0.0)

    return {
        f'recall@{k}': np.mean(recalls),
        f'mrr@{k}': np.mean(mrrs),
        f'ndcg@{k}': np.mean(ndcgs)
    }

def evaluate_all_models(base_dir: str, k: int = 25):
    results = []
    for model_folder in sorted(os.listdir(base_dir)):
        model_path = os.path.join(base_dir, model_folder)
        if not os.path.isdir(model_path):
            continue

        eval_file = os.path.join(model_path, "phase0_eval_output.csv")
        if os.path.exists(eval_file):
            print(f"🔎 Evaluating {model_folder}...")
            metrics = compute_metrics_for_file(eval_file, k=k)
            metrics['model'] = model_folder
            results.append(metrics)
        else:
            print(f"⚠️  phase0_eval_output.csv missing for {model_folder} — skipping.")

    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True,
                        help="Base directory containing model folders")
    parser.add_argument("--k", type=int, default=25,
                        help="Value of k for recall@k, mrr@k, ndcg@k")
    parser.add_argument("--output_file", type=str, default="oracle_val.csv",
                        help="Output CSV file name")
    args = parser.parse_args()

    print(f"\n🚀 Running Oracle Evaluation with k={args.k}...\n")
    metrics_df = evaluate_all_models(args.base_dir, k=args.k)

    output_path = os.path.join(args.base_dir, args.output_file)
    metrics_df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")

if __name__ == "__main__":
    main()