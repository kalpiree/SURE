import os
import argparse
import pandas as pd
from data.loader import parse_list_column
from calibration.lambda_search import search_valid_lambda_step0
from run_daur import run_daur

def main():
    parser = argparse.ArgumentParser(description="Run DAUR on a recommender + dataset")

    parser.add_argument("--dataset", type=str, required=True,
                        help="Model type folder under ./datasets_ (e.g., 'fmlp-rec')")
    parser.add_argument("--subdataset", type=str, required=True,
                        help="Subdataset folder inside dataset (e.g., 'goodreads')")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--metric", type=str, choices=["recall", "ndcg", "mrr"], default="recall")
    parser.add_argument("--freeze_inference", action="store_true")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--max_pred_set_size", type=int, default=None,
                    help="Optional maximum prediction set size per user (default: no limit)")
    parser.add_argument("--base_utility", type=float, default=0.67,
         help="Base utility value used for computing losses (default: 1.0)")

    args = parser.parse_args()

    base_dir = os.path.join("datasets_", args.dataset, args.subdataset)
    model_folders = sorted([f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))])
    output_base = os.path.join(args.output_dir, args.dataset, args.subdataset, f"alpha_{int(args.alpha * 100):03d}")
    os.makedirs(output_base, exist_ok=True)

    df_models = {}
    initial_lambdas = {}
    model_ids = list(range(len(model_folders)))

    print(f"Loading {len(model_folders)} models and calibrating λ⁰...\n")

    for model_id, model_folder in enumerate(model_folders):
        full_path = os.path.join(base_dir, model_folder)
        print(f"   🔁 Model {model_id}: {model_folder}")

        df_all = []
        for phase_idx in range(5):
            phase_file = os.path.join(full_path, f"phase{phase_idx}_eval_output.csv")
            df_phase = pd.read_csv(phase_file)
            df_phase["step"] = df_phase["step"] + phase_idx * 10
            df_all.append(df_phase)

        df_model = pd.concat(df_all).reset_index(drop=True)
        df_model["candidate_items"] = df_model["candidate_items"].apply(parse_list_column)
        df_model["scores"] = df_model["scores"].apply(parse_list_column)
        df_model["normalized_scores"] = df_model["scores"]  # already normalized!

        # Calibrate lambda⁰ for step 0
        # lambda_0, _, _ = search_valid_lambda_step0(df_model, alpha=args.alpha, metric=args.metric, step_t=0)
        lambda_0, _, _ = search_valid_lambda_step0(
            df_model,
            alpha=args.alpha,
            metric=args.metric,
            step_t=0,
            max_pred_set_size=args.max_pred_set_size,
            base_utility=args.base_utility  # ✅ ADD THIS
        )
        lambda_0 = lambda_0 if lambda_0 is not None else 1.0
        initial_lambdas[model_id] = lambda_0
        df_models[model_id] = df_model

        # Optional: Save model data
        model_output_dir = os.path.join(output_base, model_folder)
        os.makedirs(model_output_dir, exist_ok=True)
        df_model.to_csv(os.path.join(model_output_dir, f"{model_folder}_data.csv"), index=False)

    initial_weights = {model_id: 1.0 / len(model_ids) for model_id in model_ids}

    print("\nRunning DAUR pipeline...\n")
    run_daur(
        phase_files=[],  # use preloaded model DataFrames
        model_ids=model_ids,
        initial_lambdas=initial_lambdas,
        initial_weights=initial_weights,
        df_models=df_models,
        alpha=args.alpha,
        eta=args.eta,
        gamma=args.gamma,
        metric=args.metric,
        save_outputs=True,
        output_dir=output_base,
        frozen_inference=args.freeze_inference,
        max_pred_set_size=args.max_pred_set_size,
        base_utility=args.base_utility  
    )

    print(f"\n All models completed. Outputs saved to: {output_base}")

if __name__ == "__main__":
    main()
    
    
# python main.py \
#   --dataset fmlp-rec \
#   --subdataset goodreads \
#   --alpha 0.1 \
#   --eta 0.5 \
#   --gamma 2.0 \
#   --metric recall \
#   --freeze_inference

# python main.py   --dataset fmlp_runs   --subdataset goodreads   --alpha 0.1   --eta 0.5   --gamma 2.0   --metric recall   --freeze_inference --max_pred_set_size 25 --base_utility 0.7