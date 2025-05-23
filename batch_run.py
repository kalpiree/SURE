# import os
# import pandas as pd
# from tqdm import tqdm
# from data.loader import parse_list_column
# from calibration.lambda_search import search_valid_lambda_step0
# from calibration.risk_estimator import compute_empirical_risk
# from run_daur import run_daur

# # ----------------------------------------
# # CONFIGURATION
# # ----------------------------------------
# data_root = "datasets"
# processed_root = "processed"
# output_root = "outputs"

# # Customize your alpha sweep here
# alpha_values = [round(x, 2) for x in list(pd.np.arange(0.05, 0.55, 0.05))]  # 0.05 to 0.50

# # Metrics to evaluate
# metrics = ["recall", "ndcg", "mrr"]

# # Dataset name
# dataset = "my_dataset"  # 🔁 Change this to your dataset folder

# # Other parameters
# gamma = 2.0
# eta = 0.5

# # ----------------------------------------
# # MAIN LOOP
# # ----------------------------------------
# eval_records = []

# phase_files = sorted([
#     os.path.join(data_root, dataset, f)
#     for f in os.listdir(os.path.join(data_root, dataset)) if f.startswith("phase")
# ])

# model_files = sorted([
#     os.path.join(processed_root, dataset, f)
#     for f in os.listdir(os.path.join(processed_root, dataset)) if f.endswith(".csv")
# ])

# model_ids = list(range(len(model_files)))
# df_models = {}

# print("📦 Loading model files...")
# for idx, path in enumerate(model_files):
#     df = pd.read_csv(path)
#     df["candidate_items"] = df["candidate_items"].apply(parse_list_column)
#     df["normalized_scores"] = df["normalized_scores"].apply(parse_list_column)
#     df_models[idx] = df

# # Loop over all alpha values
# for alpha in tqdm(alpha_values, desc="Running DAUR for multiple alpha values"):
#     # 1. Init lambdas and weights
#     initial_lambdas = {}
#     initial_weights = {i: 1.0 / len(model_ids) for i in model_ids}

#     for i in model_ids:
#         lambda_0, _, _ = search_valid_lambda_step0(df_models[i], alpha=alpha, metric="recall")
#         initial_lambdas[i] = lambda_0 if lambda_0 is not None else 1.0

#     # 2. Run DAUR
#     output_dir = f"{output_root}_alpha_{str(alpha).replace('.', '')}"
#     lambdas, weights, ensemble_sets, model_risks = run_daur(
#         phase_files=phase_files,
#         model_ids=model_ids,
#         initial_lambdas=initial_lambdas,
#         initial_weights=initial_weights,
#         df_models=df_models,
#         alpha=alpha,
#         eta=eta,
#         gamma=gamma,
#         metric="recall",  # Only controls initial search
#         save_outputs=True,
#         output_dir=output_dir,
#         frozen_inference=True
#     )

#     # 3. Evaluate risk at final phase
#     last_phase = pd.read_csv(phase_files[-1])
#     last_step = last_phase["step"].max()
#     last_phase["candidate_items"] = last_phase["candidate_items"].apply(parse_list_column)
#     last_phase["normalized_scores"] = last_phase["normalized_scores"].apply(parse_list_column)

#     row = {"alpha": alpha}

#     for metric in metrics:
#         # Build (user_id, step) -> prediction_set dict from ensemble
#         ensemble_preds = {
#             k: v for k, v in ensemble_sets.items() if isinstance(v, dict)
#             for k, v in v.items() if k[1] == last_step
#         }
#         df_eval = last_phase[last_phase["step"] == last_step]
#         risk_val = compute_empirical_risk(df_eval, ensemble_preds, metric)
#         row[f"{metric}_risk"] = round(risk_val, 4)

#     eval_records.append(row)

# # Save master evaluation file
# results_df = pd.DataFrame(eval_records)
# results_df.to_csv(f"{output_root}/phase4_eval_metrics.csv", index=False)
# print("✅ Evaluation saved to phase4_eval_metrics.csv")


import os
import pandas as pd
from tqdm import tqdm
from calibration.loader import parse_list_column
from calibration.lambda_search import search_valid_lambda_step0
from calibration.risk_estimator import compute_empirical_risk
from run_daur import run_daur

# --------------------------
# CONFIGURATION
# --------------------------
DATA_ROOT = "datasets"
PROCESSED_ROOT = "processed"
OUTPUT_ROOT = "outputs"

ALPHA_VALUES = [round(x, 2) for x in list(pd.np.arange(0.05, 0.55, 0.05))]
METRICS = ["recall", "ndcg", "mrr"]

DATASET = "taobao"  # 🔁 Change as needed
GAMMA = 2.0
ETA = 0.5

# --------------------------
# MAIN
# --------------------------
phase_files = sorted([
    os.path.join(DATA_ROOT, DATASET, f)
    for f in os.listdir(os.path.join(DATA_ROOT, DATASET)) if f.startswith("phase")
])

model_files = sorted([
    os.path.join(PROCESSED_ROOT, DATASET, f)
    for f in os.listdir(os.path.join(PROCESSED_ROOT, DATASET)) if f.endswith(".csv")
])

model_ids = list(range(len(model_files)))
df_models = {}

print("📦 Loading model files...")
for idx, path in enumerate(model_files):
    df = pd.read_csv(path)
    df["candidate_items"] = df["candidate_items"].apply(parse_list_column)
    df["normalized_scores"] = df["normalized_scores"].apply(parse_list_column)
    df_models[idx] = df

# --------------------------
# RUN DAUR FOR EACH METRIC
# --------------------------
for metric in METRICS:
    print(f"\n🧠 Running for metric: {metric}")
    eval_records = []

    for alpha in tqdm(ALPHA_VALUES, desc=f"🔁 Alpha sweep for {metric}"):
        initial_lambdas = {}
        initial_weights = {i: 1.0 / len(model_ids) for i in model_ids}

        for i in model_ids:
            lambda_0, _, _ = search_valid_lambda_step0(df_models[i], alpha=alpha, metric=metric)
            initial_lambdas[i] = lambda_0 if lambda_0 is not None else 1.0

        output_dir = os.path.join(OUTPUT_ROOT, DATASET, metric, f"alpha_{int(alpha * 100):03d}")
        lambdas, weights, ensemble_sets, model_risks = run_daur(
            phase_files=phase_files,
            model_ids=model_ids,
            initial_lambdas=initial_lambdas,
            initial_weights=initial_weights,
            df_models=df_models,
            alpha=alpha,
            eta=ETA,
            gamma=GAMMA,
            metric=metric,
            save_outputs=True,
            output_dir=output_dir,
            frozen_inference=True
        )

        # Evaluate on final phase's final step
        last_phase = pd.read_csv(phase_files[-1])
        last_step = last_phase["step"].max()
        last_phase["candidate_items"] = last_phase["candidate_items"].apply(parse_list_column)
        last_phase["normalized_scores"] = last_phase["normalized_scores"].apply(parse_list_column)

        ensemble_preds = {
            k: v for k, v in ensemble_sets.items() if isinstance(v, dict)
            for k, v in v.items() if k[1] == last_step
        }

        df_eval = last_phase[last_phase["step"] == last_step]
        risk_val = compute_empirical_risk(df_eval, ensemble_preds, metric)

        eval_records.append({
            "metric": metric,
            "alpha": alpha,
            f"{metric}_risk": round(risk_val, 4)
        })

    # Save metric-specific evaluation
    metric_eval_path = os.path.join(OUTPUT_ROOT, DATASET, f"{metric}_eval.csv")
    pd.DataFrame(eval_records).to_csv(metric_eval_path, index=False)
    print(f"✅ {metric.upper()} evaluation saved to: {metric_eval_path}")
