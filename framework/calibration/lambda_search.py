from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

from calibration.risk_estimator import compute_empirical_risk
from calibration.set_constructor import construct_prediction_set_at_step

def search_valid_lambda_step0(
    df_model: pd.DataFrame,
    alpha: float = 0.1,
    metric: str = "recall",
    step_t: int = 0,
    lambda_grid: List[float] = None,
    max_pred_set_size: int = None,
    base_utility: float = 1.0
) -> Tuple[float, Dict[Tuple[str, int], List[int]], float]:
    if lambda_grid is None:
        lambda_grid = [round(x, 3) for x in np.linspace(1.0, 0.0, 101)]

    best_lambda = None
    best_risk = None
    best_prediction_sets = None

    print(f"\nλ⁰-Search | Step {step_t} | Target α = {alpha} | Metric = {metric} | BaseUtility = {base_utility}")
    print(f"{'λ':>8} | {'Risk':>8} | {'# Users':>9}")

    for lambda_val in lambda_grid:
        prediction_sets = construct_prediction_set_at_step(
            df_model,
            step_t,
            lambda_val,
            max_pred_set_size=max_pred_set_size
        )
        df_step = df_model[df_model["step"] == step_t]
        risk = compute_empirical_risk(df_step, prediction_sets, metric, base_utility=base_utility)
        n_users = len(df_step)

        print(f"{lambda_val:8.3f} | {risk:8.4f} | {n_users:9d}")

        if risk <= (1.0 - base_utility + alpha):
            best_lambda = lambda_val
            best_risk = risk
            best_prediction_sets = prediction_sets
            print(f"Selected λ = {lambda_val:.3f} with Risk = {risk:.4f}")
            break

    if best_lambda is None:
        best_lambda = 0.0
        best_risk = risk
        best_prediction_sets = prediction_sets
        print(f"No λ found satisfying risk ≤ target threshold. Defaulting to λ = 0.000 (Risk = {best_risk:.4f})")

    return best_lambda, best_prediction_sets, best_risk
