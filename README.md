

# DUAR Framework: Dynamically Adaptive Uncertainty-aware Recommendations

This repository implements a pipeline to generate prediction sets with guaranteed performance with the DUAR framework.

---

## 🔁 Pipeline Overview

### 1. **Data Preprocessing**
Generates phase-wise training and evaluation data.

- **Input:** raw user–item interaction logs  
- **Output:**  
  ```
  processed_datasets/{dataset}/{subdataset}/phased_data/model_{id}/
      ├── train_phase{i}.txt
      └── eval_phase{i}.csv
  ```

Each `train_phase{i}.txt` is a chronological split, and `eval_phase{i}.csv` includes per-user candidate sets and the true item.

---

### 2. **Model Training & Inference**
Train any sequential recommender (e.g., SASRec, FMLP-Rec) on each phase.

- **Input:** preprocessed data  
- **Output:** CSV files with model predictions:
  ```
  datasets_/{dataset}/{subdataset}/model_{id}/phase{i}_eval_output.csv
  ```

Each CSV must contain:
- `user_idx`
- `step`
- `true_item`
- `candidate_items` (as list)
- `scores` (matching candidate_items)
- `loss` (optional: for calibration)

---

### 3. **DUAR Calibration & Aggregation**
Use DUAR to calibrate models and adaptively ensemble them over time.

#### 🧹 Normalize Scores
```bash
python data/loader.py --dataset sasrec --subdataset goodreads --output_root datasets_
```

#### 🚀 Run DUAR
```bash
bash duar.sh
```

Or manually:
```bash
python main_.py \
  --dataset sasrec \
  --subdataset goodreads \
  --alphas 0.1 \
  --etas 0.5 \
  --gamma 2.0 \
  --base_utilities recall=0.67 \
  --output_dir outputs \
  --freeze_inference \
  --max_pred_set_size 50
```

- **Output:** Ensemble prediction stats and diagnostics:
  ```
  outputs/{dataset}/{subdataset}/alpha_{xxx}/detailed_snapshots.csv
  ```

---

## 🧠 DUAR Components

- `main_.py`: Entry point for DUAR. Runs lambda calibration + adaptive ensemble.
- `run_daur_.py`: Implements the time-evolving risk-driven aggregation loop.
- `calibration/`: Core calibration logic:
  - `adaptive_loop.py`: Adaptive λ update based on segment-level risk
  - `aggregator_.py`: Model weighting and prediction set union
  - `lambda_search_.py`: Risk-controlled lambda search
  - `risk_estimator.py`: Empirical loss computation
  - `segment_shift.py`: Segment selection via concept/covariate shift
- `data/loader.py`: Normalizes model outputs into DUAR-ready format

---

## 📂 Folder Structure

```
.
├── data/
│   └── loader.py
├── calibration/
│   ├── adaptive_loop.py
│   ├── aggregator_.py
│   ├── lambda_search_.py
│   ├── risk_estimator.py
│   ├── segment_shift.py
│   └── set_constructor_.py
├── main_.py
├── run_daur_.py
├── duar.sh
├── datasets_/           # Normalized model predictions
└── outputs/             # DUAR outputs
```

---

