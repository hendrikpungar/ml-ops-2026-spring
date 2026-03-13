# Practice 2.2 Report (Regression Branch)

## Repository
- GitHub: https://github.com/hendrikpungar/ml-ops-2026-spring
- Branch used for this practice: `regression-model`

## Versioning Summary

### Version 1
- Data: January 2021 (`green_tripdata_2021-01.parquet`)
- Model: Existing Practice 1 model (`regression_model_v1.joblib`)
- Code: Practice 1 training code

### Version 2
- Change type: Data/evaluation update only
- Data: January + February 2021
- Model: Reused Version 1 model (no retraining)
- Code: Same model, new split and evaluation logic
- Features: `trip_distance`, `passenger_count`, `PULocationID`
- Metrics: R2 = 0.89085, MAE = 3.06158, RMSE = 5.09013

### Version 3
- Change type: Model update (retrain) with same Version 2 data
- Data: January + February 2021 (same as V2)
- Model: Retrained linear regression (`regression_model_v3.joblib`)
- Code: Same model family, retrained on new training set
- Features: `trip_distance`, `passenger_count`, `PULocationID`
- Metrics: R2 = 0.89104, MAE = 3.11657, RMSE = 5.08576

## What Is Happening in V2 Results
When the old model (trained on January-only patterns) is tested on a split from the combined January+February data, performance can shift due to data distribution change between months. This is expected: seasonal, operational, or behavior differences in February can cause mild performance drift.

## V2 -> V3 Result Interpretation
Retraining on the combined training set lets the model learn from the newer month as well, which usually improves or stabilizes generalization on the combined test set. Any gain in R2 and decrease in MAE/RMSE indicates the model has adapted to the updated data distribution.

In this run, R2 increased slightly (+0.00019) and RMSE improved slightly (-0.00437), while MAE increased (+0.05499). This indicates only marginal overall change: retraining helped a little on squared-error behavior but did not improve absolute error.

## Monitoring Plan for Production (Version 3)

### 1) Data-related metric
- Metric: Population Stability Index (PSI) for `trip_distance` (or distribution drift using KS test)
- Threshold idea: PSI > 0.2 indicates moderate shift, PSI > 0.3 severe shift

### 2) Model-performance metric
- Metric: Rolling MAE (and RMSE) on labeled feedback data
- Threshold idea: Retrain if MAE worsens by >10% for 2+ consecutive windows

### 3) System metric
- Metric: Prediction latency p95
- Threshold idea: Investigate if p95 exceeds service SLO (for example, >200 ms)

## Retrain vs Rollback Policy
- Retrain when: data drift is sustained and performance degradation is gradual but recoverable.
- Roll back when: sudden severe performance collapse, data pipeline corruption, or inference/system instability after deployment.

## DVC Notes
- Data and model artifacts are tracked with DVC metadata (`.dvc` files), not committed directly to git as raw artifacts.
- Configure your Google Drive remote with:
  ```powershell
  c:/Users/hendr/Desktop/andmeteadus/mlops/.venv/Scripts/python.exe -m dvc remote add -d storage gdrive://<YOUR_FOLDER_ID>
  c:/Users/hendr/Desktop/andmeteadus/mlops/.venv/Scripts/python.exe -m dvc push
  ```
