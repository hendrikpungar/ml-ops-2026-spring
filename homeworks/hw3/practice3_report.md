# Practice 3

## 1. Steps performed in class

In class, we followed an end-to-end MLOps workflow starting from project setup and version control. We created a GitHub repository, cloned it locally, organized the solution into pipeline components inside the src structure, and prepared clean repository hygiene by excluding generated outputs such as data, model artifacts, reports, and logs through .gitignore. After that, we implemented the pipeline as separate scripts for data ingestion, preprocessing, feature engineering, model building, and model evaluation, with the main implementation snippets documented in codes.txt.

Once the modular code was ready, we automated execution using DVC by initializing the project and defining the full stage graph in dvc.yaml with dependencies, outputs, and metrics. We then reproduced the pipeline to verify that stage ordering and data flow were correct. To make the workflow configurable, we added params.yaml and updated stage scripts to read parameter values such as split ratio, feature limits, and model hyperparameters. We also practiced experiment tracking by integrating DVCLive, logging both metrics and parameters across runs, and comparing repeated experiments through DVC experiment commands. Finally, we reviewed the remote storage workflow for syncing DVC-tracked artifacts to S3-compatible backends as part of production-ready MLOps practice.

## 2. What I learned in these lectures

These lectures helped me understand that MLOps is not only about training a model but about building a reliable system around the model. Breaking work into clear pipeline stages makes projects easier to reproduce, debug, and maintain than keeping everything in a single notebook. I learned how DVC provides reproducibility by explicitly tracking stage dependencies and outputs, and how parameterization in params.yaml improves experimentation by removing hardcoded values from scripts. I also learned why disciplined experiment tracking is important: with DVCLive and DVC experiments, each run can be compared objectively using logged metrics and parameters. Overall, the key takeaway is that production-quality ML requires consistent data workflows, measurable model performance, and controlled versioning of data, code, and artifacts.

## 3. Best experiment

Based on my recorded results , the best experiment is version 3 of homework 2, where the model was retrained on the combined January and February 2021 data. This run used trip_distance, passenger_count, and PULocationID as features to predict total_amount. The evaluation results were R2 = 0.8910372009712021, MAE = 3.1165747676848325, and RMSE = 5.085759349278877. I consider this the best experiment because it achieved the highest R2 among the compared versions and also produced a slightly lower RMSE than version 2, indicating a marginally better overall fit on the updated data distribution.

## 4. GitHub repository link

My GitHub repository for this course work is available at https://github.com/hendrikpungar/ml-ops-2026-spring.

