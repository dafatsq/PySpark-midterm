# Midterm Project (UTS): Big Data & Machine Learning with PySpark MLlib

- **Name:** Maliki Kadafa Tsaqif
- **NIM:** 103012340185
- **Course:** Big Data and Artificial Intelligence
- **Evaluation:** Mid-Semester Project (*Proyek Tengah Semester* / UTS)
- **Framework:** Apache Spark (`PySpark 4.0.4` — `pyspark.sql`, `pyspark.ml.feature`, `pyspark.ml.classification`)

---

## Project Overview
This repository contains the end-to-end **Big Data and Machine Learning pipeline** built using **PySpark MLlib** on the official **NYC Taxi & Limousine Commission (NYC TLC) Yellow Taxi Trip Records 2023** dataset (`2023-01` through `2023-10`, totaling **3.19 GiB / 3.43 GB CSV** and **31,593,944 rows**).

### Repository Contents
- [`Midterm_Project_PySpark_Maliki_Kadafa_Tsaqif_103012340185.ipynb`](./Midterm_Project_PySpark_Maliki_Kadafa_Tsaqif_103012340185.ipynb): Complete Jupyter/Google Colab Notebook with full execution outputs covering all 7 rubric stages (Data Loading, EDA, Preprocessing Pipeline, Modeling, Evaluation, Hyperparameter Tuning, and Final Report).
- [`Laporan_Midterm_Project_PySpark_MLlib.pdf`](./Laporan_Midterm_Project_PySpark_MLlib.pdf): Final Midterm Project Report in PDF format.
- [`Laporan_Midterm_Project_PySpark_MLlib.md`](./Laporan_Midterm_Project_PySpark_MLlib.md): Markdown source of the Midterm Project Report.
- [`figures/`](./figures/): Visualization plots generated during Exploratory Data Analysis (EDA) and Model Evaluation.

---

## Summary of Model Performance (Test Set: 122,584 Rows)

| Model (PySpark MLlib) | Accuracy | Precision | Recall | F1-Score | AUC-ROC | AUC-PR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 0.8543 | 0.8783 | 0.8543 | 0.8411 | 0.8317 | 0.8756 |
| **Random Forest (Baseline)** | 0.8547 | 0.8805 | 0.8547 | 0.8411 | 0.8320 | 0.8805 |
| **GBT Classifier (Baseline)** | **0.8550** | 0.8805 | **0.8550** | **0.8415** | **0.8400** | **0.8877** |
| **Random Forest (Tuned 3-Fold CV)** | 0.8549 | **0.8806** | 0.8549 | 0.8414 | 0.8384 | 0.8854 |

> **Note on the >3 GB Dataset (`nyc_yellow_taxi_2023_3gb.csv`):**  
> Because GitHub enforces a 100 MB file limit, the 3.43 GB CSV file is excluded via `.gitignore`. Running **Section 1.1** in `Midterm_Project_PySpark_Maliki_Kadafa_Tsaqif_103012340185.ipynb` automatically downloads and streams all 10 months of NYC TLC 2023 data directly from the official NYC TLC CloudFront CDN into `nyc_yellow_taxi_2023_3gb.csv` (3.19 GiB / 3.43 GB).
