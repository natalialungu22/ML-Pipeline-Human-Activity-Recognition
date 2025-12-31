# ML-Pipeline-Human-Activity-Recognition

## Problem Statement

The goal of this project is to build a production-ready machine learning pipeline that predicts human physical activity (e.g. walking, sitting, standing) based on high-dimensional smartphone sensor data.

## Why this Problem

Human Activity Recognition is a real-world problem with applications in:

- health monitoring
- fitness tracking
- human–computer interaction

The dataset contains **561 numerical features**, making it suitable for **dimensionality reduction techniques such as PCA** and for demonstrating an end-to-end ML workflow.

## Dataset

- Source: UCI Machine Learning Repository

- Dataset: Human Activity Recognition Using Smartphones

- Size: ~10,000 observations, 561 features, 6 activity classes

## ML Task

- Type: **Multiclass Classification**

- Input: Smartphone sensor signals

- Output: Activity label

## Objectives

- Build a clean **ETL pipeline**

- Perform **feature engineering and PCA**

- Compare multiple ML models

- Tune hyperparameters using cross-validation

- Package everything into a **scikit-learn Pipeline**

- Communicate findings clearly

## Project Workflow

1. Extract, Transform, and Load (ETL) raw data

2. Data cleaning and validation

3. Train / validation / test split

4. Exploratory Data Analysis (EDA)

5. Feature engineering (scaling, PCA)

#### Dimensionality Reduction (PCA)

Principal Component Analysis (PCA) was applied after feature standardization to reduce the original 561-dimensional feature space.

Analysis of the cumulative explained variance showed that:

- Approximately **90% of variance** is retained with **65 components**
- Approximately **95% of variance** is retained with **104 components**

This enabled significant dimensionality reduction while preserving most of the information content, improving computational efficiency and reducing noise for downstream models.

6. Model selection and training

7. Hyperparameter tuning

8. Model evaluation and validation

9. Build a production-ready ML pipeline

10. Communicate results and conclusions
