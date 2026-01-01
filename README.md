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

### 1. Extract, Transform, and Load (ETL) raw data

### 2. Data cleaning and validation

### 3. Train / validation / test split

### 4. Exploratory Data Analysis (EDA)

### 5. Feature engineering (scaling, PCA)

#### Dimensionality Reduction (PCA)

Principal Component Analysis (PCA) was applied after feature standardization to reduce the original 561-dimensional feature space.

Analysis of the cumulative explained variance showed that:

- Approximately **90% of variance** is retained with **65 components**
- Approximately **95% of variance** is retained with **104 components**

This enabled significant dimensionality reduction while preserving most of the information content, improving computational efficiency and reducing noise for downstream models.

### 6. Model selection and training

Multiple machine learning models were evaluated to identify the most suitable approach for Human Activity Recognition. These included linear models with dimensionality reduction, kernel-based methods, and ensemble algorithms.

The following models were trained and compared:

- Logistic Regression with PCA
- Linear Support Vector Classifier (LinearSVC) with PCA
- Random Forest
- Gradient Boosting

Models were trained on the standardized feature set, and performance was evaluated using accuracy and macro F1-score to ensure balanced multiclass performance.

### 7. Hyperparameter tuning

Hyperparameter tuning was performed using GridSearchCV with 5-fold cross-validation. The search focused on optimizing macro F1-score to ensure balanced performance across all activity classes.

For the SVC-based pipeline, the following hyperparameters were tuned:

- Number of PCA components
- Kernel type (linear, RBF)
- Regularization parameter C

The optimal configuration was selected based on cross-validated performance and used to train the final model.

Based on cross-validated performance and computational efficiency, a Support Vector Classifier (RBF kernel) with PCA was selected as the final model.

### 8. Model evaluation and validation

The final model was evaluated using accuracy and macro F1-score to account for class balance. The model achieved strong overall performance,
with particularly high accuracy for static activities such as _LAYING_.Most misclassifications occur between biomechanically similar activities (e.g., SITTING vs STANDING and among walking-related activities), which is expected due to overlapping motion patterns.

#### Accuracy & Macro F1

- Accuracy: **0.942**
- Macro F1-score: **0.941**

Macro F1-score was used to ensure balanced evaluation across all activity classes.

#### Confusion Matrix

The confusion matrix confirms strong diagonal dominance across all classes, indicating robust classification performance. Errors are
primarily concentrated among similar activity classes.

#### Class-Level Performance

- Static activities (LAYING, SITTING, STANDING) show near-perfect classification.
- Dynamic activities show minor confusion between walking-related classes.

#### Validation Conclusions

The model generalizes well to unseen data, with balanced performance across classes and no evidence of systematic misclassification or
overfitting.

### 9. Build a production-ready ML pipeline

The final model was packaged into a scikit-learn Pipeline to ensure reproducibility and prevent data leakage. The pipeline includes:

- Feature standardization
- Principal Component Analysis (PCA)
- Trained classification model

This modular pipeline enables consistent preprocessing during training and inference, making the solution suitable for deployment and future extensions.

### 10. Communicate results and conclusions

This project demonstrates a complete end-to-end machine learning workflow for Human Activity Recognition. Dimensionality reduction using PCA significantly reduced feature complexity while preserving predictive power.

The final model achieved strong and balanced performance across all activity classes, with particularly high accuracy for static activities and expected confusion among similar motion-based classes.

Future improvements could include real-time inference, deep learning approaches, or sensor fusion with additional data sources.

## Key Findings and Learnings

Through this project, I learned how to design and implement a complete end-to-end machine learning pipeline, from raw data ingestion to a production-ready model. Dimensionality reduction using PCA proved to be highly effective for managing high-dimensional sensor data while preserving predictive performance.

The results aligned with expectations: static activities such as LAYING were classified with very high accuracy, while confusion occurred primarily between biomechanically similar activities like SITTING and STANDING. This behavior indicates that the model learned meaningful patterns rather than overfitting noise.

Key takeaways include the importance of proper feature scaling, careful model selection using cross-validation, and the value of macro F1-score for evaluating balanced multiclass performance.

## Conclusion and Next Steps

This project demonstrates a complete, production-ready machine learning workflow for Human Activity Recognition using high-dimensional sensor data. While the final model achieved strong and balanced performance, several opportunities for improvement and extension remain.

One limitation of the current approach is the reliance on classical machine learning models and hand-engineered features. Future work could explore deep learning architectures such as convolutional or recurrent neural networks to better capture temporal dependencies in sensor signals. Additionally, more extensive hyperparameter tuning or alternative dimensionality reduction techniques could be evaluated.

In future applications, this pipeline could be adapted for real-time activity recognition on mobile or wearable devices, or extended to include additional sensor modalities such as gyroscope or GPS data. Further analysis could also examine subject-specific patterns or personalization strategies to improve performance across diverse users.

Access to larger or more diverse datasets could enable more robust generalization and support transfer learning approaches. Overall, this project provides a strong foundation that can be extended toward real-world deployment scenarios.
