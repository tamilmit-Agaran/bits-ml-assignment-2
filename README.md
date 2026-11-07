## a. Problem Statement
This workspace builds an integrated computational framework to accurately separate malignant and benign cell samples using observed clinical dimensions. By fitting five structurally distinct statistical boundaries, this project builds high-dimensional optimization models to assist with medical diagnostic validation without requiring training loops during inference.

## b. Dataset Description
- **Dataset Focus Source:** Breast Cancer Wisconsin Fine Needle Aspirate (FNA) Matrix.
- **Data Volume Count:** 569 registered tissue rows (Satisfies requirement of >= 500 instances).
- **Attribute Dimension:** 30 unique structural feature columns (Satisfies requirement of >= 12 attributes).
- **Target Definition Vector:** `malignancy_indicator` (Binary separation mapping: 0 = Malignant, 1 = Benign).
- **Key Attributes Evaluated:** Nuclear radius, perimeter dimensions, texture variance values, tissue compactness indices, concavity depths, and fractal dimensions.

## c. GitHub Repository Link
- **Repository URL:** [Insert Your GitHub Repo URL Here]
- **Streamlit Live Link:** [Insert Your Live Streamlit Application Link Here]

## d. Models Used & Performance Evaluation

### Comparison Table

| Classifier Name   |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| LOGREG MODEL      |     0.9790 | 0.9975 |      0.9778 |   0.9888 | 0.9832 | 0.9554 |
| DTREE MODEL       |     0.9371 | 0.9419 |      0.9545 |   0.9438 | 0.9492 | 0.8665 |
| KNN MODEL         |     0.9650 | 0.9922 |      0.9565 |   0.9888 | 0.9724 | 0.9259 |
| GNB MODEL         |     0.9371 | 0.9859 |      0.9355 |   0.9663 | 0.9506 | 0.8659 |
| SVM MODEL         |     0.9720 | 0.9964 |      0.9670 |   0.9888 | 0.9778 | 0.9405 |

### Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **LOGREG MODEL** | Outstanding boundary linear convergence profile. Scaled attributes lead to clean optimization convergence with minimal variance. |
| **DTREE MODEL** | Isolates coarse high-variance nodes quickly. Shows small structural drop-offs compared to continuous boundary algorithms. |
| **KNN MODEL** | Highly reliable sample neighborhood distance matching. Distance weighting handles point spaces smoothly. |
| **GNB MODEL** | Robust and fast baseline tracking. Suffers from a tiny baseline penalty due to statistical feature correlation. |
| **SVM MODEL** | Excellent margin separator. Securely maps complex feature vectors without overfitting parameters. |
| **Overall Winner?** | **LOGREG MODEL** serves as the optimal framework deployment match, showing top-tier performance balances across F1 (0.9832) and MCC metrics (0.9554). |
