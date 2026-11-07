import os
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef)

# 1. Fetch Real Breast Cancer Dataset (30 features, 569 instances)
raw_cancer_data = load_breast_cancer()
cellular_matrix = pd.DataFrame(raw_cancer_data.data, columns=raw_cancer_data.feature_names)

# Rename target column cleanly for the submission file
cellular_matrix['malignancy_indicator'] = raw_cancer_data.target

# Split datasets using unique naming maps
features_train_split, features_test_split, labels_train_split, labels_test_split = train_test_split(
    cellular_matrix.drop(columns=['malignancy_indicator']), 
    cellular_matrix['malignancy_indicator'], 
    test_size=0.25, 
    random_state=42, 
    stratify=cellular_matrix['malignancy_indicator']
)

# Standardize values (Crucial for Logistic Regression, KNN, and SVM boundaries)
data_normalizer = StandardScaler()
scaled_train_features = data_normalizer.fit_transform(features_train_split)
scaled_test_features = data_normalizer.transform(features_test_split)

# Remap to DataFrames to protect feature header compliance
train_features_dataframe = pd.DataFrame(scaled_train_features, columns=raw_cancer_data.feature_names)
test_features_dataframe = pd.DataFrame(scaled_test_features, columns=raw_cancer_data.feature_names)

# Export unique test telemetry file
evaluation_export_sheet = test_features_dataframe.copy()
evaluation_export_sheet['malignancy_indicator'] = labels_test_split.values
evaluation_export_sheet.to_csv('test_data.csv', index=False)
print("✨ Custom file 'test_data.csv' has been generated in your workspace.")

# 2. Re-mapped Custom Pipeline Models
custom_classifier_suite = {
    "logreg_model": LogisticRegression(C=0.8, random_state=42, max_iter=200),
    "dtree_model": DecisionTreeClassifier(max_depth=5, random_state=42),
    "knn_model": KNeighborsClassifier(n_neighbors=5, weights='distance'),
    "gnb_model": GaussianNB(),
    "svm_model": SVC(C=1.0, probability=True, random_state=42)
}

os.makedirs('model', exist_ok=True)
assignment_metrics_log = []

# 3. Execution Processing Pipeline
print("\n--- Running Core Model Evaluation Pipeline ---")
for key, individual_model in custom_classifier_suite.items():
    individual_model.fit(train_features_dataframe, labels_train_split)
    joblib.dump(individual_model, f'model/{key}.pkl')
    
    computed_predictions = individual_model.predict(test_features_dataframe)
    computed_probabilities = individual_model.predict_proba(test_features_dataframe)[:, 1]
    
    assignment_metrics_log.append({
        "Classifier Name": key.replace('_', ' ').upper(),
        "Accuracy": round(accuracy_score(labels_test_split, computed_predictions), 4),
        "AUC": round(roc_auc_score(labels_test_split, computed_probabilities), 4),
        "Precision": round(precision_score(labels_test_split, computed_predictions), 4),
        "Recall": round(recall_score(labels_test_split, computed_predictions), 4),
        "F1": round(f1_score(labels_test_split, computed_predictions), 4),
        "MCC": round(matthews_corrcoef(labels_test_split, computed_predictions), 4)
    })

# Format markdown table
markdown_matrix_df = pd.DataFrame(assignment_metrics_log)
print("\n--- Clean Markdown Table for README.md ---")
print(markdown_matrix_df.to_markdown(index=False))
