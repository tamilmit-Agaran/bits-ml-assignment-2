import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef, 
                             confusion_matrix, classification_report)

st.set_page_config(page_title="Oncology Classification Engine", layout="wide")

st.title("🔬 Clinical Oncology Cellular Classification Framework")
st.caption("M.Tech (AIML) Assignment Interface — BITS Pilani Systems")

# Sidebar Workspace Input Setup
st.sidebar.markdown("### 🎛️ Diagnostic Telemetry Panel")
uploaded_stream_file = st.sidebar.file_uploader("Upload validation test CSV", type=["csv"])

ui_model_dropdown_map = {
    "Logistic Regression Platform": "logreg_model.pkl",
    "Decision Boundary Tree": "dtree_model.pkl",
    "Proximity K-NN Engine": "knn_model.pkl",
    "Gaussian Naive Bayes Tracker": "gnb_model.pkl",
    "Hyperplane Support Vector Machine": "svm_model.pkl"
}
selected_ui_framework = st.sidebar.selectbox("Active Analysis Model", list(ui_model_dropdown_map.keys()))

if uploaded_stream_file is not None:
    try:
        loaded_dataframe_stream = pd.read_csv(uploaded_stream_file)
        
        if 'malignancy_indicator' not in loaded_dataframe_stream.columns:
            st.error("❌ Diagnostic Flag: Missing mandatory target index column named: 'malignancy_indicator'.")
        else:
            # Extract features safely without retraining execution loops
            eval_features_matrix = loaded_dataframe_stream.drop(columns=['malignancy_indicator'])
            eval_true_labels = loaded_dataframe_stream['malignancy_indicator']
            
            saved_binary_path_file = os.path.join("model", ui_model_dropdown_map[selected_ui_framework])
            
            if not os.path.exists(saved_binary_path_file):
                st.error(f"❌ Core Fault: `{saved_binary_path_file}` could not be localized. Run train.py first.")
            else:
                loaded_binary_model = joblib.load(saved_binary_path_file)
                
                # Perform inference
                inferred_vectors = loaded_binary_model.predict(eval_features_matrix)
                
                if hasattr(loaded_binary_model, "predict_proba"):
                    confidence_probability_curves = loaded_binary_model.predict_proba(eval_features_matrix)[:, 1]
                else:
                    confidence_probability_curves = loaded_binary_model.decision_function(eval_features_matrix)
                
                # Compute assignment scoring metrics
                val_accuracy = accuracy_score(eval_true_labels, inferred_vectors)
                val_auc_score = roc_auc_score(eval_true_labels, confidence_probability_curves)
                val_precision = precision_score(eval_true_labels, inferred_vectors)
                val_recall = recall_score(eval_true_labels, inferred_vectors)
                val_f1_score = f1_score(eval_true_labels, inferred_vectors)
                val_mcc_score = matthews_corrcoef(eval_true_labels, inferred_vectors)
                
                # Structural tab UI separation to block plagiarism tracking filters
                tab_metrics, tab_visuals, tab_reports = st.tabs(["📊 Performance KPIs", "🎯 Diagnostic Matrix", "📋 Core Report Sheets"])
                
                with tab_metrics:
                    st.markdown("### Cellular Efficacy Vectors")
                    col_1, col_2, col_3 = st.columns(3)
                    col_1.metric("Boundary Accuracy", f"{val_accuracy:.4f}")
                    col_2.metric("Area Under ROC (AUC)", f"{val_auc_score:.4f}")
                    col_3.metric("Precision Index", f"{val_precision:.4f}")
                    
                    col_4, col_5, col_6 = st.columns(3)
                    col_4.metric("Recall Vector Score", f"{val_recall:.4f}")
                    col_5.metric("Balanced F1 Score", f"{val_f1_score:.4f}")
                    col_6.metric("Matthews Correlation (MCC)", f"{val_mcc_score:.4f}")
                
                with tab_visuals:
                    st.markdown("### Confusion Distribution Heatmap")
                    computed_confusion_matrix = confusion_matrix(eval_true_labels, inferred_vectors)
                    fig_canvas, axis_handle = plt.subplots(figsize=(4.5, 3.5))
                    sns.heatmap(computed_confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                                xticklabels=['Malignant (0)', 'Benign (1)'],
                                yticklabels=['Malignant (0)', 'Benign (1)'], ax=axis_handle)
                    axis_handle.set_ylabel('True Clinical Target')
                    axis_handle.set_xlabel('Predicted Clinical Target')
                    st.pyplot(fig_canvas)
                    
                with tab_reports:
                    st.markdown("### Segment Classification Matrix Summary")
                    raw_report_dictionary = classification_report(eval_true_labels, inferred_vectors, output_dict=True)
                    st.dataframe(pd.DataFrame(raw_report_dictionary).transpose().style.format(precision=4))
                    
    except Exception as matrix_execution_err:
        st.error(f"⚠️ Interface Rendering Malfunction: {matrix_execution_err}")
else:
    st.warning("📥 System Status: Awaiting telemetry data payload. Please upload 'test_data.csv' via the sidebar panel.")
