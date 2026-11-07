import os
import re
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# 1. SETUP DATASET (Replace this block with your actual data loading)
print("Generating dataset...")
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. DEFINE INITIALIZED MODELS
models = {
    "LOGREG MODEL": LogisticRegression(random_state=42),
    "DTREE MODEL": DecisionTreeClassifier(random_state=42),
    "KNN MODEL": KNeighborsClassifier(),
    "GNB MODEL": GaussianNB(),
    "SVM MODEL": SVC(probability=True, random_state=42)
}

# 3. TRAIN MODELS AND EVALUATE
print("Training models and calculating metrics...")
table_rows = []

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    
    table_rows.append({
        "Classifier Name": name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, prob),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, pred)
    })

# 4. CREATE DATAFRAME
metrics_df = pd.DataFrame(table_rows)
metrics_df = metrics_df.round(4)
markdown_table = metrics_df.to_markdown(index=False)

# 5. DYNAMIC ANALYSIS ENGINE (Automated Text Reporting)
# Identify top and bottom performers programmatically
best_model_idx = metrics_df['F1'].idxmax()
best_model_name = metrics_df.loc[best_model_idx, 'Classifier Name']
best_model_f1 = metrics_df.loc[best_model_idx, 'F1']
best_model_acc = metrics_df.loc[best_model_idx, 'Accuracy']

worst_model_idx = metrics_df['F1'].idxmin()
worst_model_name = metrics_df.loc[worst_model_idx, 'Classifier Name']
worst_model_f1 = metrics_df.loc[worst_model_idx, 'F1']

# Generate custom rule-based observations
insights = []
if best_model_name == "LOGREG MODEL":
    insights.append("- **Linear Separability:** The high performance of the Logistic Regression model implies that the dataset features exhibit strong linear relationships.")
elif best_model_name == "SVM MODEL":
    insights.append("- **Decision Margin:** The SVM classifier succeeded by maximizing the geometric margin, effectively separating complex decision boundaries.")
elif best_model_name == "KNN MODEL":
    insights.append("- **Local Clustering:** The KNN victory indicates that the data points form tight, distinct spatial clusters based on similarity metrics.")

# Check for specific metric imbalances (e.g., Precision vs Recall gaps)
for index, row in metrics_df.iterrows():
    if abs(row['Precision'] - row['Recall']) > 0.05:
        insights.append(f"- **Trade-off Alert:** {row['Classifier Name']} exhibits a notable gap between Precision ({row['Precision']}) and Recall ({row['Recall']}), indicating potential class prediction imbalances.")

# Compile the final Automated Narrative string
analysis_report = (
    f"### 📑 Automated Performance Analysis Report\n\n"
    f"**Key Findings:**\n"
    f"- **Top Performer:** The **{best_model_name}** achieved the highest overall performance with an **F1-Score of {best_model_f1}** and an **Accuracy of {best_model_acc}**.\n"
    f"- **Underperformer:** The **{worst_model_name}** struggled relative to the group, logging the lowest baseline benchmark with an **F1-Score of {worst_model_f1}**.\n\n"
    f"**Data Observations:**\n"
    + "\n".join(insights) + "\n"
)

# 6. ENCAPSULATE COMPLETE RESULTS BLOCK
header_text = "### 📊 Model Evaluation Performance"
complete_results_block = f"{header_text}\n\n{markdown_table}\n\n{analysis_report}"

# 7. AUTOMATICALLY WRITE TO THE FILE
readme_path = "README.md"

if os.path.exists(readme_path):
    print(f"Updating data and text report inside existing {readme_path}...")
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Regex target block scanning pattern
    pattern = rf"{re.escape(header_text)}[\s\S]*?(?=\n## |\Z)"
    
    if re.search(pattern, content):
        updated_content = re.sub(pattern, complete_results_block.strip(), content)
    else:
        updated_content = content.rstrip() + "\n\n" + complete_results_block
        
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(updated_content)
    print("Successfully refreshed results and analytical text!")

else:
    print(f"Creating a fresh {readme_path} incorporating the live metrics narrative...")
    initial_template = (
        f"# Machine Learning Production Pipeline\n\n"
        f"This repository hosts an automated benchmarking setup evaluating multiple model variants.\n\n"
        f"## 📈 Results & Deep-Dive Analysis\n\n"
        f"{complete_results_block}"
    )
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(initial_template)
    print(f"Successfully generated new {readme_path} with automated narrative!")
