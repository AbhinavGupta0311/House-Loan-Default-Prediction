# 🏠 House Loan Default Prediction using Deep Learning (TensorFlow)

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Sequential_DNN-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Preprocessing_&_Metrics-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-4B8BBE)](https://imbalanced-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen)]()

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Business Problem](#-business-problem)
3. [Dataset Information](#-dataset-information)
4. [Tech Stack](#️-tech-stack)
5. [Project Architecture](#-project-architecture)
6. [Workflow](#-project-workflow)
7. [EDA Process](#-exploratory-data-analysis-eda)
8. [Model Architecture](#-model-architecture)
9. [Key Insights](#-key-insights)
10. [Results](#-model-performance--results)
11. [Business Interpretation](#-business-interpretation)
12. [Visualizations](#-visualizations)
13. [Installation Instructions](#️-installation--setup)
14. [Usage Instructions](#-usage)
15. [Folder Structure](#-repository-structure)
16. [Future Improvements](#-future-improvements)
17. [Author](#-author)

---

## 📌 Project Overview

Financial institutions face significant risks due to loan defaults. Predicting whether a loan applicant is likely to default helps banks minimize financial losses and make better, data-driven lending decisions.

This project develops a **Deep Neural Network (DNN)** using **TensorFlow** and **Keras** to predict loan defaults based on historical customer data. The model addresses real-world challenges including:

- 📐 **High Dimensionality** — 122 features across 307,511 records
- 🔍 **Missing Values** — handled with targeted imputation strategies
- ⚖️ **Severe Class Imbalance** — ~8.1% default rate addressed using SMOTE
- 🏦 **Business-Driven Evaluation** — Recall-focused due to financial risk asymmetry

---

## 💼 Business Problem

Banks and financial institutions approve thousands of loan applications every day. Incorrectly approving a loan for a customer who later defaults leads to:

- 💸 **Direct financial losses**
- 📉 **Increased non-performing assets (NPAs)**
- ⚠️ **Elevated credit risk exposure**
- 🔻 **Regulatory pressure and capital adequacy concerns**

**The challenge:** The dataset is heavily imbalanced — only ~8% of applicants actually default — making it difficult for conventional models to correctly identify high-risk applicants.

**Our solution:** A Deep Neural Network trained on SMOTE-balanced data, optimized for **Recall (Sensitivity)** to minimize missed defaults (False Negatives), which carry a far higher business cost than False Positives.

---

## 📂 Dataset Information

| Property        | Details                              |
|-----------------|--------------------------------------|
| **File**        | `loan_data.csv`                      |
| **Source**      | Home Credit Default Risk (Kaggle-style) |
| **Rows**        | 307,511                              |
| **Columns**     | 122                                  |
| **Target**      | `TARGET` (Binary)                    |
| **Class 0**     | Loan Repaid (~91.9%)                 |
| **Class 1**     | Loan Default (~8.1%)                 |
| **Size**        | ~166 MB                              |
| **Train Split** | 80%                                  |
| **Test Split**  | 20%                                  |

### 🔑 Key Feature Groups

| Feature Group               | Examples                                                  |
|-----------------------------|-----------------------------------------------------------|
| **Demographics**            | `CODE_GENDER`, `CNT_CHILDREN`, `NAME_FAMILY_STATUS`       |
| **Financial**               | `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`           |
| **Employment**              | `DAYS_EMPLOYED`, `OCCUPATION_TYPE`, `NAME_INCOME_TYPE`    |
| **Housing**                 | `NAME_HOUSING_TYPE`, `FLAG_OWN_REALTY`, `TOTALAREA_MODE`  |
| **External Scores**         | `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`            |
| **Social Circle Defaults**  | `OBS_30_CNT_SOCIAL_CIRCLE`, `DEF_30_CNT_SOCIAL_CIRCLE`    |
| **Document Flags**          | `FLAG_DOCUMENT_2` through `FLAG_DOCUMENT_21`              |
| **Credit Bureau Inquiries** | `AMT_REQ_CREDIT_BUREAU_HOUR` to `AMT_REQ_CREDIT_BUREAU_YEAR` |

> ⚠️ **Class Imbalance Note:** The dataset exhibits ~8.1% defaults vs ~91.9% repayments. This was addressed using **SMOTE** (Synthetic Minority Over-sampling Technique) applied exclusively to the **training set** to prevent data leakage.

---

## 🛠️ Tech Stack

| Category           | Library / Tool         | Purpose                                      |
|--------------------|------------------------|----------------------------------------------|
| **Language**       | Python 3.x             | Core programming language                    |
| **Data**           | `pandas`               | Data manipulation, loading, EDA              |
| **Numerics**       | `numpy`                | Array operations, seed control               |
| **Visualization**  | `matplotlib`           | Training curves, ROC, PR curves              |
| **Visualization**  | `seaborn`              | Confusion matrix heatmap, distributions      |
| **ML Preprocessing** | `scikit-learn`       | Train-test split, LabelEncoder, StandardScaler |
| **ML Metrics**     | `scikit-learn`         | Accuracy, Precision, Recall, F1, ROC-AUC     |
| **Imbalance**      | `imbalanced-learn`     | SMOTE oversampling                           |
| **Deep Learning**  | `TensorFlow 2.x`       | Model definition, training, evaluation       |
| **DL API**         | `Keras (Sequential)`   | Layer stacking, callbacks, model saving      |
| **Environment**    | Jupyter Notebook       | Interactive development and visualization    |

---

## 🏗️ Project Architecture

```
Raw Data (loan_data.csv)
        │
        ▼
┌─────────────────────┐
│  Data Loading &     │
│  Initial Exploration│
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Missing Value      │
│  Analysis &         │
│  Imputation         │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Exploratory Data   │
│  Analysis (EDA)     │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Preprocessing      │
│  Label Encoding     │
│  Standard Scaling   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Class Imbalance    │
│  Handling (SMOTE)   │
│  (Train set only)   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  DNN Model          │
│  Dense → BN →       │
│  Dropout × 4 layers │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Training           │
│  EarlyStopping +    │
│  ReduceLROnPlateau  │
│  + ModelCheckpoint  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Evaluation         │
│  Threshold Tuning   │
│  Business Insights  │
└─────────────────────┘
```

---

## 🔄 Project Workflow

| Step | Task                                | Details                                                      |
|------|-------------------------------------|--------------------------------------------------------------|
| 1    | Data Loading                        | Load `loan_data.csv`, inspect shape (307511 × 122)           |
| 2    | Data Understanding                  | dtypes, unique values, class distribution analysis           |
| 3    | Missing Value Analysis              | Column-wise missing %, imputation strategy per feature type  |
| 4    | Exploratory Data Analysis           | Target distribution, feature correlations, key visualizations|
| 5    | Data Preprocessing                  | Label encoding (categoricals), StandardScaler (numericals)   |
| 6    | Class Imbalance Handling            | SMOTE applied to training set only                           |
| 7    | Feature Engineering                 | Derived features, irrelevant feature removal                 |
| 8    | Model Development                   | 4-layer DNN: Dense → BatchNorm → Dropout stack               |
| 9    | Model Training                      | Adam optimizer, Binary Crossentropy, callbacks               |
| 10   | Evaluation & Business Insights      | Metrics, threshold tuning, FP/FN impact analysis             |

---

## 🔍 Exploratory Data Analysis (EDA)

The EDA phase uncovered several important patterns in the data:

### Target Distribution
```
Class 0 (Repaid)  : ~282,686 records  (91.9%)
Class 1 (Default) :  ~24,825 records   (8.1%)
```
→ Severe imbalance confirms the need for SMOTE.

### Key EDA Findings

| Finding | Observation |
|---------|-------------|
| **External Scores** | `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3` are strong negative predictors of default |
| **Employment Days** | Negative `DAYS_EMPLOYED` values (encoding) required special treatment |
| **Missing Values** | Several building characteristic columns had >60% missing — handled with median/mode imputation |
| **Income & Credit** | Higher credit-to-income ratios showed slight correlation with default risk |
| **Document Flags** | Most document flags had near-zero variance and minimal predictive power |
| **Social Circle** | Customers whose social circle had prior defaults showed higher default probability |

### Missing Value Strategy

| Column Type           | Imputation Method  |
|-----------------------|--------------------|
| Numerical features    | Median imputation  |
| Categorical features  | Mode imputation    |
| High-missing columns (>60%) | Retained with median fill |

---

## 🧠 Model Architecture

```
┌──────────────────────────────────────────┐
│  Input Layer  (122 features)             │
├──────────────────────────────────────────┤
│  Dense (256 units, ReLU activation)      │
│  Batch Normalization                     │
│  Dropout (rate = 0.20)                   │
├──────────────────────────────────────────┤
│  Dense (128 units, ReLU activation)      │
│  Batch Normalization                     │
│  Dropout (rate = 0.20)                   │
├──────────────────────────────────────────┤
│  Dense (64 units, ReLU activation)       │
│  Batch Normalization                     │
│  Dropout (rate = 0.10)                   │
├──────────────────────────────────────────┤
│  Dense (32 units, ReLU activation)       │
├──────────────────────────────────────────┤
│  Output Layer (1 unit, Sigmoid)          │
└──────────────────────────────────────────┘
```

### Training Configuration

| Hyperparameter    | Value                  |
|-------------------|------------------------|
| **Optimizer**     | Adam                   |
| **Learning Rate** | 0.001 (adaptive)       |
| **Loss Function** | Binary Crossentropy    |
| **Batch Size**    | 256                    |
| **Max Epochs**    | 100                    |
| **Val Split**     | 20% of training data   |
| **Best Epoch**    | 67 (restored by checkpoint) |
| **Final LR**      | ~6.25e-5 (after 3 reductions) |

### Callbacks

| Callback              | Configuration                                         |
|-----------------------|-------------------------------------------------------|
| `EarlyStopping`       | Monitor: `val_auc`, patience: 10, restore best weights |
| `ReduceLROnPlateau`   | Monitor: `val_auc`, factor: 0.5, patience: 9          |
| `ModelCheckpoint`     | Save best model as `House_Loan_Default_Model.keras`   |

---

## 💡 Key Insights

1. **Class Imbalance is the Core Challenge** — With only 8.1% defaults, a naive model can achieve 91.9% accuracy by predicting "repaid" for everyone. SMOTE rebalancing was essential.

2. **External Credit Scores are Most Predictive** — `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3` (normalized external credit bureau scores) are the strongest predictors of default risk.

3. **Threshold Tuning Matters More Than Architecture** — Moving the decision threshold from 0.50 → 0.40 increased Recall from 23.85% to 32% with minimal accuracy loss, a critical trade-off for risk management.

4. **Batch Normalization + Dropout Prevents Overfitting** — Training accuracy (~89%) and validation accuracy (~90%) tracked closely, indicating good generalization.

5. **Social Circle Behavior is a Proxy** — Customers whose social contacts had prior defaults are measurably higher-risk, validating the social contagion hypothesis in credit risk.

6. **High Dimensionality Requires Careful Preprocessing** — With 122 features, StandardScaler was essential to prevent gradient dominance by high-magnitude features like `AMT_CREDIT`.

---

## 📊 Model Performance & Results

### Test Set Metrics (Threshold = 0.50)

| Metric                   | Score    | Interpretation                              |
|--------------------------|----------|---------------------------------------------|
| **Accuracy**             | 86.26%   | Overall correctness on 61,503 test samples  |
| **Precision (Default)**  | 20.24%   | Of predicted defaults, 20% are actual defaults |
| **Recall (Sensitivity)** | 23.85%   | Of actual defaults, 24% are correctly caught |
| **F1 Score**             | 21.89%   | Harmonic mean of Precision and Recall        |
| **ROC-AUC**              | 0.7033   | Model discriminates better than random (0.5) |

### Classification Report (Threshold = 0.50)

|               | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| **0 — Repaid**    | 0.93  | 0.92   | 0.92     | 56,538  |
| **1 — Default**   | 0.20  | 0.24   | 0.22     |  4,965  |
| **Accuracy**      |       |        | **0.86** | 61,503  |
| **Macro Avg**     | 0.57  | 0.58   | 0.57     | 61,503  |
| **Weighted Avg**  | 0.87  | 0.86   | 0.87     | 61,503  |

### Classification Report (Threshold = 0.40 — Recall-Optimized)

|               | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| **0 — Repaid**    | 0.94  | 0.88   | 0.91     | 56,538  |
| **1 — Default**   | 0.19  | 0.32   | 0.23     |  4,965  |
| **Accuracy**      |       |        | **0.83** | 61,503  |
| **Macro Avg**     | 0.56  | 0.60   | 0.57     | 61,503  |
| **Weighted Avg**  | 0.88  | 0.83   | 0.85     | 61,503  |

> 📌 **Threshold Insight:** Lowering the threshold from 0.50 → 0.40 increases Recall from **23.85% → 32%**, catching more potential defaulters. In financial risk management, **missing a defaulter is far more costly than a false alarm**.

---

## 💼 Business Interpretation

### ❌ False Positive — Predicted Default, Actually Repaid
| Impact | Description |
|--------|-------------|
| Customer Loss | Loan application unnecessarily rejected |
| Revenue Impact | Lost interest income from a creditworthy customer |
| Reputation Risk | Customer may switch to a competitor |
| **Severity** | **Medium** |

### ❌ False Negative — Predicted Repayment, Actually Defaulted
| Impact | Description |
|--------|-------------|
| Financial Loss | Direct loss of principal and interest |
| Credit Risk | Contribution to non-performing assets (NPAs) |
| Regulatory Risk | Impacts capital adequacy ratios |
| **Severity** | **HIGH — Primary concern** |

> ✅ **Conclusion:** For loan default prediction, **minimizing False Negatives (missed defaults)** is far more critical than maximizing accuracy. **Recall (Sensitivity) is the primary business metric.** The model should be positioned as a **decision-support tool**, used alongside human credit officers rather than as a standalone approver.

---

## 🖼️ Visualizations

> The notebook contains the following visualizations (run the notebook to generate):

| # | Visualization             | Description                                        |
|---|---------------------------|----------------------------------------------------|
| 1 | Target Distribution       | Bar chart showing class imbalance (91.9% vs 8.1%) |
| 2 | Training vs Val Accuracy  | Epoch-wise accuracy convergence curves             |
| 3 | Training vs Val Loss      | Binary Crossentropy loss over training epochs      |
| 4 | Training vs Val AUC       | ROC-AUC curve across training epochs               |
| 5 | Precision Curves          | Training & validation precision over epochs        |
| 6 | Recall Curves             | Training & validation recall (sensitivity) over epochs |
| 7 | Confusion Matrix          | Heatmap of TP/TN/FP/FN on test set                |
| 8 | ROC Curve                 | FPR vs TPR with AUC = 0.7033                      |
| 9 | Precision-Recall Curve    | Trade-off curve for threshold selection            |
| 10| Performance Summary Table | Final metrics table at threshold 0.50             |

> 📸 **To add screenshots:** Run the notebook and save plot images to an `images/` folder, then link them here with `![alt](images/filename.png)`.

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- ~4 GB RAM minimum (dataset is ~166 MB)
- GPU recommended (but not required) for faster training

### Step 1 — Clone the Repository

```bash
git clone https://github.com/AbhinavGupta0311/House-Loan-Default-Prediction.git
cd house-loan-default-prediction
```

### Step 2 — Create a Virtual Environment

```bash
# Create environment
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Launch Jupyter Notebook

```bash
jupyter notebook House_Loan_Default_Prediction_Tensorflow.ipynb
```

---

## 🚀 Usage

### Running the Full Pipeline

1. **Open the notebook** in Jupyter or VS Code
2. **Run all cells** sequentially (`Kernel → Restart & Run All`)
3. The notebook will:
   - Load and preprocess `loan_data.csv`
   - Apply SMOTE to balance the training set
   - Train the DNN for up to 100 epochs (early stopping at epoch 77, best at 67)
   - Save the best model as `House_Loan_Default_Model.keras`
   - Generate all evaluation plots and metrics

### Using the Saved Model for Inference

```python
import tensorflow as tf
import numpy as np

# Load the saved model
model = tf.keras.models.load_model("House_Loan_Default_Model.keras")

# Prepare your preprocessed input (shape: [n_samples, 122])
# Ensure the same StandardScaler used in training is applied
X_new = ...  # your scaled feature array

# Get default probability
y_prob = model.predict(X_new)

# Apply threshold (0.40 recommended for higher recall)
threshold = 0.40
y_pred = (y_prob >= threshold).astype(int)

# Result: 0 = Repaid, 1 = Default
```

> ⚠️ **Important:** Always apply the same `StandardScaler` fitted on the training data before predicting on new data to avoid feature scale mismatch.

---

## 📁 Repository Structure

```
📦 house-loan-default-prediction/
 │
 ├── 📓 House_Loan_Default_Prediction_Tensorflow.ipynb   ← Main analysis notebook
 ├── 📄 loan_data.csv                                    ← Raw dataset (~166 MB)
 ├── 📄 README.md                                        ← Project documentation
 ├── 📄 requirements.txt                                 ← Python dependencies
 ├── 📄 .gitignore                                       ← Git exclusion rules
 └── 📄 
```

> 💡 **Recommended structure for scaling this project:**
> ```
> 📦 house-loan-default-prediction/
>  ├── 📂 data/             ← Raw & processed datasets
>  ├── 📂 notebooks/        ← Jupyter notebooks
>  ├── 📂 models/           ← Saved model files (.keras)
>  ├── 📂 images/           ← Visualization screenshots
>  ├── 📂 src/              ← Modular Python scripts
>  ├── 📂 reports/          ← PDF/HTML reports
>  ├── 📄 README.md
>  ├── 📄 requirements.txt
>  ├── 📄 .gitignore
>  └── 📄 LICENSE
> ```

---

## 🔮 Future Improvements

### High Priority
- [ ] **Save & version the `StandardScaler`** using `joblib` for production-safe inference
- [ ] **Persist SMOTE-balanced dataset** to avoid recomputing on each run
- [ ] **Add screenshots** to the `images/` folder and link them in the README

### Medium Priority
- [ ] **Benchmark against classical ML** — XGBoost, LightGBM, Random Forest for comparison
- [ ] **SHAP explainability** — Feature importance analysis using `shap` library
- [ ] **Hyperparameter tuning** — Use `Keras Tuner` or `Optuna` for automated search
- [ ] **Cross-validation** — K-Fold CV to produce more reliable performance estimates
- [ ] **Cost-sensitive learning** — As an alternative strategy to SMOTE

### Nice to Have
- [ ] **REST API deployment** using `FastAPI` or `Flask`
- [ ] **Streamlit dashboard** for interactive loan risk assessment
- [ ] **Docker containerization** for environment reproducibility
- [ ] **CI/CD pipeline** with GitHub Actions for automated testing
- [ ] **Model card** documenting fairness, limitations, and intended use


## 👤 Author

**Abhinav Gupta**

- 💼 [LinkedIn](linkedin.com/in/abhinav-gupta-7b265a277)
- 🐙 [GitHub](https://github.com/AbhinavGupta0311) 
- 📧 [Email](abhinavgupta0311@gmail.com) 

---

## 🙏 Acknowledgements

- Dataset inspired by the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) Kaggle competition
- Deep Learning implementation built with [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/)
- Class imbalance handled using [imbalanced-learn](https://imbalanced-learn.org/)

---

*⭐ If you found this project useful, consider giving it a star on GitHub!*
