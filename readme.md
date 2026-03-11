# 🏦 LoanIQ — AI Loan Approval & Risk Analyzer

> An end-to-end machine learning application that predicts loan approval decisions using **real banking standards** — not just an ML demo.

**🔗 Live Demo → [ailoanapprovalandriskanalyzer.streamlit.app](https://ailoanapprovalandriskanalyzer-cprppeulz5z9owfjhixb2s.streamlit.app/)**

---

## 📸 Preview

| Approved Profile | Rejected Profile |
|---|---|
| High income, low loan, long employment | High loan, short employment, poor credit |
| ✅ Passes all 6 bank rules | ❌ Fails DTI / PTI limits instantly |

---

## 🎯 What Makes This Different

Most loan ML projects just train a model and call it done. LoanIQ adds a **Two-Gate Decision System** that mirrors how real banks work:

```
Application → Gate 1: Hard Bank Rules → Gate 2: AI Model Score → Final Decision
                 ↓ (any violation = instant reject)    ↓ (must be ≥ 0.50)
```

| Gate | Name | Logic |
|---|---|---|
| 1 | Hard Bank Rules | Instant reject if ANY rule violated |
| 2 | AI Model Score | Random Forest probability must be ≥ 0.50 |

Both gates must pass for approval — exactly like a real credit analyst.

---

## ✨ Features

- 🏦 **Real Bank Hard Rules** — DTI ≤ 43% (QM Standard), PTI ≤ 35%, Rate ≤ 22%, Employment ≥ 1yr
- 🤖 **Random Forest Model** — 300 trees, trained on SMOTE-balanced data
- 🧹 **Label Cleaning** — Training labels corrected using actual lending criteria before model training
- 📊 **Confidence Gauge Chart** — Semicircle probability visualization with threshold needle
- ✅ **Rule Checklist** — Shows exactly which bank rules passed or failed
- 💡 **Rejection Suggestions** — Tells applicant exactly what to change to get approved
- 📱 **Fully Responsive** — Works on mobile, tablet, and desktop (no sidebar dependency)
- 🌑 **Dark Banking UI** — Professional dark theme with Sora + JetBrains Mono fonts

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Model** | scikit-learn Random Forest |
| **Balancing** | imbalanced-learn SMOTE |
| **UI** | Streamlit |
| **Charts** | Matplotlib |
| **Data** | pandas, numpy |
| **Deployment** | Streamlit Cloud |

---

## 📁 Project Structure

```
ai-loan-approval/
│
├── app.py                    # Streamlit web app — UI, prediction, results
├── train_model.py            # Model training pipeline
├── requirements.txt          # Python dependencies for deployment
│
├── loan_model.pkl            # Trained Random Forest model
├── scaler.pkl                # Fitted StandardScaler
├── threshold.pkl             # Decision threshold (0.50)
├── feature_names.pkl         # Ordered feature list (train/predict alignment)
│
└── credit_risk_dataset.csv   # Source dataset (Kaggle)
```

---

## 🚀 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/ai-loan-approval.git
cd ai-loan-approval
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python train_model.py
```
This generates the 4 `.pkl` files. Check the sanity check output — IDEAL/GOOD should be APPROVED, WEAK should be REJECTED.

### 4. Launch the app
```bash
streamlit run app.py
```

---

## ☁️ Run on Google Colab

No local setup needed:

```python
# Cell 1
!pip install scikit-learn pandas imbalanced-learn streamlit pyngrok -q

# Cell 2 — upload credit_risk_dataset.csv
from google.colab import files
files.upload()

# Cell 3 — train (run train_model.py content here)

# Cell 4 — launch app
from pyngrok import ngrok
!streamlit run app.py &
print(ngrok.connect(8501))
```

---

## 🏦 Bank Hard Rules Applied

These rules are enforced at **prediction time** AND used to **clean training labels**:

| Rule | Limit | Standard |
|---|---|---|
| Debt-to-Income (DTI) | ≤ 43% | US Qualified Mortgage (QM) |
| Payment-to-Income (PTI) | ≤ 35% | Monthly burden limit |
| Interest Rate | ≤ 22% | Predatory lending threshold |
| Employment Length | ≥ 1 year | Minimum job stability |
| Credit History | ≥ 2 years | Minimum credit track record |
| Loan vs Annual Income | ≤ 100% | Cannot exceed annual salary |

---

## 🧪 Test Cases

| Profile | Income | Loan | Emp | Rate | Credit | Expected |
|---|---|---|---|---|---|---|
| IDEAL | $180,000 | $10,000 | 12yr | 8% | 10yr | ✅ APPROVED |
| GOOD | $100,000 | $30,000 | 7yr | 11% | 8yr | ✅ APPROVED |
| FAIR | $60,000 | $15,000 | 5yr | 12% | 6yr | ✅ APPROVED |
| BORDERLINE | $40,000 | $20,000 | 3yr | 16% | 3yr | ❌ REJECTED |
| RISKY | $60,000 | $55,000 | 5yr | 12% | 2yr | ❌ REJECTED |
| WEAK | $22,000 | $48,000 | 0.5yr | 24% | 1yr | ❌ REJECTED |

---

## 🧠 Model Features (13 Total)

| Feature | Description |
|---|---|
| `person_income` | Annual income |
| `loan_amnt` | Loan amount requested |
| `person_emp_length` | Years employed |
| `loan_int_rate` | Interest rate (%) |
| `cb_person_cred_hist_length` | Credit history (years) |
| `dti_ratio` | Loan / Income |
| `monthly_payment` | Estimated monthly payment |
| `pti_ratio` | Monthly payment / Monthly income |
| `income_loan_ratio` | Income / Loan |
| `interest_income_burden` | Rate × Loan / Income |
| `emp_credit_ratio` | Employment years / Credit years |
| `annual_interest_cost` | Loan × Rate / 100 |
| `loan_per_emp_year` | Loan / Employment years |

---

## ⚠️ Key Problems Solved

### Problem 1 — Model approved everything (threshold 0.28)
**Root cause:** F1-based threshold auto-selection on imbalanced data picked 0.28 — nearly everyone got approved.  
**Fix:** Cleaned training labels using bank rules first, then fixed threshold at 0.50.

### Problem 2 — Inverted labels
**Root cause:** Dataset uses `1 = Default (bad)` but model predicted `1 = Approved`.  
**Fix:** `df["loan_status"] = 1 - df["loan_status"]`

### Problem 3 — Feature name mismatch
**Root cause:** `train_model.py` and `app.py` used different feature names — crash at prediction.  
**Fix:** Both files use identical feature engineering, `feature_names.pkl` enforces column order.

### Problem 4 — Mobile button not visible
**Root cause:** Analyse button was inside Streamlit sidebar — collapses on mobile.  
**Fix:** Moved all inputs and button to main page in a responsive 2-column grid.

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | ~87% |
| ROC-AUC | ~0.91 |
| Threshold | 0.50 (F1-tuned after label cleaning) |
| Balancing | SMOTE oversampling |

---

## 📦 Requirements

```
streamlit
scikit-learn
pandas
numpy
matplotlib
imbalanced-learn
```

Install with:
```bash
pip install -r requirements.txt
```

---



## 📄 License

This project is licensed under the MIT License.

---

## 🙋 About

Built as a full-stack ML project demonstrating end-to-end model development, real-world domain knowledge (US lending standards), and production deployment on Streamlit Cloud.

**Dataset:** [Credit Risk Dataset — Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)

---

<p align="center">
  <strong>🏦 LoanIQ</strong> — Built with Streamlit, scikit-learn & real banking standards
</p>
