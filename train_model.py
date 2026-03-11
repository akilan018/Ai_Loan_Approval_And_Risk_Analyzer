import pandas as pd
import pickle
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE



df = pd.read_csv("credit_risk_dataset.csv")

FEATURES = [
    "person_income",
    "loan_amnt",
    "person_emp_length",
    "loan_int_rate",
    "cb_person_cred_hist_length",
    "loan_status"
]

df = df[FEATURES].dropna()

df = df[df["person_emp_length"] <= 60]
df = df[df["person_income"] <= 4_000_000]

print("Dataset loaded:", df.shape)
print("\nOriginal labels:")
print(df["loan_status"].value_counts())



df["loan_status"] = 1 - df["loan_status"]

print("\nAfter fixing labels:")
print(df["loan_status"].value_counts())
print(f"Raw approval rate: {df['loan_status'].mean():.1%}")


def apply_bank_rules(row):
    income  = row["person_income"]
    loan    = row["loan_amnt"]
    rate    = row["loan_int_rate"]
    emp     = row["person_emp_length"]
    credit  = row["cb_person_cred_hist_length"]

    monthly = (loan * rate / 100) / 12
    dti     = loan    / (income + 1)
    pti     = monthly / (income / 12 + 1)

    # Hard REJECT rules
    if dti   > 0.43:  return 0
    if pti   > 0.35:  return 0
    if rate  > 22:    return 0
    if emp   < 1.0:   return 0
    if credit < 2:    return 0
    if loan  > income: return 0

    # Hard APPROVE — clearly strong profile
    if dti <= 0.2 and emp >= 5 and credit >= 5 and rate <= 15:
        return 1

    return row["loan_status"]   # keep original for everything else

print("\n⏳ Applying bank rules to clean training labels...")
df["loan_status"] = df.apply(apply_bank_rules, axis=1)
print(f"Approval rate after bank rules: {df['loan_status'].mean():.1%}")
print(df["loan_status"].value_counts())



df["dti_ratio"]              = df["loan_amnt"]         / (df["person_income"] + 1)
df["monthly_payment"]        = (df["loan_amnt"] * df["loan_int_rate"] / 100) / 12
df["pti_ratio"]              = df["monthly_payment"]   / (df["person_income"] / 12 + 1)
df["income_loan_ratio"]      = df["person_income"]     / (df["loan_amnt"] + 1)
df["interest_income_burden"] = df["loan_int_rate"] * df["loan_amnt"] / (df["person_income"] + 1)
df["emp_credit_ratio"]       = df["person_emp_length"] / (df["cb_person_cred_hist_length"] + 1)
df["annual_interest_cost"]   = df["loan_amnt"] * df["loan_int_rate"] / 100
df["loan_per_emp_year"]      = df["loan_amnt"] / (df["person_emp_length"] + 1)

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

FEATURE_NAMES = list(X.columns)

print("\nTotal Features:", len(FEATURE_NAMES))
print("Feature names:", FEATURE_NAMES)



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



scaler = StandardScaler()

X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)



print("\nApplying SMOTE...")

smote = SMOTE(random_state=42)

X_train_bal, y_train_bal = smote.fit_resample(X_train_sc, y_train)

print("Balanced class distribution:")
print(pd.Series(y_train_bal).value_counts())


print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_bal, y_train_bal)

print("Training completed!")



probs = model.predict_proba(X_test_sc)[:, 1]

print("\n── Threshold Analysis (pick the one that looks realistic) ──")
print(f"  {'Threshold':<10} {'Approval Rate':<16} Accuracy")
for t in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    preds = (probs >= t).astype(int)
    flag  = " ← SELECTED" if t == 0.50 else ""
    print(f"  t={t:.2f}      {preds.mean():.1%}            "
          f"{accuracy_score(y_test, preds):.4f}{flag}")

THRESHOLD = 0.50



y_pred = (probs >= THRESHOLD).astype(int)

print(f"\nUsing Threshold = {THRESHOLD}")
print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("ROC AUC  :", round(roc_auc_score(y_test, probs),   4))
print(f"Approval rate on test set: {y_pred.mean():.1%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))

# Feature importance
imp = sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda x: -x[1])
print("Feature Importances:")
for name, v in imp:
    bar = "█" * int(v * 60)
    print(f"  {name:<35} {bar} {v:.4f}")


def quick_check(income, loan, emp, rate, credit, label=""):
    monthly = (loan * rate / 100) / 12
    dti     = loan / (income + 1)
    pti     = monthly / (income / 12 + 1)
    row = {
        "person_income": income, "loan_amnt": loan,
        "person_emp_length": emp, "loan_int_rate": rate,
        "cb_person_cred_hist_length": credit,
        "dti_ratio":             dti,
        "monthly_payment":       monthly,
        "pti_ratio":             pti,
        "income_loan_ratio":     income / (loan + 1),
        "interest_income_burden":rate * loan / (income + 1),
        "emp_credit_ratio":      emp / (credit + 1),
        "annual_interest_cost":  loan * rate / 100,
        "loan_per_emp_year":     loan / (emp + 1),
    }
    d  = pd.DataFrame([row])[FEATURE_NAMES]
    sc = scaler.transform(d)
    p  = model.predict_proba(sc)[0][1]
    r  = "✅ APPROVED" if p >= THRESHOLD else "❌ REJECTED"
    print(f"  {label:<12} Income=${income:>7,}  Loan=${loan:>6,}  "
          f"Emp={emp:>4}yr  Rate={rate:>4}%  DTI={dti:.0%}  →  {p:.1%} {r}")

print("\n─── Sanity Checks ────────────────────────────────────────────")
quick_check(180000, 10000, 12,   8, 10, "IDEAL")       # ✅ APPROVE
quick_check(100000, 30000,  7,  11,  8, "GOOD")        # ✅ APPROVE
quick_check( 60000, 15000,  5,  12,  6, "FAIR")        # ✅ APPROVE (DTI=25%)
quick_check( 60000, 55000,  5,  12,  2, "RISKY")       # ❌ REJECT  (DTI=91%)
quick_check( 40000, 20000,  3,  16,  3, "BORDERLINE")  # ❌ REJECT
quick_check( 22000, 48000,  1,  24,  1, "WEAK")        # ❌ REJECT
print("\nExpected:  IDEAL / GOOD / FAIR = APPROVED")
print("           RISKY / BORDERLINE / WEAK = REJECTED")



pickle.dump(model,         open("loan_model.pkl",    "wb"))
pickle.dump(scaler,        open("scaler.pkl",         "wb"))
pickle.dump(THRESHOLD,     open("threshold.pkl",      "wb"))
pickle.dump(FEATURE_NAMES, open("feature_names.pkl",  "wb"))

print("\n✅ Model files saved successfully!")
print("   📦 loan_model.pkl")
print("   📦 scaler.pkl")
print(f"  📦 threshold.pkl      ← value: {THRESHOLD}")
print(f"  📦 feature_names.pkl  ← {len(FEATURE_NAMES)} features")
print("\n🚀 Upload these 4 files and run app.py!")