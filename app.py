import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


st.set_page_config(
    page_title="LoanIQ - AI Credit Analyzer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",   
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
.stApp { background: #0a0e1a; color: #e8eaf6; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Number inputs ── */
input[type="number"] {
    background: #1a2235 !important;
    border: 1px solid #2a3a5c !important;
    color: #7dd3fc !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1rem !important;
    padding: 0.5rem !important;
}
.stNumberInput label {
    color: #94a3b8 !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
}

/* ── THE FIX: Big visible Analyse button on main page ── */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 1rem 2rem !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    width: 100% !important;
    letter-spacing: 0.05em !important;
    box-shadow: 0 6px 24px rgba(59,130,246,0.45) !important;
    transition: all 0.2s ease !important;
    margin-top: 0.5rem !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 32px rgba(59,130,246,0.6) !important;
}

/* ── Input card wrapper ── */
.input-card {
    background: linear-gradient(135deg, #111827, #1a2235);
    border: 1px solid #1e2d4a;
    border-radius: 18px;
    padding: 1.6rem 1.8rem 1.2rem;
    margin-bottom: 1.2rem;
}

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #111827, #1a2235);
    border: 1px solid #1e2d4a;
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 3px; border-radius: 16px 16px 0 0;
}
.metric-card.blue::before  { background: linear-gradient(90deg,#3b82f6,#60a5fa); }
.metric-card.green::before { background: linear-gradient(90deg,#10b981,#34d399); }
.metric-card.red::before   { background: linear-gradient(90deg,#ef4444,#f87171); }
.metric-card.amber::before { background: linear-gradient(90deg,#f59e0b,#fbbf24); }

.metric-label { font-size:.67rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:#64748b; margin-bottom:.3rem; }
.metric-value { font-family:'JetBrains Mono',monospace; font-size:1.8rem; font-weight:700; line-height:1; margin-bottom:.2rem; }
.metric-value.blue  { color:#60a5fa; }
.metric-value.green { color:#34d399; }
.metric-value.red   { color:#f87171; }
.metric-value.amber { color:#fbbf24; }
.metric-sub { font-size:.73rem; color:#475569; }

/* ── Decision banner ── */
.decision-approved {
    background: linear-gradient(135deg,#064e3b,#065f46);
    border: 1px solid #059669; border-radius: 16px;
    padding: 1.4rem 2rem; text-align: center; margin-bottom: 1.4rem;
}
.decision-rejected {
    background: linear-gradient(135deg,#450a0a,#7f1d1d);
    border: 1px solid #dc2626; border-radius: 16px;
    padding: 1.4rem 2rem; text-align: center; margin-bottom: 1.4rem;
}
.decision-text { font-size: 1.55rem; font-weight: 700; margin: 0; }
.decision-sub  { font-size: .82rem; opacity: .72; margin-top: .3rem; }

/* ── Rule badges ── */
.rule-badge {
    background:#1a0a0a; border:1px solid #7f1d1d; border-radius:8px;
    padding:.55rem 1rem; margin-bottom:.4rem; font-size:.82rem; color:#fca5a5;
    border-left:3px solid #ef4444;
}
.rule-ok {
    background:#0a1a12; border:1px solid #065f46; border-radius:8px;
    padding:.55rem 1rem; margin-bottom:.4rem; font-size:.82rem; color:#86efac;
    border-left:3px solid #10b981;
}

.section-title {
    font-size:.67rem; font-weight:700; letter-spacing:.12em;
    text-transform:uppercase; color:#3b82f6;
    margin-bottom:.85rem; padding-bottom:.4rem; border-bottom:1px solid #1e2d4a;
}
.summary-row {
    display:flex; justify-content:space-between;
    padding:.42rem 0; border-bottom:1px solid #1a2235; font-size:.82rem;
}
.summary-key   { color:#64748b; }
.summary-value { color:#e2e8f0; font-family:'JetBrains Mono',monospace; font-weight:500; }

.factor-item {
    background:#111827; border:1px solid #1e2d4a; border-radius:10px;
    padding:.58rem .9rem; margin-bottom:.4rem; font-size:.82rem; color:#cbd5e1;
}
.factor-item.warn { border-left:3px solid #f59e0b; }
.factor-item.good { border-left:3px solid #10b981; }
.factor-item.info { border-left:3px solid #3b82f6; }

.prob-bar-wrap { background:#1a2235; border-radius:100px; height:9px; overflow:hidden; margin:.4rem 0 .8rem; }
.prob-bar-fill { height:100%; border-radius:100px; }

/* ── App header ── */
.app-title {
    font-size: 1.7rem; font-weight: 700;
    background: linear-gradient(135deg,#60a5fa,#a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0;
}
.app-subtitle { font-size:.79rem; color:#475569; margin-top:.35rem; }

/* ── Section header with icon ── */
.section-header {
    font-size:.9rem; font-weight:700; color:#e2e8f0;
    margin-bottom:1rem; display:flex; align-items:center; gap:.5rem;
}

/* ── Responsive: stack columns on small screens ── */
@media (max-width: 768px) {
    .metric-value { font-size: 1.4rem !important; }
    .decision-text { font-size: 1.2rem !important; }
    .input-card { padding: 1rem !important; }
}
</style>
""", unsafe_allow_html=True)



@st.cache_resource
def load_model():
    m  = pickle.load(open("loan_model.pkl",    "rb"))
    sc = pickle.load(open("scaler.pkl",         "rb"))
    fn = pickle.load(open("feature_names.pkl",  "rb"))
    try:
        th = pickle.load(open("threshold.pkl",  "rb"))
    except FileNotFoundError:
        th = 0.50
    return m, sc, fn, th

model, scaler, feature_names, MODEL_THRESHOLD = load_model()



def check_bank_rules(income, loan, emp, rate, credit):
    monthly    = (loan * rate / 100) / 12
    dti        = loan    / (income + 1)
    pti        = monthly / (income / 12 + 1)

    violations = []
    passes     = []

    checks = [
        (dti  > 0.43,  f"DTI {dti:.1%} exceeds 43% limit (loan too large vs income)",     f"DTI {dti:.1%} within 43% limit ✓"),
        (pti  > 0.35,  f"Monthly payment {pti:.1%} of income — exceeds 35% limit",          f"Monthly payment {pti:.1%} within 35% limit ✓"),
        (rate > 22,    f"Interest rate {rate}% exceeds 22% — predatory rate",               f"Interest rate {rate}% is acceptable ✓"),
        (emp  < 1.0,   f"Employment {emp}yr — minimum 1 year required",                     f"Employment {emp}yr meets minimum ✓"),
        (credit < 2,   f"Credit history {credit}yr — minimum 2 years required",             f"Credit history {credit}yr meets minimum ✓"),
        (loan > income,f"Loan ${loan:,} exceeds annual income ${income:,}",                  f"Loan within annual income ✓"),
    ]
    for failed, fail_msg, pass_msg in checks:
        if failed:
            violations.append(fail_msg)
        else:
            passes.append(pass_msg)

    return violations, passes, dti, pti, monthly



def make_gauge(prob, threshold, hard_rejected=False):
    fig, ax = plt.subplots(figsize=(4.5, 2.6), facecolor="#0a0e1a")
    ax.set_facecolor("#0a0e1a")
    theta = np.linspace(np.pi, 0, 300)
    ax.plot(np.cos(theta), np.sin(theta), color="#1e2d4a", linewidth=18, solid_capstyle="round")

    if hard_rejected:
        color = "#374151"
        theta_f = np.linspace(np.pi, np.pi - prob * np.pi, 300)
        ax.plot(np.cos(theta_f), np.sin(theta_f), color=color, linewidth=18, solid_capstyle="round")
        ax.text(0, -0.08, "HARD REJECT", ha="center", fontsize=13, fontweight="bold", color="#ef4444", fontfamily="monospace")
        ax.text(0, -0.38, "bank rule violation", ha="center", fontsize=7.5, color="#475569")
    else:
        color = "#10b981" if prob >= 0.75 else "#f59e0b" if prob >= 0.5 else "#ef4444"
        theta_f = np.linspace(np.pi, np.pi - prob * np.pi, 300)
        ax.plot(np.cos(theta_f), np.sin(theta_f), color=color, linewidth=18, solid_capstyle="round")
        ta = np.pi - threshold * np.pi
        ax.annotate("", xy=(np.cos(ta)*0.88, np.sin(ta)*0.88), xytext=(0,0),
                    arrowprops=dict(arrowstyle="->", color="#3b82f6", lw=1.8))
        ax.text(0, -0.1, f"{prob:.1%}", ha="center", va="center",
                fontsize=22, fontweight="bold", color=color, fontfamily="monospace")
        ax.text(0, -0.38, "approval probability", ha="center", fontsize=7.5, color="#475569")
        ax.plot([], [], color="#3b82f6", linestyle="-", linewidth=2, label=f"Threshold: {threshold:.2f}")
        ax.legend(loc="lower center", fontsize=7, framealpha=0, labelcolor="#64748b")

    ax.text(-1.05, -0.15, "0%",    ha="center", fontsize=7, color="#334155")
    ax.text( 1.05, -0.15, "100%",  ha="center", fontsize=7, color="#334155")
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.6, 1.25); ax.axis("off")
    plt.tight_layout(pad=0)
    return fig



def build_features(income, loan, emp, rate, credit, monthly, pti, dti):
    return {
        "person_income":              income,
        "loan_amnt":                  loan,
        "person_emp_length":          emp,
        "loan_int_rate":              rate,
        "cb_person_cred_hist_length": credit,
        "dti_ratio":                  dti,
        "monthly_payment":            monthly,
        "pti_ratio":                  pti,
        "income_loan_ratio":          income / (loan + 1),
        "interest_income_burden":     rate * loan / (income + 1),
        "emp_credit_ratio":           emp / (credit + 1),
        "annual_interest_cost":       loan * rate / 100,
        "loan_per_emp_year":          loan / (emp + 1),
    }



st.markdown("""
<div style="padding:1.2rem 0 0.6rem">
  <p class="app-title">🏦 Loan Analyzer and Risk Detection</p>
  <p class="app-subtitle">Random Forest · SMOTE · Real Bank Standards </p>
</div>
<hr style="border-color:#1e2d4a; margin-bottom:1.2rem">
""", unsafe_allow_html=True)


st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📋 Applicant Details</div>', unsafe_allow_html=True)


col1, col2 = st.columns(2)
with col1:
    income = st.number_input(
        "Annual Income ($)", min_value=20000, max_value=500000,
        value=60000, step=1000, key="income"
    )
with col2:
    loan_amount = st.number_input(
        "Loan Amount ($)", min_value=1000, max_value=100000,
        value=15000, step=500, key="loan"
    )


col3, col4 = st.columns(2)
with col3:
    emp_length = st.number_input(
        "Employment Length (Years)", min_value=0.0, max_value=40.0,
        value=5.0, step=0.5, key="emp"
    )
with col4:
    interest_rate = st.number_input(
        "Interest Rate (%)", min_value=5.0, max_value=30.0,
        value=12.0, step=0.5, key="rate"
    )


col5, col6 = st.columns(2)
with col5:
    credit_history = st.number_input(
        "Credit History (Years)", min_value=0, max_value=30,
        value=6, step=1, key="credit"
    )
with col6:
    st.markdown("<br>", unsafe_allow_html=True)  
    analyze = st.button("⚡  Analyse Loan Application")

st.markdown('</div>', unsafe_allow_html=True)


st.markdown('<div class="section-title" style="margin-top:.2rem">💡 Quick Test Examples</div>', unsafe_allow_html=True)
ex1, ex2, ex3 = st.columns(3)
with ex1:
    st.markdown('<div class="rule-ok" style="text-align:center;font-size:.78rem;cursor:pointer">✅ <strong>Strong Profile</strong><br>$180k income · $10k loan<br>12yr emp · 8% · 10yr credit</div>', unsafe_allow_html=True)
with ex2:
    st.markdown('<div class="factor-item amber" style="text-align:center;font-size:.78rem;cursor:pointer">🟡 <strong>Borderline</strong><br>$55k income · $20k loan<br>3yr emp · 15% · 4yr credit</div>', unsafe_allow_html=True)
with ex3:
    st.markdown('<div class="rule-badge" style="text-align:center;font-size:.78rem;cursor:pointer">❌ <strong>Weak Profile</strong><br>$22k income · $48k loan<br>0.5yr emp · 24% · 1yr credit</div>', unsafe_allow_html=True)

st.markdown('<hr style="border-color:#1e2d4a; margin:1.2rem 0">', unsafe_allow_html=True)


if not analyze:
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">How the 2-Gate System Works</div>', unsafe_allow_html=True)
        for num, title, desc in [
            ("1️⃣", "Hard Bank Rules",  "Instant reject if ANY rule violated"),
            ("2️⃣", "AI Model Score",   "Random Forest approval probability"),
            ("3️⃣", "Threshold Check",  "Score must exceed 0.50"),
            ("4️⃣", "Final Decision",   "Both gates must pass to approve"),
        ]:
            st.markdown(f'<div class="factor-item info">{num} <strong>{title}</strong> — <span style="color:#475569">{desc}</span></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-title">Bank Rule Limits</div>', unsafe_allow_html=True)
        for icon, rule, detail in [
            ("🚫","DTI ≤ 43%",         "Debt-to-Income (QM Standard)"),
            ("🚫","PTI ≤ 35%",         "Monthly payment vs income"),
            ("🚫","Rate ≤ 22%",        "Non-predatory rate limit"),
            ("🚫","Employment ≥ 1yr",  "Job stability requirement"),
            ("🚫","Credit ≥ 2yr",      "Minimum credit history"),
            ("🚫","Loan ≤ Income",     "Cannot exceed annual salary"),
        ]:
            st.markdown(f'<div class="rule-badge">{icon} <strong>{rule}</strong> — {detail}</div>', unsafe_allow_html=True)

else:
  
    violations, passes, dti, pti, monthly = check_bank_rules(
        income, loan_amount, emp_length, interest_rate, credit_history
    )
    hard_rejected = len(violations) > 0

    row    = build_features(income, loan_amount, emp_length, interest_rate,
                            credit_history, monthly, pti, dti)
    df_in  = pd.DataFrame([row])[feature_names]
    df_sc  = scaler.transform(df_in)
    prob   = float(model.predict_proba(df_sc)[0][1])

    approved = (prob >= MODEL_THRESHOLD) and not hard_rejected

    if prob >= 0.75:   rc, ri, risk = "green", "🟢", "Low Risk"
    elif prob >= 0.50: rc, ri, risk = "amber", "🟡", "Medium Risk"
    else:              rc, ri, risk = "red",   "🔴", "High Risk"
    if hard_rejected:  rc, ri, risk = "red",   "🔴", "High Risk"

    pc      = "green" if prob >= 0.75 else "amber" if prob >= 0.5 else "red"
    bar_hex = {"green":"#10b981","amber":"#f59e0b","red":"#ef4444"}[pc]

    
    if approved:
        st.markdown('<div class="decision-approved"><p class="decision-text">✅ &nbsp;Loan Approved</p><p class="decision-sub">Passed all bank rules and AI credit scoring</p></div>', unsafe_allow_html=True)
    elif hard_rejected:
        st.markdown(f'<div class="decision-rejected"><p class="decision-text">❌ &nbsp;Loan Rejected</p><p class="decision-sub">Failed {len(violations)} bank rule{"s" if len(violations)>1 else ""} — automatic rejection</p></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="decision-rejected"><p class="decision-text">❌ &nbsp;Loan Rejected</p><p class="decision-sub">AI model score below approval threshold</p></div>', unsafe_allow_html=True)

    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card {pc}"><div class="metric-label">AI Score</div><div class="metric-value {pc}">{prob:.1%}</div><div class="metric-sub">Threshold: {MODEL_THRESHOLD:.2f}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card {rc}"><div class="metric-label">Risk Level</div><div class="metric-value {rc}">{ri}</div><div class="metric-sub">{risk}</div></div>', unsafe_allow_html=True)
    dc  = "red"   if dti > 0.43 else "green"
    ptc = "red"   if pti > 0.35 else "green"
    with m3:
        st.markdown(f'<div class="metric-card {dc}"><div class="metric-label">Debt-to-Income</div><div class="metric-value {dc}">{dti:.1%}</div><div class="metric-sub">Limit: 43%</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card {ptc}"><div class="metric-label">Payment-to-Income</div><div class="metric-value {ptc}">{pti:.1%}</div><div class="metric-sub">Limit: 35%</div></div>', unsafe_allow_html=True)


    left, right = st.columns([1, 1.1])

    with left:
        
        st.markdown('<div class="section-title">Confidence Gauge</div>', unsafe_allow_html=True)
        st.pyplot(make_gauge(prob, MODEL_THRESHOLD, hard_rejected), use_container_width=True)
        if not hard_rejected:
            st.markdown(f'<div class="prob-bar-wrap"><div class="prob-bar-fill" style="width:{prob*100:.1f}%;background:{bar_hex};"></div></div>', unsafe_allow_html=True)

        
        st.markdown('<div class="section-title" style="margin-top:.8rem">Financial Summary</div>', unsafe_allow_html=True)
        for k, v in [
            ("Annual Income",        f"${income:,.0f}"),
            ("Loan Amount",          f"${loan_amount:,.0f}"),
            ("Loan / Income",        f"{dti:.1%}"),
            ("Monthly Payment",      f"${monthly:,.0f}"),
            ("Interest Rate",        f"{interest_rate}%"),
            ("Employment",           f"{emp_length} yrs"),
            ("Credit History",       f"{credit_history} yrs"),
        ]:
            st.markdown(f'<div class="summary-row"><span class="summary-key">{k}</span><span class="summary-value">{v}</span></div>', unsafe_allow_html=True)

    with right:
        
        st.markdown('<div class="section-title">Bank Rule Checklist</div>', unsafe_allow_html=True)
        for v in violations:
            st.markdown(f'<div class="rule-badge">🚫 {v}</div>', unsafe_allow_html=True)
        for p in passes:
            st.markdown(f'<div class="rule-ok">✓ {p}</div>', unsafe_allow_html=True)

        
        if not approved:
            st.markdown('<div class="section-title" style="margin-top:.8rem">💡 How to Get Approved</div>', unsafe_allow_html=True)
            suggestions = []
            if loan_amount > income * 0.40:
                suggestions.append(f"Reduce loan to <strong>${int(income*0.40):,}</strong> (40% of income)")
            if interest_rate > 15:
                suggestions.append("Negotiate rate below <strong>15%</strong>")
            if emp_length < 2:
                suggestions.append("Build <strong>2+ years</strong> employment history")
            if credit_history < 3:
                suggestions.append("Build <strong>3+ years</strong> credit history")
            if pti > 0.35:
                max_loan_pti = int((0.30 * income / 12) / (interest_rate / 100 / 12))
                suggestions.append(f"Max affordable loan at this rate: <strong>${max_loan_pti:,}</strong>")
            if not suggestions:
                suggestions.append("Improve overall financial profile and reapply in 6 months")
            for s in suggestions:
                st.markdown(f'<div class="factor-item info">🔧 {s}</div>', unsafe_allow_html=True)


        if approved:
            st.markdown('<div class="section-title" style="margin-top:.8rem">✅ Approval Factors</div>', unsafe_allow_html=True)
            if dti   <= 0.3:  st.markdown(f'<div class="factor-item good">✅ Healthy DTI ({dti:.0%})</div>', unsafe_allow_html=True)
            if emp_length >= 5: st.markdown(f'<div class="factor-item good">✅ Stable employment ({emp_length:.0f} yrs)</div>', unsafe_allow_html=True)
            if credit_history >= 5: st.markdown(f'<div class="factor-item good">✅ Strong credit history ({credit_history} yrs)</div>', unsafe_allow_html=True)
            if interest_rate <= 12: st.markdown(f'<div class="factor-item good">✅ Reasonable rate ({interest_rate}%)</div>', unsafe_allow_html=True)