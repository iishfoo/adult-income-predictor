import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
LOG_FILE = BASE_DIR / "api.log"

st.set_page_config(page_title="Adult Income Predictor", page_icon="💰", layout="wide")


# ---------- API ko background mein start karna ----------
@st.cache_resource
def start_api_process():
    """API process sirf ek baar start hota hai."""
    try:
        requests.get(f"{API_URL}/health", timeout=2)
        return None  # API pehle se chal rahi hai (local run)
    except Exception:
        pass
    log = open(LOG_FILE, "w")
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api:app",
         "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(BASE_DIR), stdout=log, stderr=subprocess.STDOUT,
    )


def wait_for_api(proc, timeout=120):
    for _ in range(timeout):
        try:
            requests.get(f"{API_URL}/health", timeout=2)
            return True
        except Exception:
            if proc is not None and proc.poll() is not None:
                return False  # process crash ho gaya
            time.sleep(1)
    return False


proc = start_api_process()
with st.spinner("Starting API (pehli baar 1 minute tak lag sakta hai)..."):
    api_ok = wait_for_api(proc)

if not api_ok:
    start_api_process.clear()  # agli refresh par dobara try karega
    st.error("API start nahi ho saki. Neeche uska log hai:")
    if LOG_FILE.exists():
        st.code(LOG_FILE.read_text()[-3000:])
    st.stop()

# ---------- Dropdown values ----------
EDUCATION = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5,
    "10th": 6, "11th": 7, "12th": 8, "HS-grad": 9, "Some-college": 10,
    "Assoc-voc": 11, "Assoc-acdm": 12, "Bachelors": 13, "Masters": 14,
    "Prof-school": 15, "Doctorate": 16,
}
WORKCLASS = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
             "Local-gov", "State-gov", "Without-pay", "Never-worked", "Unknown"]
MARITAL = ["Married-civ-spouse", "Divorced", "Never-married", "Separated",
           "Widowed", "Married-spouse-absent", "Married-AF-spouse"]
OCCUPATION = ["Exec-managerial", "Prof-specialty", "Tech-support", "Sales",
              "Craft-repair", "Adm-clerical", "Other-service", "Machine-op-inspct",
              "Transport-moving", "Handlers-cleaners", "Farming-fishing",
              "Protective-serv", "Priv-house-serv", "Armed-Forces", "Unknown"]
RELATIONSHIP = ["Husband", "Wife", "Own-child", "Not-in-family", "Other-relative", "Unmarried"]
RACE = ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]
SEX = ["Male", "Female"]
COUNTRY = ["United-States", "Mexico", "Philippines", "Germany", "Canada", "Puerto-Rico",
           "El-Salvador", "India", "Cuba", "England", "Jamaica", "South", "China", "Italy",
           "Dominican-Republic", "Vietnam", "Guatemala", "Japan", "Poland", "Columbia",
           "Taiwan", "Haiti", "Iran", "Portugal", "Nicaragua", "Peru", "Greece", "France",
           "Ecuador", "Ireland", "Hong", "Cambodia", "Trinadad&Tobago", "Thailand", "Laos",
           "Yugoslavia", "Outlying-US(Guam-USVI-etc)", "Honduras", "Hungary", "Scotland",
           "Holand-Netherlands", "Unknown"]


def none_if_unknown(v):
    return None if v == "Unknown" else v


@st.cache_data(ttl=60)
def get_metrics():
    r = requests.get(f"{API_URL}/metrics", timeout=10)
    r.raise_for_status()
    return r.json()


st.title("💰 Adult Income Prediction")
st.caption("Predicts whether a person's income exceeds $50K per year.")

# ---------- Model performance + top features ----------
try:
    m = get_metrics()
    st.subheader("Model performance (test set)")
    tm = m["test_metrics"]
    c = st.columns(5)
    c[0].metric("Accuracy", f"{tm['accuracy']:.3f}")
    c[1].metric("Precision", f"{tm['precision']:.3f}")
    c[2].metric("Recall", f"{tm['recall']:.3f}")
    c[3].metric("F1", f"{tm['f1']:.3f}")
    c[4].metric("ROC-AUC", f"{tm['roc_auc']:.3f}")
    st.caption(f"Model: {m['model']}")

    st.subheader("Top 5 most important features")
    fi = pd.DataFrame(m["top_features"]).rename(
        columns={"feature": "Feature", "importance": "Importance"})
    left, right = st.columns([1, 1])
    left.dataframe(fi, hide_index=True, use_container_width=True)
    right.bar_chart(fi.set_index("Feature")["Importance"])
except Exception as e:
    st.error(f"API se data nahi mil saka ({API_URL}). Page refresh karo. Detail: {e}")
    if LOG_FILE.exists():
        st.code(LOG_FILE.read_text()[-3000:])
    st.stop()

# ---------- Input form ----------
st.subheader("Enter person details")
with st.form("predict_form"):
    a, b, c3 = st.columns(3)
    with a:
        age = st.number_input("Age", 17, 100, 40)
        education = st.selectbox("Education", list(EDUCATION.keys()), index=13)
        marital = st.selectbox("Marital status", MARITAL)
        relationship = st.selectbox("Relationship", RELATIONSHIP)
    with b:
        workclass = st.selectbox("Workclass", WORKCLASS)
        occupation = st.selectbox("Occupation", OCCUPATION)
        race = st.selectbox("Race", RACE)
        sex = st.selectbox("Sex", SEX)
    with c3:
        hours = st.number_input("Hours per week", 1, 100, 40)
        gain = st.number_input("Capital gain", 0, 100000, 0, step=100)
        loss = st.number_input("Capital loss", 0, 5000, 0, step=50)
        country = st.selectbox("Native country", COUNTRY)
    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "age": int(age),
        "workclass": none_if_unknown(workclass),
        "education_num": EDUCATION[education],
        "marital_status": marital,
        "occupation": none_if_unknown(occupation),
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital_gain": int(gain),
        "capital_loss": int(loss),
        "hours_per_week": int(hours),
        "native_country": none_if_unknown(country),
    }
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
        r.raise_for_status()
        res = r.json()
        prob = res["probability_over_50k"]
        if res["label"] == 1:
            st.success(f"Prediction: income **{res['prediction']}**")
        else:
            st.info(f"Prediction: income **{res['prediction']}**")
        st.progress(min(max(prob, 0.0), 1.0), text=f"Probability of >50K: {prob:.1%}")
    except Exception as e:
        st.error(f"Prediction fail hui: {e}")
