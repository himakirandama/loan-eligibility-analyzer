import streamlit as st
import re
import pandas as pd
import joblib
from PyPDF2 import PdfReader

# Load trained model
model = joblib.load("loan_approval_model.pkl")

# ---------------- PDF Parsing ---------------- #
# ---------------- PDF Parsing ---------------- #
def parse_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    data = {}

    # CIBIL Score
    try:
        data["cibil_score"] = re.search(r"CIBIL Score.*?:\s*(\d+)", text, re.IGNORECASE).group(1)
    except:
        data["cibil_score"] = None

    # Loan Term
    try:
        data["loan_term"] = re.search(r"Loan term.*?:\s*(\d+)", text, re.IGNORECASE).group(1)
    except:
        data["loan_term"] = None

    # Income
    try:
        data["income_annum"] = re.search(r"Income.*?:\s*(\d+)", text, re.IGNORECASE).group(1)
    except:
        data["income_annum"] = None

    # Number of Dependents
    try:
        data["no_of_dependents"] = re.search(r"Dependents.*?:\s*(\d+)", text, re.IGNORECASE).group(1)
    except:
        data["no_of_dependents"] = None

    # Loan Amount
    try:
        data["loan_amount"] = re.search(r"loan\s+amount.*?:\s*(\d+)", text, re.IGNORECASE).group(1)
    except:
        data["loan_amount"] = None

    # Residential Assets Value
    try:
        data["residential_assets_value"] = re.search(r"Residential\s+Assets\s+Value\s*:\s*(\d+)", text, re.IGNORECASE).group(1)
    except:
        data["residential_assets_value"] = None

    # Commercial Assets Value
    try:
        data["commercial_assets_value"] = re.search(r"Commercial\s+Assets\s+Value\s*:\s*(\d+)", text, re.IGNORECASE).group(1)
    except:
        data["commercial_assets_value"] = None

    # Education
    try:
        edu_match = re.search(r"Education\s*:\s*(Graduate|Graduated|Not\s*Graduate|Not\s*Graduated)", text, re.IGNORECASE)
        if edu_match:
            edu_value = edu_match.group(1).lower()
            if "graduate" in edu_value and "not" not in edu_value:
                data["education_graduate"] = "Yes"
            else:
                data["education_graduate"] = "No"
        else:
            data["education_graduate"] = None
    except:
        data["education_graduate"] = None

    # Self Employed
    try:
        emp_match = re.search(r"Self\s*Employed\s*:\s*(Yes|No)", text, re.IGNORECASE)
        data["self_employed_no"] = emp_match.group(1)
    except:
        data["self_employed_no"] = None

    return data


# ---------------- Loan Prediction ---------------- #
def predict_loan(data):

    if data["cibil_score"] is None:
        data["cibil_score"] = st.number_input("Enter your CIBIL Score:", min_value=300, max_value=900, value=750)
    if data["loan_term"] is None:
        data["loan_term"] = st.number_input("Enter Loan Term (months):", min_value=1, value=60)
    if data["income_annum"] is None:
        data["income_annum"] = st.number_input("Enter Annual Income:", min_value=0, value=500000)
    if data["no_of_dependents"] is None:
        data["no_of_dependents"] = st.number_input("Enter Number of Dependents:", min_value=0, value=0)
    if data["loan_amount"] is None:
        data["loan_amount"] = st.number_input("Enter loan amount:", min_value=0, value=0)
    if data["residential_assets_value"] is None:
        data["residential_assets_value"] = st.number_input("Enter Residential Assets Value:", min_value=0, value=0)
    if data["commercial_assets_value"] is None:
        data["commercial_assets_value"] = st.number_input("Enter Commercial Assets Value:", min_value=0, value=0)
    if data["education_graduate"] is None:
        edu_option = st.selectbox("Education Status:", ["Yes (Graduate)", "No (Not Graduate)"])
        data["education_graduate"] = "Yes" if "Yes" in edu_option else "No"
    if data["self_employed_no"] is None:
        emp_option = st.selectbox("Self Employed:", ["Yes", "No"])
        data["self_employed_no"] = emp_option
    
    df = pd.DataFrame([data])

    # Convert Yes/No fields to numeric
    df["self_employed_no"] = df["self_employed_no"].apply(lambda x: 1 if str(x).lower() in ["no", "0"] else 0)
    df["education_graduate"] = df["education_graduate"].apply(lambda x: 1 if str(x).lower() == "yes" else 0)

    # Convert numeric columns
    numeric_cols = [
        "cibil_score", "loan_term", "income_annum",
        "no_of_dependents", "residential_assets_value",
        "commercial_assets_value", "loan_amount"
    ]
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    # ✅ Match model’s expected order
    expected_order = [
        'cibil_score', 'loan_term', 'no_of_dependents',
        'commercial_assets_value', 'income_annum', 'loan_amount',
        'residential_assets_value', 'education_graduate', 'self_employed_no'
    ]
    df = df.reindex(columns=expected_order, fill_value=0)

    # Make prediction
    prediction = model.predict(df)[0]
    return "✅ Loan Eligible" if prediction == 1 else "❌ Loan Not Eligible"



st.set_page_config(page_title="Loan-Eligibility-Analyzer/", page_icon="🤖")
st.title("🤖 Loan Eligibility Analyzer/")
st.write("Upload your **loan application PDF**, and I’ll tell you whether your are eligible for the loan or not.")

uploaded_file = st.file_uploader("📂 Upload Loan Application PDF", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Extracting details from PDF..."):
        data = parse_pdf(uploaded_file)
        st.write("### Extracted Information")
        st.json(data)

        result = predict_loan(data)
        st.success(f"### {result}")
