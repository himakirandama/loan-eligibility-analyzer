import streamlit as st
import re
import pandas as pd
import joblib
from PyPDF2 import PdfReader

# Load trained model
model = joblib.load("loan_approval_model.pkl")

# ---------------- PDF Parsing ---------------- #
def parse_pdf(file):
    """
    Extract required fields from uploaded PDF (BytesIO object)
    """
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
        data["cibil_score"] = 0

    # Loan Term
    try:
        data["loan_term"] = re.search(r"Loan term.*?:\s*(\d+)", text, re.IGNORECASE).group(1)
    except:
        data["loan_term"] = 0

    # Income
    try:
        data["income_annum"] = re.search(r"Income.*?:\s*(\d+)", text, re.IGNORECASE).group(1)
    except:
        data["income_annum"] = 0

    # Number of Dependents
    try:
        data["no_of_dependents"] = re.search(r"Dependents.*?:\s*(\d+)", text, re.IGNORECASE).group(1)
    except:
        data["no_of_dependents"] = 0

    # Residential Assets Value
    try:
        data["residential_assets_value"] = re.search(r"Residential\s+Assets\s+Value\s*:\s*(\d+)", text, re.IGNORECASE).group(1)
    except:
        data["residential_assets_value"] = 0

    # Commercial Assets Value
    try:
        data["commercial_assets_value"] = re.search(r"Commercial\s+Assets\s+Value\s*:\s*(\d+)", text, re.IGNORECASE).group(1)
    except:
        data["commercial_assets_value"] = 0

    # Education: Graduate/Graduated = Yes, Not Graduate/Not Graduated = No
    try:
        edu_match = re.search(r"Education\s*:\s*(Graduate|Graduated|Not\s*Graduate|Not\s*Graduated)", text, re.IGNORECASE)
        if edu_match:
            edu_value = edu_match.group(1).lower()
            if "graduate" in edu_value and "not" not in edu_value:
                data["education_graduate"] = "Yes"
            else:
                data["education_graduate"] = "No"
        else:
            data["education_graduate"] = "No"
    except:
        data["education_graduate"] = "No"

    # Self Employed
    try:
        emp_match = re.search(r"Self\s*Employed\s*:\s*(Yes|No)", text, re.IGNORECASE)
        data["self_employed_no"] = emp_match.group(1)
    except:
        data["self_employed_no"] = "No"

    return data

# ---------------- Loan Prediction ---------------- #
def predict_loan(data):
    df = pd.DataFrame([data])

    # Convert Yes/No fields to numeric
    df["self_employed_no"] = df["self_employed_no"].apply(lambda x: 1 if str(x).lower() in ["no", "0"] else 0)
    df["education_graduate"] = df["education_graduate"].apply(lambda x: 1 if str(x).lower() == "yes" else 0)

    # Convert numeric columns
    numeric_cols = [
        "cibil_score", "loan_term", "income_annum",
        "no_of_dependents", "residential_assets_value", "commercial_assets_value","loan_amount"  
    ]
    numeric_cols = [col for col in numeric_cols if col in df.columns]

    if numeric_cols:  # Apply conversion only if there are valid numeric columns
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    expected_order = ['cibil_score', 'loan_term', 'self_employed_no', 'income_annum', 'no_of_dependents', 'residential_assets_value', 'commercial_assets_value', 'education_graduate']
    df = df[expected_order]

    # Make prediction
    prediction = model.predict(df)[0]
    return "✅ Loan Approved" if prediction == 1 else "❌ Loan Rejected"


st.set_page_config(page_title="Loan Approval Chatbot", page_icon="🤖")
st.title("🤖 Loan Approval Chatbot")
st.write("Upload your **loan application PDF**, and I’ll tell you whether your loan is approved or rejected.")

uploaded_file = st.file_uploader("📂 Upload Loan Application PDF", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Extracting details from PDF..."):
        data = parse_pdf(uploaded_file)
        st.write("### Extracted Information")
        st.json(data)

        result = predict_loan(data)
        st.success(f"### {result}")
