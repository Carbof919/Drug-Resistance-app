import streamlit as st
import pandas as pd
from predict import predict_for_all_drugs

st.set_page_config(page_title="Drug Resistance Predictor", layout="centered")

st.title("🎯 Drug Resistance Predictor for HER2+ Breast Cancer")

uploaded_file = st.file_uploader("Test gene expression file.csv", type=["csv"])

if uploaded_file:
    user_data = pd.read_csv(uploaded_file)
    results = predict_for_all_drugs(user_data)

    st.subheader("📊 Prediction Results")
    for drug, result in results.items():
        st.markdown(f"### 💊 {drug}")
        st.write(f"**Prediction:** {'Responder ✅' if result['prediction']==1 else 'Non-responder ❌'}")
        st.write(f"Matched Genes: {result['matched']}/50")
        if result["matched"] < 30:
            st.warning("⚠️ Low number of matched genes. Prediction may be inaccurate.")
