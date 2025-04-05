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

import pickle
import numpy as np
import os
from utils import preprocess_user_data

model_dir = "models"
feature_file = "feature_names.pkl"

with open(feature_file, "rb") as f:
    selected_genes = pickle.load(f)  # list of 50 genes

def predict_for_all_drugs(user_data):
    input_vector, matched_count = preprocess_user_data(user_data, selected_genes)

    results = {}
    for model_file in os.listdir(model_dir):
        drug = model_file.replace("_model.pkl", "")
        with open(os.path.join(model_dir, model_file), "rb") as f:
            model = pickle.load(f)
        prediction = model.predict([input_vector])[0]
        results[drug] = {
            "prediction": prediction,
            "matched": matched_count
        }
    return results

def preprocess_user_data(user_df, selected_genes):
    gene_expr = dict(zip(user_df.iloc[:, 0], user_df.iloc[:, 1]))  # {gene: expression}

    input_vector = []
    matched_count = 0

    for gene in selected_genes:
        if gene in gene_expr:
            input_vector.append(gene_expr[gene])
            matched_count += 1
        else:
            input_vector.append(0)  # or average value, or np.nan

    return input_vector, matched_count
