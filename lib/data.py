# lib/data.py
import os
import pandas as pd
import streamlit as st

@st.cache_data
def load_default_data():
    candidates = ["insightlab_sample20.csv", "insightlab_sample.csv"]
    for p in candidates:
        if os.path.exists(p):
            df_local = pd.read_csv(p)
            if "Pixel Count" in df_local.columns:
                df_local = df_local.drop(columns=["Pixel Count"])
            return df_local
    raise FileNotFoundError(
        "Place 'insightlab_sample20.csv' or 'insightlab_sample.csv' next to this app, or upload a CSV on Home."
    )
