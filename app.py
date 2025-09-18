# app.py / home page
import streamlit as st
import pandas as pd
import seaborn as sns
from lib.data import load_default_data

st.set_page_config(page_title="InsightLab", layout="wide")
st.title("InsightLab")

sns.set_theme(style="whitegrid", rc={"axes.spines.top": False, "axes.spines.right": False, "font.size": 12})

st.sidebar.success("Use the sidebar to switch pages.")

#st.subheader("Dataset")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if "Pixel Count" in df.columns:
            df = df.drop(columns=["Pixel Count"])
        st.session_state["df"] = df
        st.success("Loaded your dataset.")
    except Exception as e:
        st.error(f"Could not read the uploaded file: {e}")
elif "df" not in st.session_state:
    # fallback to sample data on first visit
    try:
        st.session_state["df"] = load_default_data()
        st.info("Loaded sample dataset.")
    except Exception as e:
        st.error(str(e))

if "df" in st.session_state:
    st.dataframe(st.session_state["df"].head(500), use_container_width=True)

# with st.expander("Quick Guide"):
#     st.markdown("""
#     ### Distributions
#     - **Histogram**: overall shape of your metric …
#     ### Cohort comparison
#     - **Box plots by Cohort** …
#     ### Within-cohort
#     - **Swarm/box plots per Group ID** …
#     ### Per-image bars
#     - **Bar chart per image** …
#     ### Parsing controls
#     - **Filter cohorts**, **Key tokens**, **Ignore tokens**
#     ### Statistics shown
#     - **n**, **Mean/Median**, **Std/SEM**, **95% CI**, **Skew**, **Excess kurtosis**
#     ### Axis & export
#     - Adjust Y-limits/ticks and export figures/tables.
#     """)

st.divider()
st.subheader("Pages")
# links from the home page to each page 
st.page_link("pages/1_Distribution_Explorer.py", label="Distribution Explorer")
st.page_link("pages/2_Group_Comparisons.py", label="Group Comparisons")
st.page_link("pages/3_Feature_Engineering_Toolkit.py", label="Feature Engineering Toolkit")
