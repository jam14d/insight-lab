# pages/3_Feature_Engineering_Toolkit.py
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt  # <- needed for show_hist
from lib.data import load_default_data
from lib.helpers import (
    COLOR, COLOR_DARK, stylize, safecsv, calc_table, fig_png_bytes, slugify
)

st.title("Transformation Toolkit")

sns.set_theme(style="whitegrid", rc={"axes.spines.top": False, "axes.spines.right": False, "font.size": 12})


df = st.session_state.get("df")
if df is None:
    try:
        df = load_default_data()
        st.info("Using sample dataset (load your own on Home).")
    except Exception as e:
        st.error(str(e)); st.stop()


# Choose dataset source
src = st.radio(
    "Dataset:",
    ["Use current dataset", "Load Titanic demo"],
    horizontal=True,
    key="ft_source"
)

# Use the df already loaded by app (upload/default) OR load Titanic
if src == "Load Titanic demo":
    try:
        import seaborn as sns  # ok to reuse alias
        df_ft = sns.load_dataset("titanic").drop(columns=["deck"], errors="ignore")
        st.info("Loaded Titanic from seaborn.")
    except Exception as e:
        st.warning(f"Could not load Titanic from seaborn ({e}). Falling back to current dataset.")
        df_ft = df.copy()
else:
    df_ft = df.copy()

st.dataframe(df_ft.head(), use_container_width=True)

# Column choice
num_cols = df_ft.select_dtypes(include=np.number).columns.tolist()
if not num_cols:
    st.error("No numeric columns detected in the selected dataset."); st.stop()

# Guess a nice default
def guess_metric_ft(dfx: pd.DataFrame):
    preferred = ["age", "fare", "positive area percentage"]
    cols = list(dfx.columns)
    for p in preferred:
        for c in cols:
            if isinstance(c, str) and c.strip().lower() == p:
                return c
    fallback = dfx.select_dtypes(include=np.number).columns.tolist()
    return fallback[0] if fallback else cols[0]

metric_guess_ft = guess_metric_ft(df_ft)
col = st.selectbox(
    "Numeric column",
    options=num_cols,
    index=(num_cols.index(metric_guess_ft) if metric_guess_ft in num_cols else 0),
    key="featuretool_metric"
)
s = df_ft[col].dropna().astype(float)

lib = st.radio("Use which syntax?", options=["Pandas", "NumPy"], horizontal=True, key="featuretool_lib")
show_code = st.toggle("Show the code for each action", value=False, key="featuretool_show_code")

st.divider()

# Quick buttons row
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    if st.button("Min", key="btn_min"):
        val = (s.min() if lib == "Pandas" else np.min(s))
        st.metric("Minimum", round(float(val), 6))
        if show_code:
            st.code(
f"""# Select the column as a Series
s = df['{col}']  # brackets: column selection

# Compute minimum
min_val = {"df['"+col+"'].min()" if lib=="Pandas" else "np.min(df['"+col+"'])"}
print(min_val)"""
            )

with c2:
    if st.button("Max", key="btn_max"):
        val = (s.max() if lib == "Pandas" else np.max(s))
        st.metric("Maximum", round(float(val), 6))
        if show_code:
            st.code(
f"""max_val = {"df['"+col+"'].max()" if lib=="Pandas" else "np.max(df['"+col+"'])"}"""
            )

with c3:
    if st.button("Mean", key="btn_mean"):
        val = (s.mean() if lib == "Pandas" else np.mean(s))
        st.metric("Mean", round(float(val), 6))
        if show_code:
            st.code(
f"""mean_val = {"df['"+col+"'].mean()" if lib=="Pandas" else "np.mean(df['"+col+"'])"}"""
            )

with c4:
    if st.button("Median", key="btn_median"):
        val = (s.median() if lib == "Pandas" else np.median(s))
        st.metric("Median", round(float(val), 6))
        if show_code:
            st.code(
f"""median_val = {"df['"+col+"'].median()" if lib=="Pandas" else "np.median(df['"+col+"'])"}"""
            )

with c5:
    if st.button("Std (sample)", key="btn_std"):
        # ddof=1 to match sample std
        val = (s.std(ddof=1) if lib == "Pandas" else np.std(s, ddof=1))
        st.metric("Std (sample)", round(float(val), 6))
        if show_code:
            st.code(
f"""std_val = {"df['"+col+"'].std(ddof=1)" if lib=="Pandas" else "np.std(df['"+col+"'], ddof=1)"}"""
            )

st.divider()

# Centering & Scaling helpers
st.markdown("### Centering & Scaling")
t1, t2, t3 = st.columns(3)
with t1:
    center_btn = st.button("Center (x - mean)", key="btn_center")
with t2:
    zscore_btn = st.button("Z-Score ((x - mean)/std)", key="btn_zscore")
with t3:
    minmax_btn = st.button("Min–Max Scale to [0, 1]", key="btn_minmax")

mean_val = float(s.mean()) if len(s) else np.nan
std_val = float(s.std(ddof=1)) if len(s) > 1 else np.nan
min_val = float(s.min()) if len(s) else np.nan
max_val = float(s.max()) if len(s) else np.nan
eps = 1e-12  # avoid divide-by-zero if constant column

def show_hist(series, title):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    sns.histplot(series, ax=ax, color=COLOR, edgecolor=COLOR_DARK, alpha=0.8)
    stylize(ax, title, col, "Count")
    st.pyplot(fig)
    st.download_button(
        f"Download PNG — {title}",
        data=fig_png_bytes(fig),
        file_name=f"{slugify(title)}.png",
        mime="image/png",
        key=f"dl_{slugify(title)}"
    )

if center_btn:
    centered = s - mean_val
    st.success("Centered column created in memory (not saved back).")
    st.dataframe(centered.to_frame(name=f"{col}_centered").head(), use_container_width=True)
    show_hist(centered, f"{col} — Centered (x - mean)")
    st.download_button(
        "Download CSV (centered column)",
        data=safecsv(centered.to_frame(name=f"{col}_centered")),
        file_name=f"{slugify(col)}_centered.csv",
        mime="text/csv",
        key="dl_centered_csv"
    )
    if show_code:
        if lib == "Pandas":
            st.code(
f"""s = df['{col}'].astype(float)
mean_val = s.mean()
centered = s - mean_val"""
            )
        else:
            st.code(
f"""s = df['{col}'].to_numpy(dtype=float)
mean_val = np.mean(s)
centered = s - mean_val"""
            )

if zscore_btn:
    z = (s - mean_val) / (std_val if (std_val and std_val > 0) else eps)
    st.success("Z-scored column created in memory.")
    st.dataframe(z.to_frame(name=f"{col}_zscore").head(), use_container_width=True)
    show_hist(z, f"{col} — Z-Score ((x - mean)/std)")
    st.download_button(
        "Download CSV (z-score column)",
        data=safecsv(z.to_frame(name=f"{col}_zscore")),
        file_name=f"{slugify(col)}_zscore.csv",
        mime="text/csv",
        key="dl_zscore_csv"
    )
    if show_code:
        if lib == "Pandas":
            st.code(
f"""s = df['{col}'].astype(float)
mean_val = s.mean()
std_val = s.std(ddof=1)  # sample std
z = (s - mean_val) / std_val"""
            )
        else:
            st.code(
f"""s = df['{col}'].to_numpy(dtype=float)
mean_val = np.mean(s)
std_val = np.std(s, ddof=1)  # sample std
z = (s - mean_val) / std_val"""
            )

if minmax_btn:
    rng = (max_val - min_val) if (max_val - min_val) != 0 else eps
    mm = (s - min_val) / rng
    st.success("Min–Max scaled column created in memory.")
    st.dataframe(mm.to_frame(name=f"{col}_minmax_0_1").head(), use_container_width=True)
    show_hist(mm, f"{col} — Min–Max Scaled to [0, 1]")
    st.download_button(
        "Download CSV (min–max column)",
        data=safecsv(mm.to_frame(name=f"{col}_minmax_0_1")),
        file_name=f"{slugify(col)}_minmax_0_1.csv",
        mime="text/csv",
        key="dl_minmax_csv"
    )
    if show_code:
        if lib == "Pandas":
            st.code(
f"""s = df['{col}'].astype(float)
min_val, max_val = s.min(), s.max()
mm = (s - min_val) / (max_val - min_val)"""
            )
        else:
            st.code(
f"""s = df['{col}'].to_numpy(dtype=float)
min_val, max_val = np.min(s), np.max(s)
mm = (s - min_val) / (max_val - min_val)"""
            )
