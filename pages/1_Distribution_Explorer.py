import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from lib.data import load_default_data
from lib.helpers import (
    COLOR, COLOR_DARK, stylize, apply_ylim, apply_ylim_range, apply_locator,
    safecsv, calc_table, slugify, fig_png_bytes, fname_from_title,
    DEFAULT_IGNORE_TOKENS, DEFAULT_TISSUE_TOKENS, extract_labels, cohort_sort_key,
    guess_metric, normalize_tokens
)

st.title("Distribution Explorer")
sns.set_theme(style="whitegrid", rc={"axes.spines.top": False, "axes.spines.right": False, "font.size": 12})

df = st.session_state.get("df")
if df is None:
    try:
        df = load_default_data()
        st.info("Using sample dataset (load your own on Home).")
    except Exception as e:
        st.error(str(e)); st.stop()

src = st.radio("Dataset:", ["Use current dataset", "Load Titanic demo"], horizontal=True, key="dist_source")
is_titanic = (src == "Load Titanic demo")

if is_titanic:
    try:
        df = sns.load_dataset("titanic").drop(columns=["deck"], errors="ignore")
        st.info("Loaded Titanic from seaborn.")
    except Exception as e:
        st.warning(f"Could not load Titanic ({e}). Falling back to current dataset.")


st.subheader("Preview")
st.dataframe(df.head(), use_container_width=True)

num_cols = df.select_dtypes(include=np.number).columns.tolist()
if not num_cols: st.error("No numeric columns detected."); st.stop()
metric_guess = guess_metric(df.columns) or num_cols[0]
metric_col = st.selectbox("Numeric column", options=num_cols,
                          index=(num_cols.index(metric_guess) if metric_guess in num_cols else 0),
                          key="dist_metric")
s = df[metric_col].dropna().astype(float)

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Count", int(s.count()))
c2.metric("Mean", round(float(s.mean()), 4))
c3.metric("Median", round(float(s.median()), 4))
c4.metric("Std", round(float(s.std(ddof=1)), 4) if len(s) > 1 else np.nan)
c5.metric("Skew", round(float(skew(s)), 4) if len(s) > 2 else np.nan)
c6.metric("Kurtosis (excess)", round(float(kurtosis(s)), 4) if len(s) > 3 else np.nan)
c7.metric("Min/Max", f"{round(float(s.min()),4)} / {round(float(s.max()),4)}")

title_hist = f"Distribution of {metric_col} (overall)"
fig_h, ax_h = plt.subplots(figsize=(8,4.5))
sns.histplot(s, ax=ax_h, color=COLOR, edgecolor=COLOR_DARK, alpha=0.75)
try: sns.kdeplot(s, ax=ax_h, color=COLOR_DARK, linewidth=2)
except Exception: pass
stylize(ax_h, title_hist, metric_col, "Count")
st.pyplot(fig_h)
st.download_button("Download PNG (Overall histogram)", data=fig_png_bytes(fig_h), file_name=fname_from_title(title_hist), mime="image/png")

title_box_overall = f"Box plot of {metric_col} (overall)"
fig_b, ax_b = plt.subplots(figsize=(8,2.8))
sns.boxplot(x=s, ax=ax_b, color=COLOR, fliersize=3, linewidth=1)
stylize(ax_b, title_box_overall, metric_col, None)
st.pyplot(fig_b)
st.download_button("Download PNG (Overall box)", data=fig_png_bytes(fig_b), file_name=fname_from_title(title_box_overall), mime="image/png")

obj_cols = df.select_dtypes(include="object").columns.tolist()
if obj_cols:
    if is_titanic:
        cohort_col = st.selectbox("Cohort column", options=obj_cols, index=(obj_cols.index("class") if "class" in obj_cols else 0), key="dist_cohort_col")
        id_col = st.selectbox("ID column", options=obj_cols, index=(obj_cols.index("who") if "who" in obj_cols else 0), key="dist_id_col")
        filename_col = id_col
        w = df.copy()
        w["Cohort"] = w[cohort_col].astype(str)
        w["Group_Full"] = w[id_col].astype(str)
        w["Tissue"] = np.nan
        base = w[[filename_col, metric_col, "Group_Full", "Cohort", "Tissue"]].dropna(subset=[metric_col]).copy()
        selected_cohorts = st.multiselect("Filter cohorts", options=sorted(base["Cohort"].dropna().unique(), key=cohort_sort_key),
                                          default=sorted(base["Cohort"].dropna().unique(), key=cohort_sort_key), key="dist_cohort_sel_titanic")
        if selected_cohorts: base = base[base["Cohort"].isin(selected_cohorts)]
    else:
        filename_col = st.selectbox("Filename column", options=obj_cols, index=(obj_cols.index("Name") if "Name" in obj_cols else 0), key="dist_filename")
        pre = df.copy()
        parsed0 = pre[filename_col].astype(str).apply(lambda x: extract_labels(x, ignore_tokens=set(), tissue_tokens=set()))
        pre["Cohort"] = parsed0.apply(lambda t: t[1])
        cohort_counts = pre["Cohort"].value_counts(dropna=True)
        all_cohorts = sorted([c for c in cohort_counts.index.tolist() if pd.notna(c)], key=cohort_sort_key)
        selected_cohorts = st.multiselect("Filter cohorts", options=all_cohorts, default=all_cohorts, key="dist_cohort_sel")
        tissue_text = st.text_input("Key tokens (comma-separated)", value=",".join(sorted(DEFAULT_TISSUE_TOKENS)), key="dist_tissue")
        ignore_text = st.text_input("Ignore tokens (comma-separated)", value=",".join(sorted(DEFAULT_IGNORE_TOKENS)), key="dist_ignore")
        tissue_tokens = normalize_tokens(tissue_text.split(","))
        ignore_tokens = normalize_tokens(ignore_text.split(","))
        w = df.copy()
        parsed = w[filename_col].astype(str).apply(lambda x: extract_labels(x, ignore_tokens, tissue_tokens))
        w["Group_Full"] = parsed.apply(lambda t: t[0]); w["Cohort"] = parsed.apply(lambda t: t[1]); w["Tissue"] = parsed.apply(lambda t: t[2])
        base = w[[filename_col, metric_col, "Group_Full", "Cohort", "Tissue"]].dropna(subset=[metric_col]).copy()
        if selected_cohorts: base = base[base["Cohort"].isin(selected_cohorts)]

    if not base.empty:
        order_coh = sorted(base["Cohort"].dropna().unique(), key=cohort_sort_key)
        title_cohort_box = f"{metric_col} by Cohort"
        fig_c, ax_c = plt.subplots(figsize=(10, 4.8))
        sns.boxplot(data=base, x="Cohort", y=metric_col, order=order_coh, ax=ax_c, color=COLOR, fliersize=2, linewidth=1)
        stylize(ax_c, title_cohort_box, "Cohort", metric_col)

        ymin_cohort = st.number_input("Y min (Cohort box)", value=0.0, step=0.1, key="ymin_cohort_box")
        ymax_cohort = st.number_input("Y max (Cohort box)", value=0.0, step=0.1, key="ymax_cohort_box")
        tick_step_cohort = st.number_input("Y tick step (Cohort box, 0 = auto)", min_value=0.0, value=0.0, step=0.1, key="ytick_cohort_box")
        apply_ylim_range(ax_c, data_min=base[metric_col].min(), data_max=base[metric_col].max(),
                         min_override=ymin_cohort, max_override=ymax_cohort if ymax_cohort > 0 else None)
        apply_locator(ax_c, tick_step_cohort)

        st.pyplot(fig_c)
        st.download_button("Download PNG (Cohort box)", data=fig_png_bytes(fig_c), file_name=fname_from_title(title_cohort_box), mime="image/png")

        cohort_opts = order_coh
        if cohort_opts:
            single_cohort = st.selectbox("Choose a cohort", options=cohort_opts, key="dist_single_cohort")
            within = base[base["Cohort"] == single_cohort].dropna(subset=["Group_Full"]).copy()
            if not within.empty:
                stats_per_id = within.groupby("Group_Full")[metric_col].agg(["median","mean"]).reset_index()
                order_ids = stats_per_id.sort_values("median", ascending=False)["Group_Full"].tolist()
                fig_w = max(8, min(22, 0.55*len(order_ids)))
                title_wcoh = f"{single_cohort}: {metric_col} by Group ID"
                fig_wcoh, ax_wcoh = plt.subplots(figsize=(fig_w, 4.8))
                sns.boxplot(data=within, x="Group_Full", y=metric_col, order=order_ids, ax=ax_wcoh, color=COLOR, fliersize=2, linewidth=1)
                stylize(ax_wcoh, title_wcoh, "Group ID", metric_col)
                for lbl in ax_wcoh.get_xticklabels(): lbl.set_rotation(28); lbl.set_ha("right")
                st.pyplot(fig_wcoh)
                st.download_button("Download PNG (Within-cohort box)", data=fig_png_bytes(fig_wcoh), file_name=fname_from_title(title_wcoh), mime="image/png")
