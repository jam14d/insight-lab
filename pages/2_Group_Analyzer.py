import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lib.data import load_default_data
from lib.helpers import (
    COLOR, COLOR_DARK, stylize, apply_ylim, apply_locator, safecsv, calc_table,
    fig_png_bytes, fname_from_title, DEFAULT_IGNORE_TOKENS, DEFAULT_TISSUE_TOKENS,
    extract_labels, normalize_tokens, cohort_sort_key, guess_metric
)

st.title("Group Analyzer")
sns.set_theme(style="whitegrid", rc={"axes.spines.top": False, "axes.spines.right": False, "font.size": 12})

df = st.session_state.get("df")
if df is None:
    try:
        df = load_default_data()
        st.info("Using sample dataset (load your own on Home).")
    except Exception as e:
        st.error(str(e)); st.stop()

src = st.radio("Dataset:", ["Use current dataset", "Load Titanic demo"], horizontal=True, key="group_source")
is_titanic = (src == "Load Titanic demo")
if is_titanic:
    try:
        import seaborn as sns
        df = sns.load_dataset("titanic").drop(columns=["deck"], errors="ignore")
        st.info("Loaded Titanic from seaborn.")
    except Exception as e:
        st.warning(f"Could not load Titanic ({e}). Falling back to current dataset.")

st.dataframe(df.head(), use_container_width=True)

num_cols = df.select_dtypes(include=np.number).columns.tolist()
if not num_cols: st.error("No numeric columns found."); st.stop()
metric_guess = guess_metric(df.columns) or num_cols[0]
metric_col = st.selectbox("Metric (y-axis)", options=num_cols, index=(num_cols.index(metric_guess) if metric_guess in num_cols else 0), key="group_metric")

tick_step = st.number_input("Y tick step (0 = auto)", min_value=0.0, value=0.0, step=0.1)

if is_titanic:
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    if not obj_cols: st.error("No categorical columns found."); st.stop()
    cohort_col = st.selectbox("Cohort column", options=obj_cols, index=(obj_cols.index("class") if "class" in obj_cols else 0))
    id_col = st.selectbox("ID column", options=obj_cols, index=(obj_cols.index("who") if "who" in obj_cols else 0))
    w = df.copy()
    w["Cohort"] = w[cohort_col].astype(str)
    w["Group_Full"] = w[id_col].astype(str)
    w["Tissue"] = np.nan
    base = w[[id_col, metric_col, "Group_Full", "Cohort", "Tissue"]].dropna(subset=[metric_col]).copy()
else:
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    if not obj_cols: st.error("No text/filename column found."); st.stop()
    filename_col = st.selectbox("Filename column", options=obj_cols, index=(obj_cols.index("Name") if "Name" in obj_cols else 0))
    tissue_text = st.text_input("Key tokens", value=",".join(sorted(DEFAULT_TISSUE_TOKENS)))
    ignore_text = st.text_input("Ignore tokens", value=",".join(sorted(DEFAULT_IGNORE_TOKENS)))
    drop_ignored = st.checkbox("Drop rows with ignored tokens", value=False)
    tissue_tokens = normalize_tokens(tissue_text.split(","))
    ignore_tokens = normalize_tokens(ignore_text.split(","))
    w = df.copy()
    if drop_ignored and ignore_tokens:
        w = w[~w[filename_col].astype(str).str.upper().apply(lambda s: any(tok in s for tok in ignore_tokens))].copy()
    parsed = w[filename_col].astype(str).apply(lambda x: extract_labels(x, ignore_tokens, tissue_tokens))
    w["Group_Full"] = parsed.apply(lambda t: t[0])
    w["Cohort"] = parsed.apply(lambda t: t[1])
    w["Tissue"] = parsed.apply(lambda t: t[2])
    base = w[[filename_col, metric_col, "Group_Full", "Cohort", "Tissue"]].dropna(subset=[metric_col]).copy()

if base.empty: st.warning("No rows available."); st.stop()
# Plot 1 — Compare cohorts
st.markdown("### Plot 1 — Compare cohorts")
agg_mode = st.radio("Aggregation:", options=["mean", "median"], index=0, horizontal=True, key="agg_cohort")
order_coh = sorted(base["Cohort"].dropna().unique(), key=cohort_sort_key)

fig1, ax1 = plt.subplots(figsize=(10, 5))
if agg_mode == "mean":
    sns.barplot(
        data=base, x="Cohort", y=metric_col, estimator=np.mean, ci=95,
        ax=ax1, color=COLOR, edgecolor=COLOR_DARK, order=order_coh
    )
    title1 = f"{metric_col} by Cohort (mean ± 95% CI)"
    stylize(ax1, title1, "Cohort", metric_col)
else:
    sns.boxplot(data=base, x="Cohort", y=metric_col, ax=ax1, color=COLOR, fliersize=2, linewidth=1, order=order_coh)
    sns.stripplot(data=base, x="Cohort", y=metric_col, ax=ax1, color=COLOR_DARK, alpha=0.6, jitter=0.2, size=3, order=order_coh)
    title1 = f"{metric_col} by Cohort "
    stylize(ax1, title1, "Cohort", metric_col)

y_max_1 = st.number_input("Y max (Plot 1)", min_value=0.0, value=0.0, step=0.1, key="ymax1_group")
apply_ylim(ax1, base[metric_col].max(), y_max_1)
apply_locator(ax1, tick_step)
st.pyplot(fig1)
st.download_button("Download PNG (Plot 1)", data=fig_png_bytes(fig1), file_name=fname_from_title(title1), mime="image/png")

with st.expander("Rows used (cohort plot)"):
    st.dataframe(base[[filename_col, "Cohort", "Tissue", metric_col]].reset_index(drop=True))
    st.download_button("Download rows (CSV)", data=safecsv(base[[filename_col, "Cohort", "Tissue", metric_col]]), file_name="cohort_plot_rows.csv", mime="text/csv")

with st.expander("Show the math (cohort aggregation)"):
    math_cohort = base.groupby("Cohort")[metric_col].apply(calc_table).reset_index().drop(columns=["level_1"])
    st.dataframe(math_cohort)
    st.download_button("Download cohort calculations (CSV)", data=safecsv(math_cohort), file_name="cohort_calculations.csv", mime="text/csv")

# Plot 2 — Within-cohort (per full ID) 
st.markdown("### Plot 2 — Within-cohort (per full ID)")
cohort_opts = sorted(base["Cohort"].dropna().unique().tolist(), key=cohort_sort_key)
single_cohort = st.selectbox("Choose a cohort", options=cohort_opts)
within = base[base["Cohort"] == single_cohort].dropna(subset=["Group_Full"]).copy()

if within.empty:
    st.info("No rows for the selected cohort.")
else:
    sort_mode = st.selectbox("Sort IDs by", ["ID (A→Z)", "median (desc)", "median (asc)", "mean (desc)", "mean (asc)"], index=1)
    point_size = st.slider("Point size", 3, 12, 6, step=1)
    point_alpha = st.slider("Point transparency", 0.2, 1.0, 0.75, step=0.05)
    show_ref = st.multiselect("Reference lines", ["cohort mean", "cohort median"], default=["cohort median"])

    stats_per_id = within.groupby("Group_Full")[metric_col].agg(["median", "mean"]).reset_index()
    if sort_mode == "ID (A→Z)":
        order_ids = sorted(within["Group_Full"].unique().tolist())
    elif "median" in sort_mode:
        order_ids = stats_per_id.sort_values("median", ascending=("asc" in sort_mode))["Group_Full"].tolist()
    else:
        order_ids = stats_per_id.sort_values("mean", ascending=("asc" in sort_mode))["Group_Full"].tolist()

    title2 = f"{single_cohort}: {metric_col}"
    fig_w = max(8, min(22, 0.55 * len(order_ids)))
    fig2, ax2 = plt.subplots(figsize=(fig_w, 5))

    common_kwargs = dict(data=within, x="Group_Full", y=metric_col, order=order_ids, ax=ax2,
                         linewidth=0.4, edgecolor="black", alpha=point_alpha)
    sns.swarmplot(size=point_size, color=COLOR_DARK, **common_kwargs)

    if "cohort mean" in show_ref:
        m = float(within[metric_col].mean())
        if not np.isnan(m): ax2.axhline(m, linestyle="--", linewidth=1, alpha=0.6)
    if "cohort median" in show_ref:
        md = float(within[metric_col].median())
        if not np.isnan(md): ax2.axhline(md, linestyle=":", linewidth=1, alpha=0.8)

    stylize(ax2, title2, "Group ID", metric_col)
    y_max_2 = st.number_input("Y max (Plot 2)", min_value=0.0, value=0.0, step=0.1, key="ymax2_group")
    apply_ylim(ax2, within[metric_col].max(), y_max_2)
    apply_locator(ax2, tick_step)
    for lbl in ax2.get_xticklabels():
        lbl.set_rotation(28); lbl.set_ha("right")
    st.pyplot(fig2)
    st.download_button("Download PNG (Plot 2)", data=fig_png_bytes(fig2), file_name=fname_from_title(title2), mime="image/png")

    with st.expander("Rows used (within-cohort plot)"):
        st.dataframe(within[[filename_col, "Cohort", "Group_Full", "Tissue", metric_col]].reset_index(drop=True))
        st.download_button("Download rows (CSV)", data=safecsv(within[[filename_col, "Cohort", "Group_Full", "Tissue", metric_col]]), file_name="within_cohort_rows.csv", mime="text/csv")

    with st.expander("Show the math (within-cohort)"):
        math_within = within.groupby(["Group_Full"])[metric_col].apply(calc_table).reset_index().drop(columns=["level_1"])
        st.dataframe(math_within)
        st.download_button("Download within-cohort calculations (CSV)", data=safecsv(math_within), file_name="within_cohort_calculations.csv", mime="text/csv")

# Per-image bars (filtered) 
st.markdown("### Per-image bars (filtered)")
per_img = base.copy()
all_ids = sorted(per_img["Group_Full"].dropna().unique().tolist())
ids_for_images = st.multiselect("Filter by full IDs", options=all_ids, default=all_ids[:1] if all_ids else [])
if ids_for_images:
    per_img = per_img[per_img["Group_Full"].isin(ids_for_images)]
if per_img.empty:
    st.info("No rows match filters.")
else:
    fig_d, ax_d = plt.subplots(figsize=(max(8, min(18, 0.5 * len(per_img))), 5))
    plot_df2 = per_img.copy()
    plot_df2["Label"] = plot_df2["Group_Full"].fillna(plot_df2[filename_col].astype(str))
    sns.barplot(data=plot_df2, x="Label", y=metric_col, ax=ax_d, edgecolor=COLOR_DARK, color=COLOR)
    title_d = f"Per-Image: {metric_col}"
    stylize(ax_d, title_d, "Image", metric_col)
    y_max_d = st.number_input("Y max (Per-image)", min_value=0.0, value=0.0, step=0.1, key="ymax_d2_group")
    apply_ylim(ax_d, plot_df2[metric_col].max(), y_max_d)
    apply_locator(ax_d, tick_step)
    for lbl in ax_d.get_xticklabels():
        lbl.set_rotation(35); lbl.set_ha("right")
    plt.tight_layout()
    st.pyplot(fig_d)
    st.download_button("Download PNG (Per-image)", data=fig_png_bytes(fig_d), file_name=fname_from_title(title_d), mime="image/png")

    with st.expander("Rows used (per-image)"):
        st.dataframe(plot_df2[[filename_col, "Label", "Cohort", "Tissue", metric_col]].reset_index(drop=True))
        st.download_button("Download rows (CSV)", data=safecsv(plot_df2[[filename_col, "Label", "Cohort", "Tissue", metric_col]]), file_name="per_image_rows.csv", mime="text/csv")

    with st.expander("Show the math (per-image subset)"):
        math_img = plot_df2.groupby(["Label"])[metric_col].apply(calc_table).reset_index().drop(columns=["level_1"])
        st.dataframe(math_img)
        st.download_button("Download per-image calculations (CSV)", data=safecsv(math_img), file_name="per_image_calculations.csv", mime="text/csv")
