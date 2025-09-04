import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from typing import Set
import re, io
from scipy.stats import skew, kurtosis

st.set_page_config(page_title="InsightLab: Distributions, Variation & Skewness", layout="wide")
st.title("InsightLab: Distributions, Variation & Skewness")

COLOR = "#59BBBB"; COLOR_DARK = "#59BBBB"
sns.set_theme(style="whitegrid", rc={"axes.spines.top": False, "axes.spines.right": False, "font.size": 12})

def stylize(ax, title=None, xlabel=None, ylabel=None):
    if title: ax.set_title(title, weight="600")
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    return ax

def apply_ylim(ax, data_max, override):
    ax.set_ylim(0, override if override and override > 0 else (float(data_max) * 1.10 if data_max is not None else 1.0))

def apply_locator(ax, step):
    if step and step > 0: ax.yaxis.set_major_locator(MultipleLocator(step))

def safecsv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def calc_table(series: pd.Series) -> pd.DataFrame:
    s = series.dropna().astype(float)
    n = s.size
    mean = s.mean() if n else np.nan
    median = s.median() if n else np.nan
    std = s.std(ddof=1) if n > 1 else np.nan
    sem = std / np.sqrt(n) if n > 1 else np.nan
    ci = 1.96 * sem if n > 1 else np.nan
    return pd.DataFrame({"n":[n],"mean":[mean],"median":[median],"std":[std],"sem":[sem],"ci95":[ci],
                         "skew":[skew(s) if n>2 else np.nan],
                         "kurtosis(excess)":[kurtosis(s) if n>3 else np.nan]})

def mark_outliers_iqr(df: pd.DataFrame, value_col: str, group_cols: list) -> pd.DataFrame:
    def _flag(g):
        s = g[value_col].astype(float)
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
        return pd.Series((s < low) | (s > high), index=g.index)
    if not group_cols:
        df["_outlier"] = _flag(df); return df
    df["_outlier"] = df.groupby(group_cols, dropna=False, observed=True).apply(lambda g: _flag(g)).reset_index(level=list(range(len(group_cols))), drop=True)
    return df

def slugify(s): return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_").lower()
def fig_png_bytes(fig, dpi=200):
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=dpi, bbox_inches="tight"); b.seek(0); return b
def fname_from_title(title, ext="png"): return f"{slugify(title)}.{ext}"

st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Go to", ["Distributions, Variation & Skewness", "Metric by Group (Box/Bar Plots)"])
qc_mode = st.sidebar.toggle("QC mode", value=True)

@st.cache_data
def load_default_data():
    url = "https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/imdb_1000.csv"
    return pd.read_csv(url)

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the uploaded file: {e}")
        df = load_default_data()
else:
    df = load_default_data()

def guess_metric(cols):
    for c in cols:
        if isinstance(c, str) and c.strip().lower() == "positive area percentage": return c
    for c in cols:
        if isinstance(c, str) and "positive" in c.lower() and "percent" in c.lower(): return c
    return None

DEFAULT_IGNORE_TOKENS = {"S1", "S2", "S3",}
DEFAULT_TISSUE_TOKENS = {"BLAH", "BLOOP"}
GROUP_RE = re.compile(r"(?i)\bG\d+(?:-\d+)?\b")
PAS_RE = re.compile(r"(?i)\bPAS_(WT\d+|\d{2,4}(?:-\d+)?)\b")

def normalize_tokens(tokens): return {t.strip().upper() for t in tokens if isinstance(t, str) and t.strip()}
def split_tokens(name): return [t for t in re.split(r"[_.]", str(name)) if t]

def extract_labels(name, ignore_tokens: Set[str], tissue_tokens: Set[str]):
    if not isinstance(name, str) or not name: return None, None, None
    upper = name.upper(); tokens = split_tokens(name); tissue = None
    for tok in reversed(tokens[:-1]):
        t = tok.upper()
        if t in tissue_tokens: tissue = t; break
    m = PAS_RE.search(upper)
    if m:
        full = m.group(1).upper(); cohort = full.split("-")[0]; return full, cohort, tissue
    for tok in tokens:
        t = tok.upper()
        if t in ignore_tokens or t in tissue_tokens: continue
        if GROUP_RE.fullmatch(t):
            full = t; cohort = full.split("-")[0]; return full, cohort, tissue
        if t.isdigit() and 2 <= len(t) <= 4: return t, t, tissue
    m2 = re.search(r"\b(\d{2,4})(?:-\d+)?\b", upper)
    if m2:
        num = m2.group(1); return num, num, tissue
    return None, None, tissue

def cohort_sort_key(c):
    m = re.match(r"(?i)G(\d+)", str(c))
    if m: return (0, int(m.group(1)), str(c))
    if str(c).isdigit(): return (1, int(c), str(c))
    return (2, 10**9, str(c))

# Page 1
if page == "Distributions, Variation & Skewness":
    st.subheader("Preview")
    st.dataframe(df.head())

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        st.error("No numeric columns detected."); st.stop()
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
        filename_col = st.selectbox("Filename column", options=obj_cols,
                                    index=(obj_cols.index("Name") if "Name" in obj_cols else 0),
                                    key="dist_filename")
        ignore_text = st.text_input("Ignore tokens (comma-separated)", value=",".join(sorted(DEFAULT_IGNORE_TOKENS)), key="dist_ignore")
        tissue_text = st.text_input("Tissue tokens (comma-separated)", value=",".join(sorted(DEFAULT_TISSUE_TOKENS)), key="dist_tissue")
        ignore_tokens = normalize_tokens(ignore_text.split(","))
        tissue_tokens = normalize_tokens(tissue_text.split(","))
        use_tissue = st.sidebar.toggle("Use tissue as a factor (distributions page)", value=False, key="dist_use_tissue")

        w = df.copy()
        parsed = w[filename_col].astype(str).apply(lambda x: extract_labels(x, ignore_tokens, tissue_tokens))
        w["Group_Full"] = parsed.apply(lambda t: t[0]); w["Cohort"] = parsed.apply(lambda t: t[1]); w["Tissue"] = parsed.apply(lambda t: t[2])
        base = w[[filename_col, metric_col, "Group_Full", "Cohort", "Tissue"]].dropna(subset=[metric_col]).copy()

        cohort_counts = base["Cohort"].value_counts(dropna=True)
        all_cohorts = sorted([c for c in cohort_counts.index.tolist() if pd.notna(c)], key=cohort_sort_key)
        selected_cohorts = st.multiselect("Filter cohorts", options=all_cohorts, default=all_cohorts, key="dist_cohort_sel")
        if selected_cohorts: base = base[base["Cohort"].isin(selected_cohorts)]

        if not base.empty:
            order_coh = sorted(base["Cohort"].dropna().unique(), key=cohort_sort_key)
            title_cohort_box = f"{metric_col} by Cohort "
            fig_c, ax_c = plt.subplots(figsize=(10, 4.8))
            if use_tissue and base["Tissue"].notna().any():
                sns.boxplot(data=base, x="Cohort", y=metric_col, hue="Tissue", order=order_coh, ax=ax_c, fliersize=2, linewidth=1)
                ax_c.legend(title="Tissue", frameon=True)
            else:
                sns.boxplot(data=base, x="Cohort", y=metric_col, order=order_coh, ax=ax_c, color=COLOR, fliersize=2, linewidth=1)
            stylize(ax_c, title_cohort_box, "Cohort", metric_col)
            st.pyplot(fig_c)
            st.download_button("Download PNG (Cohort box)", data=fig_png_bytes(fig_c), file_name=fname_from_title(title_cohort_box), mime="image/png")

            cohort_opts = order_coh
            if cohort_opts:
                single_cohort = st.selectbox("Choose a cohort", options=cohort_opts, key="dist_single_cohort")
                within = base[base["Cohort"] == single_cohort].dropna(subset=["Group_Full"]).copy()
                if qc_mode:
                    grp_cols2 = ["Group_Full","Tissue"] if (use_tissue and within["Tissue"].notna().any()) else ["Group_Full"]
                    within = mark_outliers_iqr(within, metric_col, grp_cols2)
                if not within.empty:
                    stats_per_id = within.groupby("Group_Full")[metric_col].agg(["median","mean"]).reset_index()
                    order_ids = stats_per_id.sort_values("median", ascending=False)["Group_Full"].tolist()
                    fig_w = max(8, min(22, 0.55*len(order_ids)))
                    title_wcoh = f"{single_cohort}: {metric_col} by Group ID"
                    fig_wcoh, ax_wcoh = plt.subplots(figsize=(fig_w, 4.8))
                    if use_tissue and within["Tissue"].notna().any():
                        sns.boxplot(data=within, x="Group_Full", y=metric_col, hue="Tissue", order=order_ids, ax=ax_wcoh, fliersize=2, linewidth=1)
                        ax_wcoh.legend(title="Tissue", frameon=True)
                    else:
                        sns.boxplot(data=within, x="Group_Full", y=metric_col, order=order_ids, ax=ax_wcoh, color=COLOR, fliersize=2, linewidth=1)
                    stylize(ax_wcoh, title_wcoh, "Group ID", metric_col)
                    for lbl in ax_wcoh.get_xticklabels(): lbl.set_rotation(28); lbl.set_ha("right")
                    st.pyplot(fig_wcoh)
                    st.download_button("Download PNG (Within-cohort box)", data=fig_png_bytes(fig_wcoh), file_name=fname_from_title(title_wcoh), mime="image/png")

# Page 2
else:
    st.subheader("Metric by Group")

    obj_cols = df.select_dtypes(include="object").columns.tolist()
    if not obj_cols:
        st.error("No text/filename column found."); st.stop()
    filename_col = st.selectbox("Filename column", options=obj_cols, index=(obj_cols.index("Name") if "Name" in obj_cols else 0))

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        st.error("No numeric columns found."); st.stop()
    metric_guess = guess_metric(df.columns) or num_cols[0]
    metric_col = st.selectbox("Metric (y-axis)", options=num_cols, index=(num_cols.index(metric_guess) if metric_guess in num_cols else 0))

    st.markdown("Y ticks")
    tick_step = st.number_input("Y tick step (0 = auto)", min_value=0.0, value=0.0, step=0.1)

    st.markdown("Parsing / Filtering")
    ignore_text = st.text_input("Ignore tokens (comma-separated)", value=",".join(sorted(DEFAULT_IGNORE_TOKENS)))
    tissue_text = st.text_input("Tissue tokens (comma-separated)", value=",".join(sorted(DEFAULT_TISSUE_TOKENS)))
    ignore_tokens = normalize_tokens(ignore_text.split(","))
    tissue_tokens = normalize_tokens(tissue_text.split(","))
    drop_ignored = st.checkbox("Drop rows containing any ignored tokens", value=False)

    w = df.copy()
    if drop_ignored:
        w = w[~w[filename_col].astype(str).str.upper().apply(lambda s: any(tok in s for tok in ignore_tokens))].copy()

    parsed = w[filename_col].astype(str).apply(lambda x: extract_labels(x, ignore_tokens, tissue_tokens))
    w["Group_Full"] = parsed.apply(lambda t: t[0]); w["Cohort"] = parsed.apply(lambda t: t[1]); w["Tissue"] = parsed.apply(lambda t: t[2])
    base = w[[filename_col, metric_col, "Group_Full", "Cohort", "Tissue"]].dropna(subset=[metric_col]).copy()

    uniq_tissues = sorted([t for t in base["Tissue"].dropna().unique().tolist()])
    default_use_tissue = len(uniq_tissues) > 1
    use_tissue = st.sidebar.toggle("Use tissue as a factor", value=default_use_tissue)

    with st.expander("Parsed preview"):
        st.dataframe(base.head(20))

    cohort_counts = base["Cohort"].value_counts(dropna=True)
    all_cohorts = sorted([c for c in cohort_counts.index.tolist() if pd.notna(c)], key=cohort_sort_key)
    selected_cohorts = st.multiselect("Filter cohorts", options=all_cohorts, default=all_cohorts)

    if use_tissue:
        tissue_counts = base["Tissue"].value_counts(dropna=True)
        all_tissues = sorted([t for t in tissue_counts.index.tolist() if pd.notna(t)])
        selected_tissues = st.multiselect("Filter tissues", options=all_tissues, default=all_tissues)
    else:
        selected_tissues = None

    if selected_cohorts: base = base[base["Cohort"].isin(selected_cohorts)]
    if use_tissue and selected_tissues: base = base[base["Tissue"].isin(selected_tissues)]
    if base.empty: st.warning("No rows available after filtering."); st.stop()

    if qc_mode:
        grp_cols = ["Cohort","Tissue"] if use_tissue else ["Cohort"]
        small_groups = base.groupby(grp_cols, dropna=False)[metric_col].size()
        if (small_groups < 3).any(): st.warning("Some groups have <3 samples. Interpret bars/CI with caution.")
        base = mark_outliers_iqr(base, metric_col, grp_cols)

    st.markdown("### Plot 1 — Compare cohorts")
    agg_mode = st.radio("Aggregation:", options=["mean", "median"], index=0, horizontal=True, key="agg_cohort")
    order_coh = sorted(base["Cohort"].dropna().unique(), key=cohort_sort_key)

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    if agg_mode == "mean":
        sns.barplot(data=base, x="Cohort", y=metric_col, estimator=np.mean, ci=95, ax=ax1, color=COLOR, edgecolor=COLOR_DARK, order=order_coh)
        if qc_mode:
            sns.stripplot(data=base, x="Cohort", y=metric_col, order=order_coh, ax=ax1, color="black", alpha=0.55, jitter=0.2, size=3)
            if "_outlier" in base and base["_outlier"].any():
                out = base[base["_outlier"] == True]
                sns.stripplot(data=out, x="Cohort", y=metric_col, order=order_coh, ax=ax1, marker="X", linewidth=0.8, edgecolor="red", color="red", size=6)
        title1 = f"{metric_col} by Cohort (mean ± 95% CI)"
        stylize(ax1, title1, "Cohort", metric_col)
    else:
        sns.boxplot(data=base, x="Cohort", y=metric_col, ax=ax1, color=COLOR, fliersize=2, linewidth=1, order=order_coh)
        sns.stripplot(data=base, x="Cohort", y=metric_col, ax=ax1, color=COLOR_DARK, alpha=0.6, jitter=0.2, size=3, order=order_coh)
        title1 = f"{metric_col} by Cohort "
        if qc_mode and "_outlier" in base and base["_outlier"].any():
            out = base[base["_outlier"] == True]
            sns.stripplot(data=out, x="Cohort", y=metric_col, order=order_coh, ax=ax1, marker="X", linewidth=0.8, edgecolor="red", color="red", size=6)
        stylize(ax1, title1, "Cohort", metric_col)

    y_max_1 = st.number_input("Y max (Plot 1)", min_value=0.0, value=0.0, step=0.1, key="ymax1")
    apply_ylim(ax1, base[metric_col].max(), y_max_1); apply_locator(ax1, tick_step)
    st.pyplot(fig1)
    st.download_button("Download PNG (Plot 1)", data=fig_png_bytes(fig1), file_name=fname_from_title(title1), mime="image/png")

    with st.expander("Rows used (cohort plot)"):
        st.dataframe(base[[filename_col,"Cohort","Tissue",metric_col]].reset_index(drop=True))
        st.download_button("Download rows (CSV)", data=safecsv(base[[filename_col,"Cohort","Tissue",metric_col]]), file_name="cohort_plot_rows.csv", mime="text/csv")
    with st.expander("Show the math (cohort aggregation)"):
        math_cohort = base.groupby("Cohort")[metric_col].apply(calc_table).reset_index().drop(columns=["level_1"])
        st.dataframe(math_cohort)
        st.download_button("Download cohort calculations (CSV)", data=safecsv(math_cohort), file_name="cohort_calculations.csv", mime="text/csv")

    st.markdown("### Plot 2 — Within-cohort (per full ID)")
    cohort_opts = sorted(base["Cohort"].dropna().unique().tolist(), key=cohort_sort_key)
    single_cohort = st.selectbox("Choose a cohort", options=cohort_opts)
    within = base[base["Cohort"] == single_cohort].dropna(subset=["Group_Full"]).copy()

    if within.empty:
        st.info("No rows for the selected cohort.")
    else:
        if qc_mode:
            grp_cols2 = ["Group_Full","Tissue"] if (use_tissue and within["Tissue"].notna().any()) else ["Group_Full"]
            within = mark_outliers_iqr(within, metric_col, grp_cols2)
            small_ids = within.groupby(grp_cols2, dropna=False)[metric_col].size()
            if (small_ids < 3).any(): st.warning(f"{single_cohort}: Some subgroups have <3 samples.")

        sort_mode = st.selectbox("Sort IDs by", ["ID (A→Z)", "median (desc)", "median (asc)", "mean (desc)", "mean (asc)"], index=1)
        point_size = st.slider("Point size", 3, 12, 6, step=1)
        point_alpha = st.slider("Point transparency", 0.2, 1.0, 0.75, step=0.05)
        show_ref = st.multiselect("Reference lines", ["cohort mean", "cohort median"], default=["cohort median"])

        stats_per_id = within.groupby("Group_Full")[metric_col].agg(["median","mean"]).reset_index()
        if sort_mode == "ID (A→Z)":
            order_ids = sorted(within["Group_Full"].unique().tolist())
        elif "median" in sort_mode:
            order_ids = stats_per_id.sort_values("median", ascending=("asc" in sort_mode))["Group_Full"].tolist()
        else:
            order_ids = stats_per_id.sort_values("mean", ascending=("asc" in sort_mode))["Group_Full"].tolist()

        title2 = f"{single_cohort}: {metric_col}"
        use_hue = bool(use_tissue and within["Tissue"].notna().any())
        fig_w = max(8, min(22, 0.55*len(order_ids) * (1.2 if use_hue else 1.0)))
        fig2, ax2 = plt.subplots(figsize=(fig_w, 5))

        common_kwargs = dict(data=within, x="Group_Full", y=metric_col, order=order_ids, ax=ax2,
                             linewidth=0.4, edgecolor="black", alpha=point_alpha)
        if use_hue:
            sns.swarmplot(hue="Tissue", dodge=True, size=point_size, **common_kwargs)
            ax2.legend(title="Tissue", frameon=True)
        else:
            sns.swarmplot(size=point_size, color=COLOR_DARK, **common_kwargs)

        if qc_mode and "_outlier" in within and within["_outlier"].any():
            o2 = within[within["_outlier"] == True]
            sns.stripplot(data=o2, x="Group_Full", y=metric_col, order=order_ids, ax=ax2,
                          marker="X", linewidth=0.9, edgecolor="red", color="red",
                          size=min(point_size+2, 12), dodge=use_hue, hue=("Tissue" if use_hue else None), jitter=0.0)
            if use_hue and ax2.get_legend():
                handles, labels = ax2.get_legend_handles_labels()
                uniq = list(dict(zip(labels, handles)).items())
                ax2.legend([h for _, h in uniq], [l for l, _ in uniq], title="Tissue", frameon=True)

        if "cohort mean" in show_ref:
            m = float(within[metric_col].mean())
            if not np.isnan(m): ax2.axhline(m, linestyle="--", linewidth=1, alpha=0.6)
        if "cohort median" in show_ref:
            md = float(within[metric_col].median())
            if not np.isnan(md): ax2.axhline(md, linestyle=":", linewidth=1, alpha=0.8)

        stylize(ax2, title2, "Group ID", metric_col)
        y_max_2 = st.number_input("Y max (Plot 2)", min_value=0.0, value=0.0, step=0.1)
        apply_ylim(ax2, within[metric_col].max(), y_max_2); apply_locator(ax2, tick_step)
        for lbl in ax2.get_xticklabels(): lbl.set_rotation(28); lbl.set_ha("right")
        st.pyplot(fig2)
        st.download_button("Download PNG (Plot 2)", data=fig_png_bytes(fig2), file_name=fname_from_title(title2), mime="image/png")

        with st.expander("Rows used (within-cohort plot)"):
            st.dataframe(within[[filename_col,"Cohort","Group_Full","Tissue",metric_col]].reset_index(drop=True))
            st.download_button("Download rows (CSV)", data=safecsv(within[[filename_col,"Cohort","Group_Full","Tissue",metric_col]]), file_name="within_cohort_rows.csv", mime="text/csv")

        with st.expander("Show the math (within-cohort)"):
            if use_hue:
                math_within = within.groupby(["Group_Full","Tissue"])[metric_col].apply(calc_table).reset_index().drop(columns=["level_2"])
            else:
                math_within = within.groupby(["Group_Full"])[metric_col].apply(calc_table).reset_index().drop(columns=["level_1"])
            st.dataframe(math_within)
            st.download_button("Download within-cohort calculations (CSV)", data=safecsv(math_within), file_name="within_cohort_calculations.csv", mime="text/csv")

    st.markdown("### Per-image bars (filtered)")
    per_img = base.copy()
    all_ids = sorted(per_img["Group_Full"].dropna().unique().tolist())
    ids_for_images = st.multiselect("Filter by full IDs", options=all_ids, default=all_ids[:1] if all_ids else [])
    if ids_for_images: per_img = per_img[per_img["Group_Full"].isin(ids_for_images)]
    if per_img.empty:
        st.info("No rows match filters.")
    else:
        fig_d, ax_d = plt.subplots(figsize=(max(8, min(18, 0.5*len(per_img))), 5))
        plot_df2 = per_img.copy()
        plot_df2["Label"] = plot_df2["Group_Full"].fillna(plot_df2[filename_col].astype(str))
        hue_arg = "Tissue" if (use_tissue and plot_df2["Tissue"].notna().any()) else None
        sns.barplot(data=plot_df2, x="Label", y=metric_col, hue=hue_arg, ax=ax_d, edgecolor=COLOR_DARK)
        if qc_mode:
            sns.stripplot(data=plot_df2, x="Label", y=metric_col, hue=hue_arg, dodge=bool(hue_arg), alpha=0.5, size=3, jitter=0.15, ax=ax_d)
        title_d = f"Per-Image: {metric_col}"
        stylize(ax_d, title_d, "Image", metric_col)
        y_max_d = st.number_input("Y max (Per-image)", min_value=0.0, value=0.0, step=0.1, key="ymax_d2")
        apply_ylim(ax_d, plot_df2[metric_col].max(), y_max_d); apply_locator(ax_d, tick_step)
        for lbl in ax_d.get_xticklabels(): lbl.set_rotation(35); lbl.set_ha("right")
        plt.tight_layout()
        st.pyplot(fig_d)
        st.download_button("Download PNG (Per-image)", data=fig_png_bytes(fig_d), file_name=fname_from_title(title_d), mime="image/png")

        with st.expander("Rows used (per-image)"):
            st.dataframe(plot_df2[[filename_col,"Label","Cohort","Tissue",metric_col]].reset_index(drop=True))
            st.download_button("Download rows (CSV)", data=safecsv(plot_df2[[filename_col,"Label","Cohort","Tissue",metric_col]]), file_name="per_image_rows.csv", mime="text/csv")
        with st.expander("Show the math (per-image subset)"):
            gcols = ["Label"] + ([hue_arg] if hue_arg else [])
            math_img = plot_df2.groupby(gcols)[metric_col].apply(calc_table).reset_index()
            math_img = math_img.drop(columns=["level_{}".format(2 if hue_arg else 1)], errors="ignore")
            st.dataframe(math_img)
            st.download_button("Download per-image calculations (CSV)", data=safecsv(math_img), file_name="per_image_calculations.csv", mime="text/csv")
