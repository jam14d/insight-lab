import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import re, io

# App + theme
st.set_page_config(page_title="InsightLab: Interactive Data Exploration", layout="wide")
st.title("InsightLab: Interactive Data Exploration")
COLOR = "#87ae73"; COLOR_DARK = "#8A9A5B"
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

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Go to", ["EDA", "Metric by Group (Box/Bar Plots)"])

# Data
@st.cache_data
def load_default_data():
    url = "https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/imdb_1000.csv"
    return pd.read_csv(url)

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    try: df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the uploaded file: {e}"); df = load_default_data()
else:
    df = load_default_data()

# Helpers
def guess_metric(cols):
    for c in cols:
        if isinstance(c, str) and c.strip().lower() == "positive area percentage": return c
    for c in cols:
        if isinstance(c, str) and "positive" in c.lower() and "percent" in c.lower(): return c
    return None

ID_PAT = re.compile(r"(?:^|_)PAS_(\d{2,4}|WT\d+)(?:_|$)", re.IGNORECASE)
def parse_id(name):
    if not isinstance(name, str): return None
    m = ID_PAT.search(name);  return m.group(1).upper() if m else None

def parse_type(name, ignore_tokens):
    if not isinstance(name, str): return None
    tokens = re.split(r"[_.]", name)
    for tok in reversed(tokens[:-1]):
        t = tok.upper()
        if t and t not in ignore_tokens: return t
    return None

def has_ignored(name, ignore_tokens):
    if not isinstance(name, str): return False
    low = name.lower()
    return any(tok.lower() in low for tok in ignore_tokens)

# Page: EDA
if page == "EDA":
    st.subheader("Preview")
    st.dataframe(df.head())

    st.subheader("Explore a column")
    colname = st.selectbox("Column", df.columns)
    if pd.api.types.is_numeric_dtype(df[colname]):
        s = df[colname].dropna()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Count", int(s.count()))
        c2.metric("Mean", round(float(s.mean()), 4) if len(s) else np.nan)
        c3.metric("Std", round(float(s.std()), 4) if len(s) else np.nan)
        c4.metric("Min", round(float(s.min()), 4) if len(s) else np.nan)
        c5.metric("Max", round(float(s.max()), 4) if len(s) else np.nan)

        fig, ax = plt.subplots()
        sns.histplot(s, ax=ax, color=COLOR, edgecolor=COLOR_DARK, alpha=0.75)
        try: sns.kdeplot(s, ax=ax, color=COLOR_DARK, linewidth=2)
        except Exception: pass
        stylize(ax, f"Distribution of {colname}", colname, "Count")
        st.pyplot(fig)
    else:
        st.write("Value counts")
        st.write(df[colname].value_counts(dropna=False))

    st.subheader("Describe (all)")
    st.write(df.describe(include="all"))

# Page: Plots
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

    st.markdown("**Y ticks**")
    tick_step = st.number_input("Y tick step (0 = auto)", min_value=0.0, value=0.0, step=0.1)

    st.markdown("**Grouping**")
    default_ignore = ["s1", "s3", "ctrl"]
    ignore_text = st.text_input("Ignore tokens (comma-separated)", value=",".join(default_ignore))
    ignore_tokens = {t.strip().upper() for t in ignore_text.split(",") if t.strip()}
    drop_ignored = st.checkbox("Drop rows containing ignored tokens", value=False)

    w = df.copy()
    if drop_ignored: w = w[~w[filename_col].apply(lambda x: has_ignored(x, ignore_tokens))].copy()
    w["Group_ID"] = w[filename_col].apply(parse_id)
    w["Group_Type"] = w[filename_col].apply(lambda x: parse_type(x, ignore_tokens))
    base = w[[filename_col, metric_col, "Group_ID", "Group_Type"]].dropna(subset=[metric_col]).copy()

    with st.expander("Parsed preview"):
        st.dataframe(base.head(20))

    plot_num_df = base.dropna(subset=["Group_ID"])[["Group_ID", metric_col]].rename(columns={"Group_ID": "Group"})
    plot_tis_df = base.dropna(subset=["Group_Type"])[["Group_Type", metric_col]].rename(columns={"Group_Type": "Group"})
    if plot_num_df.empty and plot_tis_df.empty:
        st.warning("No data available."); st.stop()

    def make_boxplot(data, y_col, group_label):
        order = data.groupby("Group")[y_col].median().sort_values(ascending=False).index.tolist()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=data, x="Group", y=y_col, order=order, ax=ax, color=COLOR, fliersize=2, linewidth=1)
        sns.stripplot(data=data, x="Group", y=y_col, order=order, ax=ax, color=COLOR_DARK, alpha=0.65, jitter=0.2, size=3)
        if len(order) > 8:
            for lbl in ax.get_xticklabels(): lbl.set_rotation(30); lbl.set_ha("right")
        stylize(ax, f"{y_col} by {group_label}", group_label, y_col)
        return fig, ax

    # Plot A — by ID
    st.markdown("**Plot A — by ID**")
    y_max_a = st.number_input("Y max (Plot A)", min_value=0.0, value=0.0, step=0.1, key="ymax_a")
    if not plot_num_df.empty:
        fig_a, ax_a = make_boxplot(plot_num_df, metric_col, "ID")
        apply_ylim(ax_a, plot_num_df[metric_col].max(), y_max_a); apply_locator(ax_a, tick_step)
        st.pyplot(fig_a)
        buf_a = io.BytesIO(); fig_a.savefig(buf_a, format="png", dpi=200, bbox_inches="tight"); buf_a.seek(0)
        st.download_button("Download Plot A (PNG)", data=buf_a, file_name=f"{metric_col}_by_ID.png", mime="image/png")

    # Plot B — by Type
    st.markdown("**Plot B — by Type**")
    y_max_b = st.number_input("Y max (Plot B)", min_value=0.0, value=0.0, step=0.1, key="ymax_b")
    if not plot_tis_df.empty:
        fig_b, ax_b = make_boxplot(plot_tis_df, metric_col, "Type")
        apply_ylim(ax_b, plot_tis_df[metric_col].max(), y_max_b); apply_locator(ax_b, tick_step)
        st.pyplot(fig_b)
        buf_b = io.BytesIO(); fig_b.savefig(buf_b, format="png", dpi=200, bbox_inches="tight"); buf_b.seek(0)
        st.download_button("Download Plot B (PNG)", data=buf_b, file_name=f"{metric_col}_by_Type.png", mime="image/png")

    # Plot C — Per-sample bars
    st.markdown("**Plot C — Per-sample bars**")
    ids = sorted(base["Group_ID"].dropna().unique().tolist())
    selected_ids = st.multiselect("Sample ID(s)", options=ids, default=ids[:1])
    agg_mode = st.radio("If multiple rows per (ID, Type):", options=["mean", "median", "max", "min"], horizontal=True)
    y_max_c = st.number_input("Y max (Plot C)", min_value=0.0, value=0.0, step=0.1, key="ymax_c")

    def sample_barplot(sample_id):
        subset = base[(base["Group_ID"] == sample_id) & base["Group_Type"].notna()].copy()
        if subset.empty: return None, None, None
        agg_df = subset.groupby("Group_Type", as_index=False)[metric_col].agg(agg_mode).sort_values("Group_Type")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.barplot(data=agg_df, x="Group_Type", y=metric_col, ax=ax, color=COLOR, edgecolor=COLOR_DARK)
        for lbl in ax.get_xticklabels(): lbl.set_rotation(20); lbl.set_ha("right")
        stylize(ax, f"Sample {sample_id}: {metric_col} by Type ({agg_mode})", "Type", metric_col)
        apply_ylim(ax, agg_df[metric_col].max(), y_max_c); apply_locator(ax, tick_step)
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=200, bbox_inches="tight"); buf.seek(0)
        return fig, buf, f"{metric_col}_sample_{sample_id}.png"

    if selected_ids:
        for sid in selected_ids:
            fig_c, buf_c, fname_c = sample_barplot(sid)
            if fig_c is None:
                st.info(f"No types for sample {sid}.")
                continue
            st.pyplot(fig_c)
            st.download_button(f"Download Sample {sid} (PNG)", data=buf_c, file_name=fname_c, mime="image/png")

    # Plot D — Per-image bars
    st.markdown("**Plot D — Per-image bars**")
    all_ids = sorted(base["Group_ID"].dropna().unique().tolist())
    all_tis = sorted(base["Group_Type"].dropna().unique().tolist())
    c1, c2, c3 = st.columns([2,2,1])
    with c1:
        ids_for_images = st.multiselect("Filter by ID(s)", options=all_ids, default=all_ids[:1] if all_ids else [])
    with c2:
        tis_for_images = st.multiselect("Filter by type(s)", options=all_tis, default=[])
    with c3:
        short_labels = st.checkbox("Short labels", value=True)
    y_max_d = st.number_input("Y max (Plot D)", min_value=0.0, value=0.0, step=0.1, key="ymax_d")

    per_img = base.copy()
    if ids_for_images: per_img = per_img[per_img["Group_ID"].isin(ids_for_images)]
    if tis_for_images: per_img = per_img[per_img["Group_Type"].isin(tis_for_images)]

    if per_img.empty:
        st.info("No rows match filters.")
    else:
        def mk_label(row):
            if not short_labels: return str(row[filename_col])
            sid, tis = row.get("Group_ID"), row.get("Group_Type")
            if pd.notna(sid) and pd.notna(tis): return f"{sid}_{tis}"
            parts = re.split(r"[_.]", str(row[filename_col])); return parts[-2] if len(parts) >= 2 else str(row[filename_col])

        plot_df = per_img[[filename_col, "Group_ID", "Group_Type", metric_col]].copy()
        plot_df["Label"] = plot_df.apply(mk_label, axis=1)
        plot_df = plot_df.sort_values(by=["Group_ID", "Group_Type", "Label"])

        fig_d, ax_d = plt.subplots(figsize=(max(8, min(18, 0.5 * len(plot_df))), 5))
        sns.barplot(data=plot_df, x="Label", y=metric_col, ax=ax_d, color=COLOR, edgecolor=COLOR_DARK)
        for lbl in ax_d.get_xticklabels(): lbl.set_rotation(35); lbl.set_ha("right")
        stylize(ax_d, f"Per-Image: {metric_col}", "Image", metric_col)
        apply_ylim(ax_d, plot_df[metric_col].max(), y_max_d); apply_locator(ax_d, tick_step)
        plt.tight_layout()

        buf_d = io.BytesIO(); fig_d.savefig(buf_d, format="png", dpi=200, bbox_inches="tight"); buf_d.seek(0)
        st.pyplot(fig_d)
        st.download_button("Download Per-Image Plot (PNG)", data=buf_d, file_name=f"{metric_col}_per_image.png", mime="image/png")
        with st.expander("Rows used"):
            st.dataframe(plot_df[[filename_col, "Group_ID", "Group_Type", metric_col]].reset_index(drop=True))
