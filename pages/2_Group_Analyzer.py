import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
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
metric_col = st.selectbox(
    "Metric (y-axis)",
    options=num_cols,
    index=(num_cols.index(metric_guess) if metric_guess in num_cols else 0),
    key="group_metric"
)

tick_step = st.number_input("Y tick step (0 = auto)", min_value=0.0, value=0.0, step=0.1)

# 
# Build base dataframe
# 
if is_titanic:
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    if not obj_cols: st.error("No categorical columns found."); st.stop()
    cohort_col = st.selectbox("Cohort column", options=obj_cols, index=(obj_cols.index("class") if "class" in obj_cols else 0))
    id_col = st.selectbox("ID column", options=obj_cols, index=(obj_cols.index("who") if "who" in obj_cols else 0))

    w = df.copy()

    # If the chosen column contains patterns like A1/B2, collapse to letter; otherwise pass through.
    def _letter_or_self(x):
        s = str(x)
        m = re.search(r'\b([A-Za-z])(\d+)\b', s)
        return m.group(1).upper() if m else s

    w["Cohort"] = w[cohort_col].apply(_letter_or_self)
    w["Group_Full"] = w[id_col].astype(str)
    w["Cohort_Sub"] = np.nan
    w["Marker"] = np.nan  # Titanic demo doesn’t have marker concept
    base = w[[id_col, metric_col, "Group_Full", "Cohort", "Cohort_Sub", "Marker"]].dropna(subset=[metric_col]).copy()

else:
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    if not obj_cols: st.error("No text/filename column found."); st.stop()
    filename_col = st.selectbox("Filename column", options=obj_cols, index=(obj_cols.index("Name") if "Name" in obj_cols else 0))

    tissue_text = st.text_input("Key tokens", value=",".join(sorted(DEFAULT_TISSUE_TOKENS)))
    # Add CD/CD3/CD8 by default so they’re never treated as cohorts
    extra_ignores = ["CD", "CD3", "CD8", "M"]
    ignore_text = st.text_input(
        "Ignore tokens",
        value=",".join(sorted(set(list(DEFAULT_IGNORE_TOKENS) + extra_ignores)))
    )
    drop_ignored = st.checkbox("Drop rows with ignored tokens", value=False)

    tissue_tokens = normalize_tokens(tissue_text.split(","))
    ignore_tokens = normalize_tokens(ignore_text.split(","))
    ignore_set = set(tok.upper() for tok in ignore_tokens if tok)

    w = df.copy()
    if drop_ignored and ignore_set:
        w = w[~w[filename_col].astype(str).str.upper().apply(lambda s: any(tok in s for tok in ignore_set))].copy()

    #  Smart parser: extract cohorts (A/B/...) and marker (CD3/CD8), ignore 'M' 
    def parse_labels_with_marker(name, ignore_set, tissue_tokens):
        """
        Return (group_full, cohort, tissue, subnum, marker)
        Handles:
          A3M-CD8.qptiff  -> cohort=A, sub=3, marker=CD8
          D5-M-CD3.qptiff -> cohort=D, sub=5, marker=CD3
        Ignores single 'M' and any tokens in ignore_set as cohorts.
        """
        group = cohort = tissue = marker = None
        subnum = np.nan
        s = str(name).upper()

        # original helper, in case encoded tissue/group elsewhere
        try:
            g, c, t = extract_labels(s, ignore_set, tissue_tokens)
            group, cohort, tissue = g or None, c or None, t or None
        except Exception:
            pass

        # 1) A3M-CD8 or E1-M-CD3 etc.  (M optional / ignored)
        if cohort is None or marker is None:
            m = re.search(r'\b([A-Z])(\d+)[-_]?(?:M)?[-_]?CD(\d+)\b', s)
            if m:
                cohort = m.group(1)
                subnum = int(m.group(2))
                marker = f"CD{m.group(3)}"
                if group is None:
                    group = f"{cohort}{subnum}"

        # 2) fallback if no marker present: just A1/B2 etc.
        if cohort is None:
            m2 = re.search(r'\b([A-Z])(\d+)\b', s)
            if m2:
                cohort = m2.group(1)
                subnum = int(m2.group(2))
                if group is None:
                    group = f"{cohort}{subnum}"

        # Never treat ignored tokens as cohorts
        if isinstance(cohort, str) and cohort.upper() in ignore_set:
            cohort = None

        if group is None:
            group = cohort or s
        return group, cohort, tissue, subnum, marker

    parsed = w[filename_col].astype(str).apply(lambda x: parse_labels_with_marker(x, ignore_set, tissue_tokens))
    w["Group_Full"] = parsed.apply(lambda t: t[0])
    w["Cohort"]     = parsed.apply(lambda t: t[1])   # letter only
    w["Tissue"]     = parsed.apply(lambda t: t[2])
    w["Cohort_Sub"] = parsed.apply(lambda t: t[3])   # numeric sub if present
    w["Marker"]     = parsed.apply(lambda t: t[4])   # CD3/CD8 if present
    base = w[[filename_col, metric_col, "Group_Full", "Cohort", "Cohort_Sub", "Marker"]].dropna(subset=[metric_col]).copy()

if base.empty: st.warning("No rows available."); st.stop()


# Marker filter + global Y max

markers_present = sorted([m for m in base["Marker"].dropna().unique().tolist()])
if markers_present:
    marker_choice = st.radio("Marker", options=markers_present + ["Both"], index=0, horizontal=True, key="marker_choice")
else:
    marker_choice = "Both"  # nothing to filter on (e.g., Titanic demo)

shared_ymax = st.number_input("Global Y max (applies to all plots)", min_value=0.0, value=0.0, step=0.1, key="shared_ymax")

# Prepare data by marker selection
if marker_choice != "Both" and marker_choice in markers_present:
    base_sel = base[base["Marker"] == marker_choice].copy()
else:
    base_sel = base.copy()


# Plot 1 — Compare cohorts

st.markdown("### Plot 1 — Compare cohorts")
agg_mode = st.radio("Aggregation:", options=["mean", "median"], index=0, horizontal=True, key="agg_cohort")

order_coh = sorted([c for c in base_sel["Cohort"].dropna().unique().tolist()], key=lambda z: str(z))

fig1, ax1 = plt.subplots(figsize=(10, 5))
if marker_choice == "Both" and markers_present:
    # overlay: two bars per cohort (dodge) by Marker
    if agg_mode == "mean":
        sns.barplot(
            data=base_sel, x="Cohort", y=metric_col, hue="Marker",
            estimator=np.mean, ci=95, ax=ax1, edgecolor=COLOR_DARK, order=order_coh
        )
        title1 = f"{metric_col} by Cohort (mean ± 95% CI) — Both markers"
    else:
        sns.boxplot(data=base_sel, x="Cohort", y=metric_col, hue="Marker",
                    ax=ax1, fliersize=2, linewidth=1, order=order_coh)
        sns.stripplot(data=base_sel, x="Cohort", y=metric_col, hue="Marker",
                      dodge=True, ax=ax1, alpha=0.6, jitter=0.2, size=3, order=order_coh)
        title1 = f"{metric_col} by Cohort — Both markers"
    ax1.legend(title="Marker", frameon=False)
else:
    # single-marker or no marker concept
    if agg_mode == "mean":
        sns.barplot(
            data=base_sel, x="Cohort", y=metric_col, estimator=np.mean, ci=95,
            ax=ax1, color=COLOR, edgecolor=COLOR_DARK, order=order_coh
        )
        title1 = f"{metric_col} by Cohort (mean ± 95% CI)" + (f" — {marker_choice}" if marker_choice != "Both" and markers_present else "")
    else:
        sns.boxplot(data=base_sel, x="Cohort", y=metric_col, ax=ax1, color=COLOR, fliersize=2, linewidth=1, order=order_coh)
        sns.stripplot(data=base_sel, x="Cohort", y=metric_col, ax=ax1, color=COLOR_DARK, alpha=0.6, jitter=0.2, size=3, order=order_coh)
        title1 = f"{metric_col} by Cohort" + (f" — {marker_choice}" if marker_choice != "Both" and markers_present else "")

stylize(ax1, title1, "Cohort", metric_col)
apply_ylim(ax1, base_sel[metric_col].max(), shared_ymax)
apply_locator(ax1, tick_step)
st.pyplot(fig1)
st.download_button("Download PNG (Plot 1)", data=fig_png_bytes(fig1), file_name=fname_from_title(title1), mime="image/png")

with st.expander("Rows used (cohort plot)"):
    cols_common = ["Cohort", "Cohort_Sub", "Marker", metric_col]
    try:
        st.dataframe(base_sel[[filename_col] + cols_common + ["Group_Full"]].reset_index(drop=True))
        st.download_button("Download rows (CSV)", data=safecsv(base_sel[[filename_col] + cols_common + ["Group_Full"]]), file_name="cohort_plot_rows.csv", mime="text/csv")
    except Exception:
        st.dataframe(base_sel[cols_common + ["Group_Full"]].reset_index(drop=True))
        st.download_button("Download rows (CSV)", data=safecsv(base_sel[cols_common + ["Group_Full"]]), file_name="cohort_plot_rows.csv", mime="text/csv")

with st.expander("Show the math (cohort aggregation)"):
    math_cohort = base_sel.groupby(["Cohort"] + (["Marker"] if marker_choice == "Both" and markers_present else []))[metric_col].apply(calc_table).reset_index().drop(columns=["level_1"])
    st.dataframe(math_cohort)
    st.download_button("Download cohort calculations (CSV)", data=safecsv(math_cohort), file_name="cohort_calculations.csv", mime="text/csv")


# Plot 2 — Within-cohort (per full ID)

st.markdown("### Plot 2 — Within-cohort (per full ID)")
cohort_opts = sorted([c for c in base_sel["Cohort"].dropna().unique().tolist()], key=lambda z: str(z))
single_cohort = st.selectbox("Choose a cohort", options=cohort_opts)

within = base_sel[base_sel["Cohort"] == single_cohort].dropna(subset=["Group_Full"]).copy()

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

    title2 = f"{single_cohort}: {metric_col}" + (f" — {marker_choice}" if marker_choice != "Both" and markers_present else "")
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
    apply_ylim(ax2, within[metric_col].max(), shared_ymax)
    apply_locator(ax2, tick_step)
    for lbl in ax2.get_xticklabels():
        lbl.set_rotation(28); lbl.set_ha("right")
    st.pyplot(fig2)
    st.download_button("Download PNG (Plot 2)", data=fig_png_bytes(fig2), file_name=fname_from_title(title2), mime="image/png")

    with st.expander("Rows used (within-cohort plot)"):
        try:
            st.dataframe(within[[filename_col, "Cohort", "Cohort_Sub", "Marker", "Group_Full", metric_col]].reset_index(drop=True))
            st.download_button("Download rows (CSV)", data=safecsv(within[[filename_col, "Cohort", "Cohort_Sub", "Marker", "Group_Full", metric_col]]), file_name="within_cohort_rows.csv", mime="text/csv")
        except Exception:
            st.dataframe(within[["Cohort", "Cohort_Sub", "Marker", "Group_Full", metric_col]].reset_index(drop=True))
            st.download_button("Download rows (CSV)", data=safecsv(within[["Cohort", "Cohort_Sub", "Marker", "Group_Full", metric_col]]), file_name="within_cohort_rows.csv", mime="text/csv")


# Per-image bars (filtered)

st.markdown("### Per-image bars (filtered)")
per_img = base_sel.copy()
all_ids = sorted(per_img["Group_Full"].dropna().unique().tolist())
ids_for_images = st.multiselect("Filter by full IDs", options=all_ids, default=all_ids[:1] if all_ids else [])
if ids_for_images:
    per_img = per_img[per_img["Group_Full"].isin(ids_for_images)]
if per_img.empty:
    st.info("No rows match filters.")
else:
    fig_d, ax_d = plt.subplots(figsize=(max(8, min(18, 0.5 * len(per_img))), 5))
    plot_df2 = per_img.copy()
    plot_df2["Label"] = per_img["Group_Full"].astype(str)
    if marker_choice == "Both" and markers_present:
        sns.barplot(data=plot_df2, x="Label", y=metric_col, hue="Marker", ax=ax_d, edgecolor=COLOR_DARK)
        ax_d.legend(title="Marker", frameon=False)
    else:
        sns.barplot(data=plot_df2, x="Label", y=metric_col, ax=ax_d, edgecolor=COLOR_DARK, color=COLOR)
    title_d = f"Per-Image: {metric_col}" + (f" — {marker_choice}" if marker_choice != "Both" and markers_present else "")
    stylize(ax_d, title_d, "Image", metric_col)
    apply_ylim(ax_d, plot_df2[metric_col].max(), shared_ymax)
    apply_locator(ax_d, tick_step)
    for lbl in ax_d.get_xticklabels():
        lbl.set_rotation(35); lbl.set_ha("right")
    plt.tight_layout()
    st.pyplot(fig_d)
    st.download_button("Download PNG (Per-image)", data=fig_png_bytes(fig_d), file_name=fname_from_title(title_d), mime="image/png")

    with st.expander("Rows used (per-image)"):
        try:
            st.dataframe(plot_df2[[filename_col, "Label", "Cohort", "Cohort_Sub", "Marker", metric_col]].reset_index(drop=True))
            st.download_button("Download rows (CSV)", data=safecsv(plot_df2[[filename_col, "Label", "Cohort", "Cohort_Sub", "Marker", metric_col]]), file_name="per_image_rows.csv", mime="text/csv")
        except Exception:
            st.dataframe(plot_df2[["Label", "Cohort", "Cohort_Sub", "Marker", metric_col]].reset_index(drop=True))
            st.download_button("Download rows (CSV)", data=safecsv(plot_df2[["Label", "Cohort", "Cohort_Sub", "Marker", metric_col]]), file_name="per_image_rows.csv", mime="text/csv")
