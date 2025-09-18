# lib/helpers.py
import io, re
from typing import Set
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

COLOR = "#59BBBB"
COLOR_DARK = "#59BBBB"

def stylize(ax, title=None, xlabel=None, ylabel=None):
    if title: ax.set_title(title, weight="600")
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    return ax

def apply_ylim(ax, data_max, override):
    ax.set_ylim(0, override if override and override > 0 else (float(data_max) * 1.10 if data_max is not None else 1.0))

def apply_ylim_range(ax, data_min=None, data_max=None, min_override=None, max_override=None):
    y0 = min_override if (min_override is not None) else (float(data_min) if data_min is not None else 0.0)
    y1 = max_override if (max_override and max_override > 0) else (float(data_max) * 1.10 if data_max is not None else 1.0)
    ax.set_ylim(y0, y1)

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

def slugify(s): 
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_").lower()

def fig_png_bytes(fig, dpi=200):
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=dpi, bbox_inches="tight"); b.seek(0); return b

def fname_from_title(title, ext="png"): 
    return f"{slugify(title)}.{ext}"

DEFAULT_IGNORE_TOKENS = {"S1", "S2", "S3"}
DEFAULT_TISSUE_TOKENS = {"HEART", "LIVER", "HRT", "LVR"}

GROUP_RE = re.compile(r"(?i)\bG\d+(?:-\d+)?\b")
PAS_RE = re.compile(r"(?i)\bPAS_(WT\d+|\d{2,4}(?:-\d+)?)\b")

def normalize_tokens(tokens): 
    return {t.strip().upper() for t in tokens if isinstance(t, str) and t.strip()}

def split_tokens(name): 
    return [t for t in re.split(r"[_.]", str(name)) if t]

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

def guess_metric(cols):
    for c in cols:
        if isinstance(c, str) and c.strip().lower() == "positive area percentage": return c
    for c in cols:
        if isinstance(c, str) and "positive" in c.lower() and "percent" in c.lower(): return c
    return None
