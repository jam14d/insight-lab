import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Title
st.title("Insight: Interactive Data Exploration")

# Sidebar Info
st.sidebar.header("Data Analysis Reference")
st.sidebar.markdown("""
**Example Use Case:**
To find the **maximum star rating for a crime film**:
1. Go to the "Conditional Value Lookup" section
2. Select `genre` as the filter column
3. Choose `Crime` as the value
4. Pick `star_rating` as the numeric column
5. Set the aggregation to `max`

**Tip:** Avoid using min, max, or median when there's only one value per group.
For example, calculating median `duration` for `The Shawshank Redemption` isn't meaningful—there's only one value, so all stats will reflect that single value.

---
**Common Descriptive Stats:**
- `df.column.mean()` → Average of values in column
- `df.column.std()` → Standard deviation
- `df.column.median()` → Median value
- `df.column.max()` → Maximum value
- `df.column.min()` → Minimum value
- `df.column.count()` → Count of non-null values
- `df.column.nunique()` → Number of unique values
- `df.column.unique()` → Array of unique values

**Spread Metrics:**
- Range: `df.column.max() - df.column.min()`
- IQR: `df.column.quantile(0.75) - df.column.quantile(0.25)`
- Variance: `df.column.var()`
- MAD: `df.column.mad()`

**Categorical Analysis:**
- `df.column.value_counts()` → Count per category
- `df.column.value_counts(normalize=True)` → Proportion per category
""")

# Load default movie dataset
@st.cache_data
def load_default_data():
    url = "https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/imdb_1000.csv"
    return pd.read_csv(url)

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

df = pd.read_csv(uploaded_file) if uploaded_file else load_default_data()

# Show basic dataframe
st.subheader("Preview of Data")
st.dataframe(df.head())

# EDA options
st.subheader("Exploratory Data Analysis")
selected_column = st.selectbox("Select a column for analysis", df.columns)

# Define available functions
numeric_functions = [
    "Mean", "Standard Deviation", "Median", "Maximum", "Minimum",
    "Count", "Number of Unique Values", "List of Unique Values",
    "Range", "Interquartile Range (IQR)", "Variance", "Mean Absolute Deviation (MAD)"
]
categorical_functions = ["Value Counts", "Normalized Value Counts"]

if pd.api.types.is_numeric_dtype(df[selected_column]):
    function_options = numeric_functions + categorical_functions
else:
    function_options = categorical_functions + [
        "Count", "Number of Unique Values", "List of Unique Values"
    ]

analysis_type = st.selectbox("Select an analysis function", function_options)

st.write("### Result")
try:
    if analysis_type == "Mean":
        st.write(df[selected_column].mean())
    elif analysis_type == "Standard Deviation":
        st.write(df[selected_column].std())
    elif analysis_type == "Median":
        st.write(df[selected_column].median())
    elif analysis_type == "Maximum":
        st.write(df[selected_column].max())
    elif analysis_type == "Minimum":
        st.write(df[selected_column].min())
    elif analysis_type == "Count":
        st.write(df[selected_column].count())
    elif analysis_type == "Number of Unique Values":
        st.write(df[selected_column].nunique())
    elif analysis_type == "List of Unique Values":
        st.write(df[selected_column].unique())
    elif analysis_type == "Range":
        st.write(df[selected_column].max() - df[selected_column].min())
    elif analysis_type == "Interquartile Range (IQR)":
        st.write(df[selected_column].quantile(0.75) - df[selected_column].quantile(0.25))
    elif analysis_type == "Variance":
        st.write(df[selected_column].var())
    elif analysis_type == "Mean Absolute Deviation (MAD)":
        st.write(df[selected_column].mad())
    elif analysis_type == "Value Counts":
        st.write(df[selected_column].value_counts())
    elif analysis_type == "Normalized Value Counts":
        st.write(df[selected_column].value_counts(normalize=True))
except Exception as e:
    st.error(f"Error applying function: {e}")

# Seaborn Plot
if pd.api.types.is_numeric_dtype(df[selected_column]):
    st.write(f"### Distribution Plot for {selected_column}")
    fig, ax = plt.subplots()
    sns.histplot(df[selected_column].dropna(), kde=True, ax=ax)
    st.pyplot(fig)

# Filter-Based Conditional Summary
st.subheader("Conditional Value Lookup")

filter_col = st.selectbox("Select a categorical column to filter by", df.select_dtypes(include='object').columns)
filter_val = st.selectbox(f"Select a value in '{filter_col}'", df[filter_col].dropna().unique())
numeric_col_for_lookup = st.selectbox("Select a numeric column to summarize", df.select_dtypes(include='number').columns)
agg_lookup_func = st.selectbox("Select an aggregation function", ["max", "min", "mean", "median", "std"])

if st.button("Run Conditional Lookup"):
    try:
        filtered_df = df[df[filter_col] == filter_val]
        result = getattr(filtered_df[numeric_col_for_lookup], agg_lookup_func)()
        st.success(f"{agg_lookup_func.upper()} {numeric_col_for_lookup} for {filter_val} in {filter_col}: {result}")
    except Exception as e:
        st.error(f"Failed to compute value: {e}")


# Full Descriptive Statistics
st.write("### Full Descriptive Statistics")
st.write(df.describe(include='all'))
