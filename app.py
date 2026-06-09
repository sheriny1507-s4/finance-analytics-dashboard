import streamlit as st
import pandas as pd
import pickle
import os
import pdfplumber
import matplotlib.pyplot as plt
from login import login
import os
import pickle

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Personal Finance Dashboard",
    layout="wide"
)

# ==========================================
# LOGIN SYSTEM
# ==========================================
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
    st.stop()

# ==========================================
# HEADER
# ==========================================
st.title("💰 Personal Finance Analytics Dashboard")

st.sidebar.success(
    f"Welcome {st.session_state.get('username','User')}"
)

if st.sidebar.button("Logout"):
    st.session_state["logged_in"] = False

    if "username" in st.session_state:
        del st.session_state["username"]

    st.rerun()

# ==========================================
# DASHBOARD CARDS
# ==========================================
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("📂 Upload CSV")

with col2:
    st.info("📊 Analyze Spending")

with col3:
    st.info("⬇ Download Reports")

st.markdown("---")

# ==========================================
# LOAD MODEL
# ==========================================
MODEL_PATH = os.path.join("models", "model.pkl")

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model not found: {MODEL_PATH}")
    st.stop()

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ==========================================
# FILE UPLOAD
# ==========================================
uploaded_file = st.file_uploader(
    "📂 Upload CSV, Excel or PDF",
    type=["csv", "xlsx", "pdf"]
)

# ==========================================
# PROCESS FILE
# ==========================================
if uploaded_file is not None:

    try:

        # CSV
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        # Excel
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)

        # PDF
        elif uploaded_file.name.endswith(".pdf"):

            dataframes = []

            with pdfplumber.open(uploaded_file) as pdf:

                for page in pdf.pages:

                    tables = page.extract_tables()

                    if tables:

                        for table in tables:

                            if len(table) > 1:

                                temp_df = pd.DataFrame(
                                    table[1:],
                                    columns=table[0]
                                )

                                dataframes.append(temp_df)

                    else:

                        text = page.extract_text()

                        if text:

                            lines = text.split("\n")

                            temp_df = pd.DataFrame(
                                {"Description": lines}
                            )

                            dataframes.append(temp_df)

            if len(dataframes) == 0:
                st.error("❌ No readable data found in PDF")
                st.stop()

            df = pd.concat(
                dataframes,
                ignore_index=True
            )

            st.subheader("📄 Extracted PDF Data")
            st.dataframe(df.head())

        else:
            st.error("Unsupported file format")
            st.stop()

    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # ==========================================
    # CLEAN COLUMN NAMES
    # ==========================================
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    # ==========================================
    # DETECT COLUMNS
    # ==========================================
    description_col = None
    amount_col = None
    date_col = None

    for col in df.columns:

        c = col.lower()

        if any(word in c for word in
               ["description", "details", "narration", "remarks"]):
            description_col = col

        if any(word in c for word in
               ["amount", "debit", "credit", "bill"]):
            amount_col = col

        if any(word in c for word in
               ["date", "time"]):
            date_col = col

    # ==========================================
    # MANUAL COLUMN SELECTION
    # ==========================================
    if description_col is None:

        st.warning(
            "Description column not found."
        )

        description_col = st.selectbox(
            "Select Description Column",
            df.columns
        )

    # ==========================================
    # STANDARDIZE COLUMNS
    # ==========================================
    df.rename(
        columns={description_col: "Description"},
        inplace=True
    )

    df["Amount"] = (
    df["Amount"]
    .astype(str)
    .str.replace(",", "")
    .str.replace("₹", "")
    .str.replace("$", "")
)

df["Amount"] = pd.to_numeric(
    df["Amount"],
    errors="coerce"
)

df["Amount"] = df["Amount"].fillna(0)

    if date_col:

        df.rename(
            columns={date_col: "Date"},
            inplace=True
        )

    
st.write("Columns Found:")
st.write(df.columns.tolist())

st.write("Rows After Cleaning:")
st.write(len(df))
   

    # ==========================================
    # SIDEBAR INSIGHTS
    # ==========================================
    st.sidebar.header("📌 Key Insights")

    st.sidebar.metric(
        "Total Transactions",
        len(df)
    )

    st.sidebar.metric(
        "Categories",
        df["Predicted Category"].nunique()
    )

    total_expense = df["Amount"].sum()

    st.sidebar.metric(
        "Total Expense",
        f"${total_expense:,.2f}"
    )

    # ==========================================
    # TABLE
    # ==========================================
    st.subheader("📋 Prediction Results")

    st.dataframe(
        df,
        use_container_width=True
    )

    # ==========================================
    # PIE CHART
    # ==========================================
    category_expense = (
        df.groupby("Predicted Category")["Amount"]
        .sum()
    )

    category_expense = category_expense[
        category_expense > 0
    ]

    if len(category_expense) > 0:

        st.subheader(
            "📊 Expense Distribution by Category"
        )

        fig1, ax1 = plt.subplots()

        ax1.pie(
            category_expense.values,
            labels=category_expense.index,
            autopct="%1.1f%%"
        )

        ax1.axis("equal")

        st.pyplot(fig1)

    # ==========================================
    # MONTHLY TREND
    # ==========================================
    if "Date" in df.columns:

        try:

            df["Date"] = pd.to_datetime(
                df["Date"],
                errors="coerce"
            )

            monthly = (
                df.groupby(
                    df["Date"].dt.to_period("M")
                )["Amount"]
                .sum()
            )

            if len(monthly) > 0:

                st.subheader(
                    "📈 Monthly Expense Trend"
                )

                fig2, ax2 = plt.subplots()

                monthly.plot(
                    kind="line",
                    ax=ax2
                )

                st.pyplot(fig2)

        except:
            pass

    # ==========================================
    # DOWNLOAD CSV
    # ==========================================
    csv = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇ Download Results as CSV",
        csv,
        "predicted_expenses.csv",
        "text/csv"
    )
