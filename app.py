import streamlit as st
import pandas as pd
import pickle
import os
import pdfplumber
import matplotlib.pyplot as plt
from login import login

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Personal Finance Dashboard",
    layout="wide"
)

# ==========================================
# LOGIN
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
    st.rerun()

# ==========================================
# DASHBOARD CARDS
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:
    st.info("📂 Upload CSV / Excel / PDF")

with col2:
    st.info("📊 Analyze Spending")

with col3:
    st.info("⬇ Download Results")

st.markdown("---")

# ==========================================
# LOAD MODEL
# ==========================================

MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ model.pkl not found")
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
    "Upload CSV, Excel or PDF",
    type=["csv", "xlsx", "pdf"]
)

# ==========================================
# PROCESS FILE
# ==========================================

if uploaded_file is not None:

    try:

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)

        elif uploaded_file.name.endswith(".pdf"):

            tables_data = []

            with pdfplumber.open(uploaded_file) as pdf:

                for page in pdf.pages:

                    tables = page.extract_tables()

                    for table in tables:

                        if len(table) > 1:

                            temp_df = pd.DataFrame(
                                table[1:],
                                columns=table[0]
                            )

                            tables_data.append(temp_df)

            if len(tables_data) == 0:
                st.error("No table found in PDF")
                st.stop()

            df = pd.concat(
                tables_data,
                ignore_index=True
            )

        else:
            st.error("Unsupported file")
            st.stop()

    except Exception as e:
        st.error(f"File Error: {e}")
        st.stop()

    # ==========================================
    # CLEAN COLUMN NAMES
    # ==========================================

    df.columns = (
        df.columns.astype(str)
        .str.strip()
    )

    # ==========================================
    # DETECT COLUMNS
    # ==========================================

    description_col = None
    amount_col = None
    date_col = None

    for col in df.columns:

        lower = col.lower()

        if any(x in lower for x in
               ["description", "details", "narration", "remarks"]):
            description_col = col

        if any(x in lower for x in
               ["amount", "debit", "credit"]):
            amount_col = col

        if any(x in lower for x in
               ["date", "time"]):
            date_col = col

    # ==========================================
    # MANUAL DESCRIPTION
    # ==========================================

    if description_col is None:

        description_col = st.selectbox(
            "Select Description Column",
            df.columns
        )

    # ==========================================
    # RENAME COLUMNS
    # ==========================================

    df.rename(
        columns={description_col: "Description"},
        inplace=True
    )

    # ==========================================
    # AMOUNT
    # ==========================================

    if amount_col:

        df.rename(
            columns={amount_col: "Amount"},
            inplace=True
        )

        df["Amount"] = (
            df["Amount"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("₹", "", regex=False)
            .str.replace("$", "", regex=False)
        )

        df["Amount"] = pd.to_numeric(
            df["Amount"],
            errors="coerce"
        )

        df["Amount"] = df["Amount"].fillna(0)

    else:

        df["Amount"] = 100

    # ==========================================
    # DATE
    # ==========================================

    if date_col:

        df.rename(
            columns={date_col: "Date"},
            inplace=True
        )

    # ==========================================
    # REMOVE EMPTY DESCRIPTIONS
    # ==========================================

    df = df.dropna(
        subset=["Description"]
    )

    if len(df) == 0:
        st.error("No valid descriptions found")
        st.stop()

    # ==========================================
    # PREDICTION
    # ==========================================

    try:

        df["Predicted Category"] = model.predict(
            df["Description"].astype(str)
        )

    except Exception as e:

        st.error(f"Prediction Failed: {e}")
        st.stop()

    # ==========================================
    # SIDEBAR
    # ==========================================

    st.sidebar.header("📌 Key Insights")

    st.sidebar.metric(
        "Transactions",
        len(df)
    )

    st.sidebar.metric(
        "Categories",
        df["Predicted Category"].nunique()
    )

    st.sidebar.metric(
        "Total Expense",
        f"₹{df['Amount'].sum():,.2f}"
    )

    # ==========================================
    # RESULTS TABLE
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
    # MONTHLY GRAPH
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
    # DOWNLOAD
    # ==========================================

    csv = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇ Download Results",
        csv,
        "predicted_expenses.csv",
        "text/csv"
    )
