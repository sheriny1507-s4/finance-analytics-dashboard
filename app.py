import streamlit as st
import pandas as pd
import pickle
import os
import pdfplumber
import matplotlib.pyplot as plt
from login import login

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Personal Finance Dashboard",
    layout="wide"
)

# ==========================
# LOGIN
# ==========================
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
    st.stop()

# ==========================
# HEADER
# ==========================
st.title("💰 Personal Finance Analytics Dashboard")

st.sidebar.success(
    f"Welcome {st.session_state.get('username','User')}"
)

if st.sidebar.button("Logout"):
    st.session_state["logged_in"] = False
    st.rerun()

# ==========================
# MODEL
# ==========================
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

# ==========================
# FILE UPLOAD
# ==========================
uploaded_file = st.file_uploader(
    "Upload CSV, Excel or PDF",
    type=["csv", "xlsx", "pdf"]
)

# ==========================
# PROCESS FILE
# ==========================
if uploaded_file is not None:

    try:

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)

        elif uploaded_file.name.endswith(".pdf"):

            pdf_data = []

            with pdfplumber.open(uploaded_file) as pdf:

                for page in pdf.pages:

                    text = page.extract_text()

                    if text:
                        lines = text.split("\n")

                        temp_df = pd.DataFrame(
                            {"Description": lines}
                        )

                        pdf_data.append(temp_df)

            if len(pdf_data) == 0:
                st.error("No readable data found")
                st.stop()

            df = pd.concat(
                pdf_data,
                ignore_index=True
            )

        else:
            st.error("Unsupported file")
            st.stop()

    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # ==========================
    # CLEAN COLUMNS
    # ==========================
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    description_col = None
    amount_col = None
    date_col = None

    for col in df.columns:

        c = col.lower()

        if any(x in c for x in [
            "description",
            "details",
            "narration",
            "remarks"
        ]):
            description_col = col

        if any(x in c for x in [
            "amount",
            "debit",
            "credit"
        ]):
            amount_col = col

        if "date" in c:
            date_col = col

    if description_col is None:

        st.warning(
            "Description column not detected."
        )

        description_col = st.selectbox(
            "Select Description Column",
            df.columns
        )

    # ==========================
    # DESCRIPTION
    # ==========================
    df.rename(
        columns={description_col: "Description"},
        inplace=True
    )

    df["Description"] = (
        df["Description"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["Description"] != ""
    ]

    # ==========================
    # AMOUNT
    # ==========================
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
        df["Amount"] = 0

    # ==========================
    # DATE
    # ==========================
    if date_col:

        df.rename(
            columns={date_col: "Date"},
            inplace=True
        )

    # ==========================
    # EMPTY CHECK
    # ==========================
    if len(df) == 0:

        st.error(
            "No valid transactions found."
        )

        st.stop()

    # ==========================
    # PREDICTION
    # ==========================
    try:

        df["Predicted Category"] = model.predict(
            df["Description"]
        )

    except Exception as e:

        st.error(
            f"Prediction Failed: {e}"
        )

        st.stop()

    # ==========================
    # SIDEBAR
    # ==========================
    st.sidebar.header("Key Insights")

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
        f"${df['Amount'].sum():,.2f}"
    )

    # ==========================
    # TABLE
    # ==========================
    st.subheader(
        "Prediction Results"
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    st.subheader("Debug Information")

st.write("Columns:")
st.write(df.columns.tolist())

st.write("Amount Sum:")
st.write(df["Amount"].sum())

st.write("Category Expense:")
st.write(
    df.groupby("Predicted Category")["Amount"].sum()
)
    # ==========================
    # PIE CHART
    # ==========================
    category_expense = (
        df.groupby(
            "Predicted Category"
        )["Amount"]
        .sum()
    )

    if category_expense.sum() > 0:

        st.subheader(
            "Expense Distribution"
        )

        fig, ax = plt.subplots()

        ax.pie(
            category_expense,
            labels=category_expense.index,
            autopct="%1.1f%%"
        )

        st.pyplot(fig)

    # ==========================
    # DOWNLOAD
    # ==========================
    csv = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇ Download Results",
        csv,
        "predicted_expenses.csv",
        "text/csv"
    )
