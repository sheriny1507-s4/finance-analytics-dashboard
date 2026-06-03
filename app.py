import streamlit as st
import pandas as pd
import pickle
import os
import pdfplumber
import matplotlib.pyplot as plt

# ---------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------
st.set_page_config(page_title="Personal Finance Dashboard", layout="wide")

st.title("💰 Personal Finance Analytics Dashboard")

# LOAD MODEL
import pathlib

BASE_DIR = pathlib.Path(__file__).parent

MODEL_PATH = BASE_DIR / "models" / "model.pkl"

model = None

if MODEL_PATH.exists():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    st.error(f"❌ Model not found: {MODEL_PATH}")
    st.stop()

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload CSV, Excel or PDF",
    type=["csv", "xlsx", "pdf"]
)

# ---------------------------------------------------
# PROCESS FILE
# ---------------------------------------------------
if uploaded_file is not None:

    try:

        # ---------------- CSV ----------------
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        # ---------------- EXCEL ----------------
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)

        # ---------------- PDF ----------------
        elif uploaded_file.name.endswith(".pdf"):

            text_data = []

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

                                text_data.append(temp_df)

                    else:

                        page_text = page.extract_text()

                        if page_text:

                            lines = page_text.split("\n")

                            temp_df = pd.DataFrame(
                                {"Description": lines}
                            )

                            text_data.append(temp_df)

            if len(text_data) == 0:
                st.error("❌ No readable data found in PDF")
                st.stop()

            df = pd.concat(text_data, ignore_index=True)

            st.subheader("📄 Extracted PDF Data")
            st.write(df.head())

        else:
            st.error("Unsupported file type")
            st.stop()

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()

    # ---------------------------------------------------
    # CLEAN COLUMN NAMES
    # ---------------------------------------------------
    df.columns = df.columns.astype(str)

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    # ---------------------------------------------------
    # DETECT COLUMNS
    # ---------------------------------------------------
    description_col = None
    amount_col = None
    date_col = None

    for col in df.columns:

        col_lower = col.lower()

        if any(word in col_lower for word in
               ["description", "details", "narration", "remarks"]):
            description_col = col

        if any(word in col_lower for word in
               ["amount", "debit", "credit", "withdrawal", "bill"]):
            amount_col = col

        if any(word in col_lower for word in
               ["date", "time"]):
            date_col = col

    # ---------------------------------------------------
    # MANUAL DESCRIPTION SELECTION
    # ---------------------------------------------------
    if description_col is None:

        st.warning("⚠ Description column not detected")

        description_col = st.selectbox(
            "Select Description Column",
            df.columns.tolist()
        )

    # ---------------------------------------------------
    # RENAME COLUMNS
    # ---------------------------------------------------
    df.rename(
        columns={description_col: "Description"},
        inplace=True
    )

    if amount_col:

        df.rename(
            columns={amount_col: "Amount"},
            inplace=True
        )

        df["Amount"] = pd.to_numeric(
            df["Amount"],
            errors="coerce"
        )

        df = df.dropna(subset=["Amount"])

        df = df[df["Amount"] > 0]

    else:
        df["Amount"] = 0

    if date_col:

        df.rename(
            columns={date_col: "Date"},
            inplace=True
        )

    df = df.dropna(subset=["Description"])

    # ---------------------------------------------------
    # PREDICTION
    # ---------------------------------------------------
    try:

        df["Predicted Category"] = model.predict(
            df["Description"].astype(str)
        )

    except Exception as e:

        st.error(f"❌ Prediction failed: {e}")
        st.stop()

    # ---------------------------------------------------
    # SIDEBAR
    # ---------------------------------------------------
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

    # ---------------------------------------------------
    # TABLE
    # ---------------------------------------------------
    st.subheader("📋 Prediction Results")

    st.dataframe(
        df,
        use_container_width=True
    )

    # ---------------------------------------------------
    # PIE CHART
    # ---------------------------------------------------
    if "Amount" in df.columns:

        st.subheader(
            "📊 Expense Distribution by Category"
        )

        category_expense = (
            df.groupby("Predicted Category")["Amount"]
            .sum()
        )

        category_expense = category_expense[
            category_expense > 0
        ]

        if len(category_expense) > 0:

            try:

                fig1, ax1 = plt.subplots()

                ax1.pie(
                    category_expense.values,
                    labels=category_expense.index,
                    autopct="%1.1f%%"
                )

                ax1.axis("equal")

                st.pyplot(fig1)

            except:

                st.warning(
                    "Pie chart failed"
                )

    # ---------------------------------------------------
    # MONTHLY TREND
    # ---------------------------------------------------
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

    # ---------------------------------------------------
    # DOWNLOAD
    # ---------------------------------------------------
    csv = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇ Download Results as CSV",
        data=csv,
        file_name="predicted_expenses.csv",
        mime="text/csv"
    )
