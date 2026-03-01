
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

# ---------------------------------------------------
# LOAD MODEL (pipeline)
# ---------------------------------------------------
MODEL_PATH = "models/model.pkl"

model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    st.error("❌ Model not found in models folder")

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload CSV, Excel or PDF", type=["csv", "xlsx", "pdf"])

# ---------------------------------------------------
# MAIN LOGIC
# ---------------------------------------------------
if uploaded_file is not None:
    try:
        # ---------------- READ FILE ----------------
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".pdf"):
            text_data = []
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            # Convert each table into a DataFrame
                            temp_df = pd.DataFrame(table[1:], columns=table[0])
                            text_data.append(temp_df)
                    else:
                        # Fallback: extract text lines if no tables
                        lines = page.extract_text().split("\n")
                        temp_df = pd.DataFrame(lines, columns=["Description"])
                        text_data.append(temp_df)
            df = pd.concat(text_data, ignore_index=True)
        else:
            st.error("Unsupported file type")
            st.stop()
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()

    # ---------------- CLEAN COLUMNS ----------------
    df.columns = df.columns.astype(str)
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(r'\s+', ' ', regex=True)

    description_col = None
    amount_col = None
    date_col = None

    for col in df.columns:
        col_lower = col.lower()
        if any(k in col_lower for k in ["description", "details", "narration", "remarks"]):
            description_col = col
        if any(k in col_lower for k in ["amount", "debit", "credit", "withdrawal", "bill"]):
            amount_col = col
        if any(k in col_lower for k in ["date", "time"]):
            date_col = col

    if description_col is None:
        st.error("❌ No Description column found")
        st.stop()

    df.rename(columns={description_col: "Description"}, inplace=True)

    if amount_col:
        df.rename(columns={amount_col: "Amount"}, inplace=True)
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

    if date_col:
        df.rename(columns={date_col: "Date"}, inplace=True)

    df = df.dropna(subset=["Description"])

    # ---------------- PREDICTION ----------------
    try:
        # Always pass raw text to the pipeline
        df["Predicted Category"] = model.predict(df["Description"].astype(str).tolist())
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.stop()

    # ---------------------------------------------------
    # SIDEBAR INSIGHTS
    # ---------------------------------------------------
    st.sidebar.header("📌 Key Insights")
    st.sidebar.write("Total Transactions:", len(df))
    st.sidebar.write("Unique Categories:", df["Predicted Category"].nunique())

    if "Amount" in df.columns:
        total_expense = df["Amount"].sum()
        st.sidebar.write("Total Expense:", round(total_expense, 2))
    else:
        st.sidebar.write("Total Expense: N/A")

    # ---------------------------------------------------
    # TABLE OUTPUT
    # ---------------------------------------------------
    st.subheader("📋 Prediction Results")
    st.dataframe(df, use_container_width=True)

    # ---------------------------------------------------
    # PIE CHART WITH TOTALS
    # ---------------------------------------------------
    st.subheader("📊 Expense Distribution by Category")

    if "Amount" in df.columns:
        category_expense = df.groupby("Predicted Category")["Amount"].sum()
        if not category_expense.empty:
            fig1, ax1 = plt.subplots()
            labels = [f"{cat} (${amt:.2f})" for cat, amt in category_expense.items()]
            ax1.pie(category_expense.values, labels=labels, autopct="%1.1f%%")
            ax1.axis("equal")
            st.pyplot(fig1)
        else:
            st.warning("No category expense data to display")
    else:
        st.warning("No Amount column available for expense distribution")

    # ---------------------------------------------------
    # MONTHLY TREND
    # ---------------------------------------------------
    if "Date" in df.columns and "Amount" in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date", "Amount"])
            df["Month"] = df["Date"].dt.to_period("M")
            monthly_expense = df.groupby("Month")["Amount"].sum()

            if not monthly_expense.empty:
                st.subheader("📈 Monthly Expense Trend")
                fig2, ax2 = plt.subplots()
                monthly_expense.plot(kind="line", ax=ax2)
                st.pyplot(fig2)
            else:
                st.warning("No monthly data available")
        except Exception as e:
            st.warning(f"⚠ Could not generate monthly graph: {e}")

    # ---------------------------------------------------
    # DOWNLOAD BUTTON
    # ---------------------------------------------------
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download Results as CSV",
        csv,
        "predicted_expenses.csv",
        "text/csv"
    )

