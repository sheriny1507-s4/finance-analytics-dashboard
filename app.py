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
# LOAD MODEL (TUPLE SAFE)
# ---------------------------------------------------
MODEL_PATH = r"C:\Users\Sherin Y\OneDrive\Desktop\finance_analytics\models\model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model not found at: {MODEL_PATH}")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    loaded_obj = pickle.load(f)

# HANDLE TUPLE (vectorizer, model)
if isinstance(loaded_obj, tuple):
    vectorizer, model = loaded_obj
else:
    vectorizer = None
    model = loaded_obj

st.sidebar.success("✅ Model Loaded")

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload CSV, Excel or PDF", type=["csv", "xlsx", "pdf"])

# ---------------------------------------------------
# MAIN LOGIC
# ---------------------------------------------------
if uploaded_file is not None:

    # ---------------- READ FILE ----------------
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)

        elif uploaded_file.name.endswith(".pdf"):
            text_data = ""
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_data += text + "\n"

            lines = text_data.split("\n")
            df = pd.DataFrame(lines, columns=["Description"])

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

    # ---------------- AUTO DETECT COLUMNS ----------------
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
        if vectorizer is not None:
            X = vectorizer.transform(df["Description"].astype(str))
            df["Predicted Category"] = model.predict(X)
        else:
            df["Predicted Category"] = model.predict(df["Description"].astype(str))

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
    # PIE CHART
    # ---------------------------------------------------
    st.subheader("📊 Expense Distribution by Category")

    category_counts = df["Predicted Category"].value_counts()

    if not category_counts.empty:
        fig1, ax1 = plt.subplots()
        ax1.pie(category_counts.values, labels=category_counts.index, autopct="%1.1f%%")
        ax1.axis("equal")
        st.pyplot(fig1)
    else:
        st.warning("No category data to display")

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