import streamlit as st

def login():

    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if username.strip() == "admin" and password.strip() == "admin123":

            st.session_state["logged_in"] = True
            st.session_state["username"] = username

            st.success("Login Successful")
            st.rerun()

        else:
            st.error("Invalid Username or Password")
