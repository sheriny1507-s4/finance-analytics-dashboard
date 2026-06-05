import streamlit as st
import pandas as pd
import os

USER_FILE = "users.csv"

def create_user_file():
    if not os.path.exists(USER_FILE):
        df = pd.DataFrame({
            "username": ["admin"],
            "password": ["admin123"]
        })
        df.to_csv(USER_FILE, index=False)

def login():

    create_user_file()

    st.title("🔐 Login System")

    menu = st.sidebar.selectbox(
        "Menu",
        ["Login", "Register"]
    )

    users = pd.read_csv(USER_FILE)

    if menu == "Login":

        username = st.text_input("Username")
        password = st.text_input(
            "Password",
            type="password"
        )

        if st.button("Login"):

            valid_user = users[
                (users["username"] == username)
                &
                (users["password"] == password)
            ]

            if len(valid_user) > 0:

                st.session_state["logged_in"] = True
                st.session_state["username"] = username

                st.success("Login Successful")
                st.rerun()

            else:
                st.error("Invalid Username or Password")

    elif menu == "Register":

        new_user = st.text_input("Create Username")
        new_pass = st.text_input(
            "Create Password",
            type="password"
        )

        if st.button("Register"):

            if new_user in users["username"].values:

                st.warning(
                    "Username already exists"
                )

            else:

                users.loc[len(users)] = [
                    new_user,
                    new_pass
                ]

                users.to_csv(
                    USER_FILE,
                    index=False
                )

                st.success(
                    "Registration Successful"
                )
