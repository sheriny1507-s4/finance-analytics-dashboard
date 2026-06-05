import pandas as pd
import os

USER_FILE = "../data/users.csv"

def load_users():

    if not os.path.exists(USER_FILE):

        pd.DataFrame(
            columns=["username", "password"]
        ).to_csv(USER_FILE, index=False)

    return pd.read_csv(USER_FILE)

def register_user(username, password):

    users = load_users()

    if username in users["username"].values:
        return False

    users.loc[len(users)] = [username, password]

    users.to_csv(USER_FILE, index=False)

    return True

def login_user(username, password):

    users = load_users()

    user = users[
        (users["username"] == username) &
        (users["password"] == password)
    ]

    return len(user) > 0