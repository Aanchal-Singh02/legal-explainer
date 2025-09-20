# auth.py
import streamlit as st
import json
import pathlib

APP_DIR = pathlib.Path(__file__).parent
USERS_FILE = APP_DIR / "users.json"

# Ensure session_state keys exist
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "user_email" not in st.session_state:
    st.session_state["user_email"] = ""

def load_users():
    if not USERS_FILE.exists():
        with open(USERS_FILE, "w") as f:
            json.dump({}, f)
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def signup():
    st.subheader("Create a new account")
    email = st.text_input("Email", key="signup_email")
    password = st.text_input("Password", type="password", key="signup_pass")
    password2 = st.text_input("Confirm Password", type="password", key="signup_pass2")
    if st.button("Sign Up", key="signup_btn"):
        users = load_users()
        if not email or not password:
            st.error("Enter email and password")
        elif password != password2:
            st.error("Passwords do not match")
        elif email in users:
            st.error("User already exists")
        else:
            users[email] = password
            save_users(users)
            st.success("Signup successful! Please log in.")

def login():
    st.subheader("Login to your account")
    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_pass")
    login_clicked = st.button("Login", key="login_btn")

    if login_clicked:
        users = load_users()
        if email in users and users[email] == password:
            st.session_state["logged_in"] = True
            st.session_state["user_email"] = email
            st.success(f"Welcome {email}!")
        else:
            st.error("Invalid email or password")

def auth_ui():
    if not st.session_state["logged_in"]:
        choice = st.radio("Login or Sign Up?", ["Login", "Sign Up"])
        if choice == "Login":
            login()
        else:
            signup()
        st.stop()  # stop main app until logged in
