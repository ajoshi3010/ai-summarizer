import streamlit as st
from app1 import app1_main
from app2 import app2_main

# Sidebar for navigation
st.sidebar.title("Navigation")
selected_app = st.sidebar.selectbox("Choose an App", ["PDF Summarizer", "Youtube Video Summarizer"])

# Display selected app
if selected_app == "PDF Summarizer":
    app1_main()
elif selected_app == "Youtube Video Summarizer":
    app2_main()
