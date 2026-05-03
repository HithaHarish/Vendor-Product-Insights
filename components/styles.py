import streamlit as st

def load_styles():
    st.markdown("""
        <style>

        /* Remove ALL top padding inside sidebar */
        section[data-testid="stSidebar"] > div {
            padding-top: 0rem !important;
            margin-top: 0rem !important;
        }

        section[data-testid="stSidebar"] div:first-child {
            margin-top: 0rem !important;
            padding-top: 0rem !important;
        }

        .block-container {
            padding-top: 2.2rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }

        .main {
            background-color: #f6f8fb;
        }

        .header-title {
            font-size: 30px;
            font-weight: 600;
            color: #1f2f3d;
            margin-top: 0px;
            margin-bottom: 14px;
        }

        .section-heading {
            font-size: 20px;
            font-weight: 600;
            color: #243447;
            margin-bottom: 6px;
            margin-top: 18px;
        }

        .section-underline {
            width: 80px;
            height: 3px;
            background-color: #6fa3d6; /* added mild accent color */
            margin-bottom: 18px;
        }

        .section-text {
            font-size: 16px;
            font-weight: 400;
            line-height: 1.75;
            color: #2a3b4d;
        }

        /* NEW: Model Body Style */
        .model-body {
            font-size: 17px;
            font-weight: 400;
            line-height: 1.8;
            color: #2a3b4d;
        }

        /* NEW: Dataset Heading */
        .dataset-heading {
            font-size: 20px;
            font-weight: 600;
            color: #243447;
            margin-top: 18px;
            margin-bottom: 8px;
        }

        ul.custom-list {
            padding-left: 15px;
            margin-top: 6px;
        }

        ul.model-list {
            padding-left: 22px;
            margin-top: 6px;
        }

        ul.custom-list li {
            margin-bottom: 26px;
            line-height: 1.9;
        }

        ul.model-list li {
            margin-bottom: 8px;
            line-height: 1.7;
        }

        /* Cleaner cards/containers for professional feel */
        div[data-testid="stExpander"] {
            border: 1px solid #dbe3ec;
            border-radius: 10px;
            background: #ffffff;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid #dbe3ec;
            border-radius: 8px;
        }

        .stButton > button {
            border-radius: 8px;
            border: 1px solid #c8d4e0;
            background: #edf2f7;
            color: #1f3348;
            font-weight: 600;
        }

        .stButton > button:hover {
            border-color: #aebfd1;
            background: #e3ebf3;
        }

        section[data-testid="stSidebar"] {
            width: 400px !important;
            min-width: 400px !important;
            max-width: 420px !important;
            padding-top: 0.5rem !important;
            border-right: 1px solid #dbe3ec;
        }

        </style>
        """, unsafe_allow_html=True)