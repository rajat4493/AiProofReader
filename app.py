import streamlit as st

st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    .stTextArea textarea {
        max-height: 260px;
        overflow-y: auto !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

long_text = "\n".join([f"Line {i}" for i in range(1, 200)])

st.text_area("Test scroll", long_text, height=200)
