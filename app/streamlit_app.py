import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
from src.predict import predict

st.title("Sentiment Analysis")

text = st.text_area("Enter review")

if st.button("Analyze"):
    result = predict(text)
    st.write(result)