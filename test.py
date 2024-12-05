import streamlit as st
import streamlit.components.v1 as components

# Title for your Streamlit app
st.title("Power BI Dashboard Embed")

# Embed Power BI report using iframe
power_bi_url = "https://app.powerbi.com/reportEmbed?reportId=f85c3ce5-e5b6-4275-bfea-05fc5c98ff68&appId=2b99fecb-2377-4f75-8496-792db7e2b66a&autoAuth=true&ctid=8d513303-4b89-4180-b233-bebb388ad37f"  # Replace with your Power BI embed link
components.iframe(power_bi_url, width=1000, height=600)