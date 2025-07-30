import streamlit as st

st.set_page_config(
    page_title = "Trading App",
    page_icon = ":chart_with_downwards_trend:",
    layout = "wide"
)

st.title("Trading Guide App :bar_chart:")

st.header("We provide the greatest platform for you to collect all information prior to investing in stocks.")

st.image("app.png")

st.markdown("## What We Do:")

st.markdown("#### one: Stock Information")
st.write("Through this page, you can see all the information about stock.")

st.markdown("#### two: Stock Prediction")
st.write("You can explore predicted closing prices for the next 30 days based on historical stock data and advanced forecasting models.")

st.markdown("#### three: Stock Analysis")
st.write("Fundamental analysis of stocks focuses on the company itself and seeks to determine the true, fair value of a company's stock price, based on recent earnings, past earnings growth rates, projected earnings, multiples like PE ratio, debts and other financial and accounting measures")
