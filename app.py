import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from tensorflow.keras.models import load_model
from joblib import load

# Load the saved LSTM model and scaler
lstm_model = load_model('lstm_model.h5')
scaler = load('scaler.joblib')

# Dictionary of stock tickers and their full names
stock_dict = {
    'GOOG': 'Alphabet Inc.',
    'AAPL': 'Apple Inc.',
    'TSLA': 'Tesla, Inc.',
    'AMZN': 'Amazon.com, Inc.',
    'MSFT': 'Microsoft Corporation'
}

# Function to get the last row of stock data
def get_last_stock_data(ticker):
    try:
        start_date = '2010-01-01'
        end_date = datetime.now().strftime('%Y-%m-%d')
        data = yf.download(ticker, start=start_date, end=end_date)
        last_row = data.iloc[-1]
        return last_row.to_dict()
    except Exception as e:
        return str(e)

# Function to make predictions
def predict_stock_price(ticker, open_price, close_price):
    try:
        start_date = '2010-01-01'
        end_date = datetime.now().strftime('%Y-%m-%d')
        data = yf.download(ticker, start=start_date, end=end_date)

        # Prepare the data
        data = data[['Close']]
        dataset = data.values
        scaled_data = scaler.transform(dataset)

        # Append the user inputs as the last row in the data
        user_input = np.array([[close_price]])
        user_input_scaled = scaler.transform(user_input)
        scaled_data = np.vstack([scaled_data, user_input_scaled])

        # Prepare the data for LSTM
        x_test_lstm = []
        for i in range(60, len(scaled_data)):
            x_test_lstm.append(scaled_data[i-60:i])

        x_test_lstm = np.array(x_test_lstm)
        x_test_lstm = np.reshape(x_test_lstm, (x_test_lstm.shape[0], x_test_lstm.shape[1], 1))

        # LSTM Predictions
        lstm_predictions = lstm_model.predict(x_test_lstm)
        lstm_predictions = scaler.inverse_transform(lstm_predictions)
        next_day_lstm_price = lstm_predictions[-1][0]
        
        result = f"Predicted future price for {ticker}: ${next_day_lstm_price:.2f}"

        return result
    except Exception as e:
        return str(e)

# Function to predict next month's price
def predict_next_month_price(ticker, close_price):
    try:
        start_date = '2010-01-01'
        end_date = datetime.now().strftime('%Y-%m-%d')
        data = yf.download(ticker, start=start_date, end=end_date)

        # Prepare the data
        data = data[['Close']]
        dataset = data.values
        scaled_data = scaler.transform(dataset)

        # Append the user inputs as the last row in the data
        user_input = np.array([[close_price]])
        user_input_scaled = scaler.transform(user_input)
        scaled_data = np.vstack([scaled_data, user_input_scaled])

        # Prepare the data for LSTM
        x_test_lstm = []
        for i in range(60, len(scaled_data)):
            x_test_lstm.append(scaled_data[i-60:i])

        x_test_lstm = np.array(x_test_lstm)
        x_test_lstm = np.reshape(x_test_lstm, (x_test_lstm.shape[0], x_test_lstm.shape[1], 1))

        # Predicting the next 30 days
        predictions = []
        for _ in range(30):
            pred = lstm_model.predict(x_test_lstm[-1].reshape(1, 60, 1))
            predictions.append(pred[0])
            new_input = np.append(x_test_lstm[-1][1:], pred)
            x_test_lstm = np.append(x_test_lstm, new_input.reshape(1, 60, 1), axis=0)

        predictions = np.array(predictions)
        next_month_predictions = scaler.inverse_transform(predictions)
        next_month_price = next_month_predictions[-1][0]

        result = f"Predicted price for {ticker} next month: ${next_month_price:.2f}"

        return result
    except Exception as e:
        return str(e)

# Function to display historical data
def display_historical_data(ticker):
    try:
        start_date = '2010-01-01'
        end_date = datetime.now().strftime('%Y-%m-%d')
        data = yf.download(ticker, start=start_date, end=end_date)
        return data.tail(30).iloc[::-1]  # Reverse to have the latest date on top
    except Exception as e:
        return str(e)

# Streamlit interface
st.title("Stockstream")

# Sidebar for adding new stocks
st.sidebar.header("Add a New Stock Ticker")
new_ticker = st.sidebar.text_input("Stock Ticker", value="")
new_full_name = st.sidebar.text_input("Full Name", value="")
if st.sidebar.button("Add Stock Ticker"):
    if new_ticker and new_full_name:
        stock_dict[new_ticker.upper()] = new_full_name

# Sidebar for viewing historical trends
st.sidebar.header("View Historical Trends")
historical_ticker_input = st.sidebar.selectbox("Stock Ticker", [f"{key} - {value}" for key, value in stock_dict.items()], key="sidebar_historical_ticker")
if st.sidebar.button("View Historical Data"):
    ticker = historical_ticker_input.split(' - ')[0]
    data = display_historical_data(ticker)
    st.sidebar.line_chart(data['Close'])

# Tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Today's Price", "Next Month's Price", "Historical Data"])

with tab1:
    st.header("Today's Price")
    ticker_input = st.selectbox("Stock Ticker", [f"{key} - {value}" for key, value in stock_dict.items()], key="today_ticker")
    open_price = st.number_input("Open Price", value=0.0, key="today_open_price")
    close_price = st.number_input("Close Price", value=0.0, key="today_close_price")
    if st.button("Predict Today's Price"):
        ticker = ticker_input.split(' - ')[0]
        result = predict_stock_price(ticker, open_price, close_price)
        st.write(result)

with tab2:
    st.header("Next Month's Price")
    next_month_ticker_input = st.selectbox("Stock Ticker", [f"{key} - {value}" for key, value in stock_dict.items()], key="next_month_ticker")
    next_month_close_price = st.number_input("Close Price", value=0.0, key="next_month_close_price")
    if st.button("Predict Next Month's Price"):
        ticker = next_month_ticker_input.split(' - ')[0]
        result = predict_next_month_price(ticker, next_month_close_price)
        st.write(result)

with tab3:
    st.header("Historical Data")
    historical_ticker_input = st.selectbox("Stock Ticker", [f"{key} - {value}" for key, value in stock_dict.items()], key="historical_ticker")
    if st.button("View Data"):
        ticker = historical_ticker_input.split(' - ')[0]
        data = display_historical_data(ticker)
        st.dataframe(data)
