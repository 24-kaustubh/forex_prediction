from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Function to fetch historical forex data
def fetch_currency_data(currency_pair, start_date="2023-01-01", end_date="2024-01-01"):
    data = yf.download(currency_pair, start=start_date, end=end_date, interval="1d")
    data = data[['Close']]
    data.rename(columns={'Close': 'Price'}, inplace=True)
    return data

# Function to prepare data for training
def prepare_data(df, n_lags=5):
    for i in range(1, n_lags + 1):
        df[f'Lag_{i}'] = df['Price'].shift(i)
    df.dropna(inplace=True)
    return df

# Function to train the Linear Regression model
def train_model(df):
    features = [f'Lag_{i}' for i in range(1, 6)]
    X = df[features]
    y = df['Price']
    model = LinearRegression()
    model.fit(X, y)
    return model

# Function to predict future prices
def predict_future_price(df, model, days=5):
    features = [f'Lag_{i}' for i in range(1, 6)]
    last_row = df[features].iloc[-1].values.reshape(1, -1)
    predictions = []

    for _ in range(days):
        next_price = model.predict(last_row)[0]
        predictions.append(round(next_price, 4))
        last_row = np.roll(last_row, -1)
        last_row[0, -1] = next_price

    return predictions

@app.route("/", methods=["GET", "POST"])
def index():
    predictions = None
    currency_pair = None

    if request.method == "POST":
        currency_pair = request.form["currency"].upper() + "=X"
        df = fetch_currency_data(currency_pair)

        if df is not None and not df.empty:
            df = prepare_data(df)
            model = train_model(df)
            predictions = predict_future_price(df, model, days=5)

    return render_template("index.html", predictions=predictions, currency_pair=currency_pair)

if __name__ == "__main__":
    app.run(debug=True)
