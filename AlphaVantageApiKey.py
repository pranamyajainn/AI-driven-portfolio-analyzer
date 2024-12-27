import requests
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from datetime import datetime

# Load environment variables
load_dotenv()

# Alpha Vantage API Key
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"

app = Flask(__name__)

def calculate_metrics(time_series):
    """
    Calculate metrics like price change and volatility from the time series data.
    """
    try:
        prices = [float(details["4. close"]) for details in time_series.values()]
        dates = list(time_series.keys())

        # Ensure we have enough data to calculate metrics
        if len(prices) < 2:
            return None

        # Calculate metrics
        price_change = ((prices[0] - prices[-1]) / prices[-1]) * 100  # From most recent to oldest
        volatility = (max(prices) - min(prices)) / min(prices) * 100  # High-low range percentage

        return {
            "price_change_percentage": round(price_change, 2),
            "volatility_percentage": round(volatility, 2),
            "start_date": dates[-1],
            "end_date": dates[0],
            "latest_close_price": round(prices[0], 2),
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None

@app.route("/api/market_data", methods=["GET"])
def get_market_data():
    """
    Fetch market data based on the symbol provided by the frontend.
    """
    try:
        # Retrieve the symbol from request arguments
        symbol = request.args.get("symbol", None)
        if not symbol:
            return jsonify({"success": False, "error": "Symbol parameter is required"}), 400

        # Define the API request parameters
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "apikey": ALPHA_VANTAGE_API_KEY,
        }

        # Make a request to the Alpha Vantage API
        response = requests.get(BASE_URL, params=params)

        # Check if the response is successful
        if response.status_code != 200:
            return jsonify({"success": False, "error": "Failed to fetch data from Alpha Vantage"}), response.status_code

        data = response.json()

        # Check if the response contains the expected data
        if "Time Series (Daily)" not in data:
            return jsonify({"success": False, "error": "Invalid data format received"}), 400

        # Extract time series data
        time_series = data["Time Series (Daily)"]

        # Calculate additional metrics
        metrics = calculate_metrics(time_series)
        if metrics is None:
            return jsonify({"success": False, "error": "Insufficient data to calculate metrics"}), 400

        # Extract and format the market data
        formatted_data = [
            {
                "date": date,
                "open": float(details["1. open"]),
                "high": float(details["2. high"]),
                "low": float(details["3. low"]),
                "close": float(details["4. close"]),
                "volume": int(details["5. volume"]),
            }
            for date, details in time_series.items()
        ]

        return jsonify({"success": True, "data": formatted_data, "metrics": metrics})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
