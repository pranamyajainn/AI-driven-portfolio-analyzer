from flask import Flask, request, jsonify, render_template
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Environment variables
LYZR_API_KEY = os.getenv("LYZR_API_KEY")
LYZR_API_ENDPOINT = os.getenv("LYZR_API_ENDPOINT")

# Routes
@app.route("/")
def home():
    # Serve the HTML file directly
    return render_template("index.html")

@app.route("/api/analyze", methods=["POST"])
def analyze_portfolio():
    """
    This route will handle form submissions from the frontend and 
    process data using Lyzr's API.
    """
    try:
        # Parse JSON data from frontend
        data = request.json
        risk_tolerance = data.get("risk_tolerance")
        financial_goals = data.get("financial_goals")
        timeline = data.get("timeline")
        current_financials = data.get("current_financials")

        # Prepare the payload for Lyzr API
        payload = {
            "risk_tolerance": risk_tolerance,
            "financial_goals": financial_goals,
            "timeline": timeline,
            "current_financials": current_financials,
        }

        # Send the payload to Lyzr API
        headers = {
            "Authorization": f"Bearer {LYZR_API_KEY}",
            "Content-Type": "application/json"
        }
        response = requests.post(f"{LYZR_API_ENDPOINT}/analyze", json=payload, headers=headers)

        # Handle the response from Lyzr API
        if response.status_code == 200:
            return jsonify({"success": True, "data": response.json()})
        else:
            return jsonify({"success": False, "error": response.text}), response.status_code

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/recommendations", methods=["GET"])
def get_recommendations():
    """
    This route fetches recommendations from Lyzr API.
    """
    try:
        # Fetch recommendations (mock endpoint)
        headers = {
            "Authorization": f"Bearer {LYZR_API_KEY}",
            "Content-Type": "application/json"
        }
        response = requests.get(f"{LYZR_API_ENDPOINT}/recommendations", headers=headers)

        if response.status_code == 200:
            return jsonify({"success": True, "data": response.json()})
        else:
            return jsonify({"success": False, "error": response.text}), response.status_code

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# Run the Flask server
if __name__ == "__main__":
    app.run(debug=True)
