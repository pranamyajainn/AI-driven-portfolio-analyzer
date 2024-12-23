import requests

# Load .env file
from dotenv import load_dotenv
import os

load_dotenv()

# Get environment variables
LYZR_API_KEY = os.getenv('LYZR_API_KEY')
LYZR_API_ENDPOINT = os.getenv('LYZR_API_ENDPOINT')

# Example request to Lyzr API
headers = {
    "Authorization": f"Bearer {LYZR_API_KEY}",
    "Content-Type": "application/json"
}

response = requests.get(f"{LYZR_API_ENDPOINT}/test-endpoint", headers=headers)  # Replace '/test-endpoint' with the actual endpoint

if response.status_code == 200:
    print("API Response:", response.json())
else:
    print(f"Error {response.status_code}: {response.text}")