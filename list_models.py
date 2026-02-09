import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

response = requests.get(
    "https://openrouter.ai/api/v1/models",
    headers={"Authorization": f"Bearer {api_key}"}
)

if response.status_code == 200:
    data = response.json()
    print("Available models:")
    for model in data["data"]:
        # Print all models, or refine filter if needed
        # Just searching for 'llama' generally
        if "llama" in model["id"].lower():
             print(f"- {model['id']}")
else:
    print(f"Error: {response.status_code} - {response.text}")
