import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

try:
    response = requests.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}"}
    )
    
    if response.status_code == 200:
        data = response.json()
        all_models = data["data"]
        
        # Search specifically for llama-2
        llama2_models = [m["id"] for m in all_models if "llama-2" in m["id"].lower()]
        
        # Search for 13b-chat models
        chat13b_models = [m["id"] for m in all_models if "13b-chat" in m["id"].lower()]
        
        print(f"Total models found: {len(all_models)}")
        
        print("\n--- 'llama-2' matches ---")
        if llama2_models:
            for m in llama2_models:
                print(m)
        else:
            print("No models matching 'llama-2' found.")
            
        print("\n--- '13b-chat' matches ---")
        if chat13b_models:
            for m in chat13b_models:
                print(m)
        else:
            print("No models matching '13b-chat' found.")
            
    else:
        print(f"Error: {response.status_code} - {response.text}")
        
except Exception as e:
    print(f"An error occurred: {e}")
