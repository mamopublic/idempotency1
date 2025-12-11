import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    print("Error: OPENROUTER_API_KEY not found.")
    exit(1)

response = requests.get(
    "https://openrouter.ai/api/v1/models",
    headers={"Authorization": f"Bearer {api_key}"}
)

if response.status_code == 200:
    models = response.json()["data"]
    gemini_models = [m["id"] for m in models if "gemini" in m["id"].lower()]
    print("Found Gemini models:")
    for m in sorted(gemini_models):
        print(f"- {m}")
else:
    print(f"Error fetching models: {response.status_code} - {response.text}")
