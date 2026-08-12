import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
print(f"API Key loaded: {'Yes' if api_key and api_key != 'your-api-key-here' else 'No (or default)'}")

try:
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        max_output_tokens=16,
        input="Return the word hello.",
    )
    print("API call successful!")
    print(response.output_text)
except Exception as e:
    print(f"API call failed: {e}")
