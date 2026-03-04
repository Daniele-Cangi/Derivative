import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
print(f"API Key loaded: {'Yes' if api_key and api_key != 'your-api-key-here' else 'No (or default)'}")
if api_key:
    print(f"Key preview: {api_key[:8]}...{api_key[-4:]}")

try:
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=10,
        messages=[{"role": "user", "content": "Hello"}]
    )
    print("API call successful!")
    print(response.content[0].text)
except Exception as e:
    print(f"API call failed: {e}")
