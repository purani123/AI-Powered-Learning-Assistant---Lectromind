# check_gemini_models.py
import os
import google.generativeai as genai
import sys

# Get API key from environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY environment variable not found.")
    print("Please set your API key as an environment variable and restart your terminal.")
    sys.exit(1)

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Gemini API configured. Listing available models...")
    
    # List all available models
    for m in genai.list_models():
        # Only print models that support generateContent (for text generation)
        if "generateContent" in m.supported_generation_methods:
            print(f"Name: {m.name}")
            print(f"  Description: {m.description}")
            print(f"  Input Token Limit: {m.input_token_limit}")
            print(f"  Output Token Limit: {m.output_token_limit}")
            print(f"  Supported Methods: {m.supported_generation_methods}\n")

except Exception as e:
    print(f"An error occurred while listing models: {e}")
    print("This might indicate an issue with your API key, network, or access permissions.")