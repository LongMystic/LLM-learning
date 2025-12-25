"""
Basic LLM API calls - both hosted (OpenAI) and local (Ollama)
Run this after setting up your API keys and/or Ollama
"""

import os
from openai import OpenAI
import requests
from dotenv import load_dotenv

load_dotenv()  # Load API keys from .env file

# ============================================
# Option 1: OpenAI (hosted API)
# ============================================

def call_openai(messages, model="gpt-4o-mini", temperature=0.7, max_tokens=256):
    """
    Call OpenAI API
    
    Args:
        messages: List of message dicts, e.g. [{"role": "user", "content": "Hello"}]
        model: Model name (gpt-4o-mini is cheaper, gpt-4o is more powerful)
        temperature: 0.0 (deterministic) to 2.0 (very creative)
        max_tokens: Maximum tokens in response
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    return response.choices[0].message.content


# ============================================
# Option 2: Ollama (local, free)
# ============================================

def call_ollama(prompt, model="llama3.2", temperature=0.7, max_tokens=256):
    """
    Call Ollama running locally
    
    First, install Ollama: https://ollama.ai
    Then run: ollama pull llama3.2
    
    Args:
        prompt: Plain text prompt (not messages format)
        model: Model name (llama3.2, mistral, etc.)
        temperature: 0.0 to 1.0
        max_tokens: Maximum tokens in response
    """
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "num_predict": max_tokens,
        "stream": False,  # Set to True for streaming responses
    }
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    return response.json()["response"]


# ============================================
# Example usage
# ============================================

if __name__ == "__main__":
    # Example 1: OpenAI
    print("=== OpenAI Example ===")
    messages = [
        {"role": "user", "content": "What is a transformer in machine learning? Answer in one sentence."}
    ]
    try:
        result = call_openai(messages, temperature=0.3)
        print(result)
    except Exception as e:
        print(f"OpenAI error (check your API key): {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Ollama
    print("=== Ollama Example ===")
    prompt = "What is a transformer in machine learning? Answer in one sentence."
    try:
        result = call_ollama(prompt, temperature=0.3)
        print(result)
    except Exception as e:
        print(f"Ollama error (is it running?): {e}")
        print("To start Ollama, run: ollama serve")

