"""
Experiment with different temperature and top_p values
Shows how parameters affect output creativity/consistency
Uses Ollama (local) instead of OpenAI API
"""

from basic_api_calls import call_ollama
import csv
from datetime import datetime
import requests

def generate_with_params(prompt, temperature=None, top_p=None, model="llama3.2"):
    """
    Generate response with specific parameters using Ollama
    
    Args:
        prompt: The prompt to use (plain text)
        temperature: Controls randomness (0.0 = deterministic, 1.0 = very random)
        top_p: Nucleus sampling - alternative to temperature (0.1 = conservative, 1.0 = diverse)
        model: Ollama model name (default: llama3.2)
    
    Note: Ollama supports both temperature and top_p, but we'll use one at a time for comparison
    """
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "num_predict": 100,  # max tokens
        "stream": False,
    }
    
    # Set temperature or top_p (Ollama supports both, but we test them separately)
    if top_p is not None:
        payload["top_p"] = top_p
        # Don't set temperature when testing top_p
    elif temperature is not None:
        payload["temperature"] = temperature
    else:
        # Default if neither is provided
        payload["temperature"] = 0.7
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    return response.json()["response"]


def run_experiment():
    """
    Run the same prompt with different parameters
    Save results to CSV for comparison
    """
    # Test prompt
    test_prompt = "Write a one-sentence creative tagline for a coffee shop."
    
    # Different parameter values to test
    experiments = [
        {"name": "temp_0.0", "temperature": 0.0, "top_p": None},
        {"name": "temp_0.3", "temperature": 0.3, "top_p": None},
        {"name": "temp_0.7", "temperature": 0.7, "top_p": None},
        {"name": "temp_1.0", "temperature": 1.0, "top_p": None},
        {"name": "topp_0.1", "temperature": None, "top_p": 0.1},
        {"name": "topp_0.5", "temperature": None, "top_p": 0.5},
        {"name": "topp_0.9", "temperature": None, "top_p": 0.9},
    ]
    
    results = []
    
    print("Running experiments...")
    print("=" * 60)
    
    for exp in experiments:
        print(f"\n{exp['name']}:", end=" ")
        try:
            response = generate_with_params(
                test_prompt,
                temperature=exp["temperature"],
                top_p=exp["top_p"]
            )
            print(response[:80] + "..." if len(response) > 80 else response)
            
            results.append({
                "experiment": exp["name"],
                "temperature": exp["temperature"],
                "top_p": exp["top_p"],
                "response": response,
                "timestamp": datetime.now().isoformat(),
            })
        except Exception as e:
            print(f"Error: {e}")
    
    # Save to CSV
    csv_filename = f"parameter_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {csv_filename}")
    print("\nObservations:")
    print("- Lower temperature (0.0-0.3): More consistent, predictable responses")
    print("- Higher temperature (0.7-1.0): More creative, varied responses")
    print("- top_p works similarly: lower = more focused, higher = more diverse")
    
    return results


# ============================================
# Parameter explanations (for reference)
# ============================================

def explain_parameters():
    """
    Quick reference: What do these parameters do?
    """
    explanations = {
        "temperature": {
            "range": "0.0 to 1.0 (Ollama)",
            "meaning": "Controls randomness in token selection",
            "low (0.0-0.3)": "Deterministic, consistent outputs. Good for: classification, extraction, code generation",
            "medium (0.5-0.7)": "Balanced creativity. Good for: general chat, writing",
            "high (0.8-1.0)": "Very creative, unpredictable. Good for: creative writing, brainstorming",
            "note": "Ollama typically uses 0.0-1.0 range (vs OpenAI's 0.0-2.0)",
        },
        "top_p": {
            "range": "0.0 to 1.0",
            "meaning": "Nucleus sampling - considers top tokens whose cumulative probability >= top_p",
            "low (0.1-0.3)": "Very focused, conservative. Only considers most likely tokens",
            "medium (0.5-0.7)": "Balanced diversity",
            "high (0.8-1.0)": "Considers many possibilities, more diverse outputs",
        },
        "top_k": {
            "range": "1 to vocab_size",
            "meaning": "Only sample from top K most likely tokens",
            "note": "Less commonly used with modern models (temperature/top_p are preferred)",
        },
        "max_tokens": {
            "range": "1 to model limit (usually 4096+)",
            "meaning": "Maximum tokens in the response",
            "tip": "Set based on expected response length. Too low = truncated, too high = wastes tokens",
        },
    }
    
    for param, info in explanations.items():
        print(f"\n{param.upper()}")
        print("-" * 40)
        for key, value in info.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    print("=" * 60)
    print("LLM Parameter Experiment (using Ollama)")
    print("=" * 60)
    print("\nNote: Make sure Ollama is running (ollama serve)")
    print("      and you have a model pulled (e.g., ollama pull llama3.2)\n")
    
    # Show parameter explanations
    explain_parameters()
    
    print("\n" + "=" * 60)
    input("\nPress Enter to run the experiment...")
    
    # Run the experiment
    try:
        results = run_experiment()
    except requests.exceptions.ConnectionError:
        print("\n" + "=" * 60)
        print("ERROR: Cannot connect to Ollama!")
        print("Make sure Ollama is running:")
        print("  1. Start Ollama: ollama serve")
        print("  2. Pull a model: ollama pull llama3.2")
        print("=" * 60)
    except Exception as e:
        print(f"\nError: {e}")

