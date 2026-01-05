"""
Different prompt engineering patterns
Each function demonstrates a different technique
"""

from openai import OpenAI
import json
from basic_api_calls import call_ollama
import requests


# ============================================
# Pattern 1: Zero-shot (no examples)
# ============================================

def call_ollama(prompt, model="llama3.2", temperature=0.7, max_tokens=256):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "num_predict": max_tokens,
        "stream": False,
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["response"]


def zero_shot_classification(text):
    """
    Zero-shot: Just give the task, no examples
    Good for: Simple tasks, model already understands the pattern
    """
    prompt = f"""Classify the sentiment of this text as positive, negative, or neutral.
        Text: {text}
        Sentiment:
    """
    
    response = call_ollama(
        prompt=prompt,
        temperature=0.3
    )
    return response


# ============================================
# Pattern 2: Few-shot (with examples)
# ============================================

def few_shot_classification(text):
    """
    Few-shot: Provide examples before the actual task
    Good for: Complex patterns, custom categories, formatting
    """
    prompt = f"""Classify customer feedback into these categories:
- Feature Request: asking for new functionality
- Bug Report: describing a problem
- Question: asking for information
- Compliment: positive feedback

Examples:
Feedback: "The app crashes when I click save"
Category: Bug Report

Feedback: "Can you add dark mode?"
Category: Feature Request

Feedback: "How do I export my data?"
Category: Question

Feedback: {text}
Category:"""
    
    response = call_ollama(
        prompt=prompt,
        temperature=0.3
    )
    return response


# ============================================
# Pattern 3: Chain-of-Thought (reasoning step by step)
# ============================================

def chain_of_thought_reasoning(question):
    """
    Chain-of-Thought: Ask model to think step by step
    Good for: Math problems, logic puzzles, complex reasoning
    """
    prompt = f"""Solve this problem step by step.

Question: {question}

Let's think through this step by step:"""
    
    response = call_ollama(
        prompt=prompt,
        temperature=0.5
    )
    return response


# ============================================
# Pattern 4: Role-Task-Context-Style structure
# ============================================

def structured_prompt(user_query):
    """
    Structured prompt with clear sections:
    - Role: who the AI is
    - Task: what to do
    - Context: background information
    - Style: how to respond
    """
    prompt = f"""You are an expert data engineer with 10 years of experience.

Task: {user_query}

Context:
- The system uses Python 3.10+
- We prefer pandas for data manipulation
- Our team is junior-level, so explanations should be clear

Style: 
- Be concise but thorough
- Include code examples when relevant
- Explain "why" not just "what"

Response:"""
    
    response = call_ollama(
        prompt=prompt,
        temperature=0.4
    )
    return response


# ============================================
# Pattern 5: JSON / Structured output
# ============================================

def structured_json_extraction(text):
    """
    Extract structured data as JSON
    """
    prompt = f"""Extract information from this text and return it as JSON.

Text: {text}

Return a JSON object with these fields:
- "person_name": name mentioned (or null)
- "location": location mentioned (or null)
- "date": date mentioned (or null)
- "sentiment": overall sentiment (positive/negative/neutral)

Return ONLY the JSON object.
"""
    
    response = call_ollama(
        prompt=prompt,
        temperature=0.2,
    )
    
    return response


# ============================================
# Example usage
# ============================================

if __name__ == "__main__":
    print("=== Zero-shot Classification ===")
    zero_shot_prompt = "I love this product!"
    print(f"Zero-shot prompt: {zero_shot_prompt}")
    result = zero_shot_classification(zero_shot_prompt)
    print(result)
    
    print("\n=== Few-shot Classification ===")
    few_shot_prompt = "The dashboard is too slow"
    print(f"Few-shot prompt: {few_shot_prompt}")
    result = few_shot_classification(few_shot_prompt)
    print(result)
    
    print("\n=== Chain-of-Thought ===")
    chain_of_thought_prompt = "If a train travels 60 mph for 2 hours, how far does it go?"
    print(f"Chain-of-thought prompt: {chain_of_thought_prompt}")
    result = chain_of_thought_reasoning(chain_of_thought_prompt)
    print(result)
    
    print("\n=== Structured Prompt ===")
    structured_prompt_str = "How should I handle missing data in a pandas DataFrame?"
    print(f"Structured prompt: {structured_prompt_str}")
    result = structured_prompt(structured_prompt_str)
    print(result)
    
    print("\n=== JSON Extraction ===")
    json_extraction_prompt = "John visited Paris on March 15th. He had a wonderful time!"
    print(f"JSON extraction prompt: {json_extraction_prompt}")
    result = structured_json_extraction(json_extraction_prompt)
    print(json.dumps(result, indent=2))

