"""
Different prompt engineering patterns
Each function demonstrates a different technique
"""

from openai import OpenAI
import os
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================
# Pattern 1: Zero-shot (no examples)
# ============================================

def zero_shot_classification(text):
    """
    Zero-shot: Just give the task, no examples
    Good for: Simple tasks, model already understands the pattern
    """
    prompt = f"""Classify the sentiment of this text as positive, negative, or neutral.

Text: {text}
Sentiment:"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content


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
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content


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
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return response.choices[0].message.content


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
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return response.choices[0].message.content


# ============================================
# Pattern 5: JSON / Structured output
# ============================================

def structured_json_extraction(text):
    """
    Extract structured data as JSON
    Use response_format={"type": "json_object"} to force JSON output
    """
    prompt = f"""Extract information from this text and return it as JSON.

Text: {text}

Return a JSON object with these fields:
- "person_name": name mentioned (or null)
- "location": location mentioned (or null)
- "date": date mentioned (or null)
- "sentiment": overall sentiment (positive/negative/neutral)

JSON:"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        response_format={"type": "json_object"},  # Forces JSON output
    )
    
    result = response.choices[0].message.content
    # Parse JSON to validate it's valid
    return json.loads(result)


# ============================================
# Example usage
# ============================================

if __name__ == "__main__":
    print("=== Zero-shot Classification ===")
    result = zero_shot_classification("I love this product!")
    print(result)
    
    print("\n=== Few-shot Classification ===")
    result = few_shot_classification("The dashboard is too slow")
    print(result)
    
    print("\n=== Chain-of-Thought ===")
    result = chain_of_thought_reasoning("If a train travels 60 mph for 2 hours, how far does it go?")
    print(result)
    
    print("\n=== Structured Prompt ===")
    result = structured_prompt("How should I handle missing data in a pandas DataFrame?")
    print(result)
    
    print("\n=== JSON Extraction ===")
    result = structured_json_extraction("John visited Paris on March 15th. He had a wonderful time!")
    print(json.dumps(result, indent=2))

