# Code Explanation: Section 02 - LLM APIs & Prompt Patterns

This file explains the code examples in section 02, written for someone with data engineering background but new to LLM APIs.

---

## File 1: `basic_api_calls.py`

### Purpose
Shows how to call LLM APIs: both hosted (OpenAI) and local (Ollama).

### Key Concepts

#### 1. **API Keys & Environment Variables**

```python
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

**Why?**
- API keys are secrets (like database passwords).
- Never hardcode them in code.
- `.env` file stores them locally (and `.gitignore` prevents committing them).

**How to set up:**
1. Create a `.env` file in your project root:
   ```
   OPENAI_API_KEY=sk-your-key-here
   ```
2. Install: `pip install python-dotenv`

#### 2. **Messages Format (OpenAI)**

```python
messages = [
    {"role": "user", "content": "What is a transformer?"}
]
```

**Why this format?**
- OpenAI uses a **conversation format** (chat history).
- Each message has a `role`: `"user"`, `"assistant"`, or `"system"`.
- `"system"` sets behavior (optional): `{"role": "system", "content": "You are a helpful assistant."}`

**What about Ollama?**
- Ollama uses **plain text prompts** (simpler but less flexible for multi-turn conversations).

#### 3. **Temperature Parameter**

```python
temperature=0.3
```

**What it does:**
- Controls **randomness** in token selection.
- `0.0` = deterministic (same prompt → same output).
- `1.0+` = very random/creative.

**When to use:**
- Low (`0.0-0.3`): Classification, extraction, code generation (you want consistency).
- Medium (`0.5-0.7`): General chat, writing.
- High (`0.8-2.0`): Creative writing, brainstorming.

---

## File 2: `prompt_patterns.py`

### Purpose
Demonstrates different **prompt engineering techniques** - ways to structure prompts for better results.

### Pattern 1: Zero-shot

```python
prompt = f"""Classify the sentiment of this text as positive, negative, or neutral.

Text: {text}
Sentiment:"""
```

**What is zero-shot?**
- You give the task **without examples**.
- The model uses its pre-trained knowledge to understand.

**When to use:**
- Simple, common tasks (sentiment, language detection).
- Model already understands the pattern.

**Analogy:** Like asking a human "Is this positive or negative?" without showing examples first.

---

### Pattern 2: Few-shot

```python
prompt = f"""Classify customer feedback into these categories:
- Feature Request: asking for new functionality
- Bug Report: describing a problem

Examples:
Feedback: "The app crashes when I click save"
Category: Bug Report

Feedback: "Can you add dark mode?"
Category: Feature Request

Feedback: {text}
Category:"""
```

**What is few-shot?**
- Provide **examples** before the actual task.
- Shows the model the pattern you want.

**When to use:**
- Custom categories or formats.
- Complex patterns the model might not know.
- You want specific formatting.

**Analogy:** Like showing someone 2-3 examples of what you want, then asking them to do the same for a new item.

---

### Pattern 3: Chain-of-Thought (CoT)

```python
prompt = f"""Solve this problem step by step.

Question: {question}

Let's think through this step by step:"""
```

**What is chain-of-thought?**
- Ask the model to **show its reasoning** step by step.
- Instead of jumping to the answer, it explains the process.

**When to use:**
- Math problems.
- Logic puzzles.
- Complex reasoning tasks.
- When you want to understand HOW the model got the answer.

**Why it works:**
- Models are better at following a process than jumping directly to complex answers.
- Breaking down problems helps (just like for humans).

**Example:**
- Bad: "What is 23 * 47?" → might get wrong answer
- Good: "Solve 23 * 47 step by step." → model shows: "20*47 = 940, 3*47 = 141, 940+141 = 1081"

---

### Pattern 4: Role-Task-Context-Style

```python
prompt = f"""You are an expert data engineer with 10 years of experience.

Task: {user_query}

Context:
- The system uses Python 3.10+
- We prefer pandas for data manipulation

Style: 
- Be concise but thorough
- Include code examples when relevant

Response:"""
```

**What is this?**
- **Structured prompt** with clear sections:
  - **Role**: Who the AI should act as.
  - **Task**: What to do.
  - **Context**: Background information.
  - **Style**: How to respond.

**Why use it?**
- Makes prompts **easier to modify** (change one section without rewriting).
- Ensures model considers all aspects (role, context, style).

**When to use:**
- Complex requests where you need specific expertise or format.
- You want consistency across different queries.

---

### Pattern 5: JSON / Structured Output

```python
response_format={"type": "json_object"}
```

**What is this?**
- Forces the model to return **valid JSON**.
- No more parsing errors from text responses.

**Why it's useful:**
- Data engineers need **structured data**.
- Easy to load into pandas, databases, etc.
- No manual parsing of text responses.

**Requirements:**
1. Set `response_format={"type": "json_object"}`.
2. In your prompt, **explicitly ask for JSON** and describe the structure.

**Example:**
```
Return a JSON object with these fields:
- "person_name": name mentioned (or null)
- "location": location mentioned (or null)

JSON:
```

The model will return something like:
```json
{"person_name": "John", "location": "Paris", ...}
```

---

## File 3: `parameter_experiment.py`

### Purpose
Experiment with different parameter values to understand how they affect outputs.

### Why Experiment?

Different tasks need different parameters:
- **Classification** → low temperature (0.0-0.3) for consistency.
- **Creative writing** → high temperature (0.7-1.0) for variety.
- **Extraction** → low temperature for accuracy.

You need to **test** to find what works for your specific use case.

### Key Parameters Explained

#### Temperature vs Top_p

Both control randomness, but differently:

- **Temperature**: Directly scales the probabilities before sampling.
  - Simple to understand.
  - Good default choice.

- **Top_p (Nucleus sampling)**: Considers tokens whose cumulative probability adds up to `top_p`.
  - More sophisticated.
  - Can be better for some tasks.

**Recommendation:** Start with `temperature`. Only use `top_p` if you need fine-grained control.

#### Max_tokens

```python
max_tokens=100
```

**What it does:**
- Limits response length.

**How to choose:**
- Too low → response gets cut off.
- Too high → wastes tokens (and money if using paid API).
- Start with a reasonable guess (100-200 for short answers, 500+ for long-form).

**Note:** Tokens ≠ characters. Roughly: 1 token ≈ 4 characters for English text.

---

## Common Questions

### Q: Should I use OpenAI or Ollama?

**OpenAI (hosted):**
- ✅ Easy to start (just API key).
- ✅ Powerful models (GPT-4o, GPT-4o-mini).
- ❌ Costs money (though GPT-4o-mini is cheap).
- ❌ Data goes to external service (privacy concerns).

**Ollama (local):**
- ✅ Free.
- ✅ Runs on your machine (privacy).
- ✅ No API limits.
- ❌ Requires GPU for good performance (or runs slow on CPU).
- ❌ Models are smaller/less powerful than GPT-4.

**Recommendation:** Start with OpenAI for learning (easy setup). Use Ollama when you need privacy or want to experiment locally.

### Q: What if the API call fails?

Common errors:
- **401 Unauthorized**: Wrong API key.
- **429 Rate Limited**: Too many requests. Add delays between calls.
- **500 Server Error**: OpenAI's servers are down. Retry later.

Always use `try/except`:
```python
try:
    result = call_openai(messages)
except Exception as e:
    print(f"Error: {e}")
```

### Q: How do I choose the right model?

For OpenAI:
- **GPT-4o-mini**: Cheap, fast, good for most tasks.
- **GPT-4o**: More powerful, more expensive, use for complex reasoning.

For Ollama:
- **llama3.2**: Good general-purpose model.
- **mistral**: Fast, efficient.
- **qwen2.5**: Good for coding tasks.

**Start with the cheapest/fastest, upgrade if needed.**

---

## Next Steps

After running these examples:

1. **Try modifying prompts** - see how small changes affect outputs.
2. **Test different tasks** - classification, extraction, generation.
3. **Tune parameters** - find the best temperature for your use case.
4. **Compare models** - try the same prompt on different models.

Then move to section 03 (RAG) where you'll combine LLM calls with retrieval!

