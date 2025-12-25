# LLM Dictionary

This dictionary grows as you learn. Terms are added when you ask about them.

## Token

A **small unit of text** that the model reads or writes.

- Depending on the tokenizer, a token can be:
  - A whole word (`hello`)
  - A sub‑word piece (`trans`, `former`)
  - Punctuation or special symbols

- When we feed data to an LLM:
  1. Text → tokenizer → **sequence of token ids** (integers)
  2. Token ids → **embeddings** (vectors) → transformer layers.

- Example:
  - Text: `"Hello world!"`
  - Tokens (simplified): `["Hello", "world", "!"]`
  - Token ids (example): `[101, 203, 999]`  
    (the ids themselves don't matter, only that the model uses them consistently).

---

## Prompt

The **input text** you send to an LLM to get a response.

- Can be a simple question: `"What is machine learning?"`
- Or structured instructions with examples, context, and format requirements.
- **Prompt engineering** = designing effective prompts to get better results.

**Example:**
```
Classify this text as positive or negative.

Text: "I love this product!"
Sentiment:
```

---

## Zero-shot

Asking the LLM to do a task **without providing examples** beforehand.

- The model uses its pre-trained knowledge to understand what you want.
- Good for simple, common tasks.

**Example:**
```
Classify this text as positive, negative, or neutral.

Text: "This is great!"
```

---

## Few-shot

Providing **examples** in your prompt before asking the model to do the same task.

- Shows the model the pattern you want.
- Good for custom categories, specific formats, or complex patterns.

**Example:**
```
Classify customer feedback.

Examples:
Feedback: "The app crashes" → Category: Bug
Feedback: "Can you add dark mode?" → Category: Feature Request

Feedback: "How do I export data?" → Category:
```

---

## Chain-of-Thought (CoT)

Asking the model to **show its reasoning step by step** instead of jumping directly to the answer.

- Better for math problems, logic puzzles, complex reasoning.
- Helps you understand HOW the model got the answer.

**Example:**
```
Solve step by step: If a train travels 60 mph for 2 hours, how far does it go?

Step 1: Distance = Speed × Time
Step 2: Distance = 60 mph × 2 hours
Step 3: Distance = 120 miles
```

---

## Temperature

A parameter that controls **randomness** in the model's output.

- Range: `0.0` to `2.0`
- `0.0` = deterministic (same input → same output, very consistent)
- `1.0+` = very random/creative (same input → different outputs)

**When to use:**
- Low (0.0-0.3): Classification, extraction, code generation (need consistency)
- Medium (0.5-0.7): General chat, writing (balanced)
- High (0.8-2.0): Creative writing, brainstorming (want variety)

**Analogy:** Like adjusting how "creative" vs "precise" you want the model to be.

---

## Top_p (Nucleus Sampling)

Alternative to temperature for controlling randomness.

- Range: `0.0` to `1.0`
- Considers tokens whose cumulative probability adds up to `top_p`
- Lower = more focused/conservative, Higher = more diverse

**Example:**
- `top_p=0.1` → only considers the top 10% most likely tokens (very conservative)
- `top_p=0.9` → considers top 90% of tokens (more creative)

**Note:** Use either `temperature` OR `top_p`, not both (they control similar things).

---

## Max_tokens

The **maximum number of tokens** the model will generate in its response.

- Prevents responses from going on too long.
- Too low → response gets cut off mid-sentence.
- Too high → wastes tokens (and money on paid APIs).

**Note:** 1 token ≈ 4 characters for English text, but it varies.

---

*More terms will be added here as you progress through sections 03–07.*

