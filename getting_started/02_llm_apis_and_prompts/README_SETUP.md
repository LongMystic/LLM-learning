# Setup Instructions for Section 02

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r ../requirements.txt
   ```

2. **Set up API keys:**
   - Copy `.env.example` to `.env`
   - Get an OpenAI API key from https://platform.openai.com/api-keys
   - Add it to `.env`:
     ```
     OPENAI_API_KEY=sk-your-key-here
     ```

3. **Optional: Set up Ollama (for local models):**
   - Download from https://ollama.ai
   - Install and run: `ollama serve`
   - Pull a model: `ollama pull llama3.2`

4. **Run the examples:**
   ```bash
   # Basic API calls
   python basic_api_calls.py
   
   # Different prompt patterns
   python prompt_patterns.py
   
   # Parameter experiment
   python parameter_experiment.py
   ```

## File Structure

- `basic_api_calls.py` - How to call OpenAI and Ollama APIs
- `prompt_patterns.py` - Different prompt engineering techniques
- `parameter_experiment.py` - Experiment with temperature/top_p
- `explain_about_code.md` - Detailed explanations for beginners
- `.env.example` - Template for API keys

## Troubleshooting

**"API key not found" error:**
- Make sure `.env` file exists in this folder
- Check that `OPENAI_API_KEY=sk-...` is in `.env`
- Restart your terminal/IDE after creating `.env`

**Ollama connection error:**
- Make sure Ollama is running: `ollama serve`
- Check if it's running on port 11434: http://localhost:11434
- Try pulling the model again: `ollama pull llama3.2`

**Rate limit errors:**
- OpenAI has rate limits based on your plan
- Add delays between API calls if testing many prompts
- Consider using Ollama for local experimentation

