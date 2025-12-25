## 4. Fine-tuning & Adapters (3–4 weeks)

### Goal
Fine-tune a small open model using LoRA/QLoRA and compare it to the base model.

### Learn
- LoRA/QLoRA concepts (adapters, low-rank updates, quantization)
- Instruction-tuning data formats
- Overfitting signs and metrics

### Do
- Choose a 7B instruct model (e.g., Mistral 7B Instruct).
- Prepare data:
  - Collect and clean examples
  - Deduplicate
  - Convert into instruction format
  - Split into train/validation
- Configure QLoRA:
  - Use `bitsandbytes` 4-bit quantization
  - Set LoRA target modules (e.g., q_proj, k_proj, v_proj, o_proj)
- Train:
  - Monitor training and validation loss
  - Save checkpoints regularly
- Compare:
  - Evaluate base vs fine-tuned model on your RAG or task dataset

### Done when
- Fine-tuned model shows measurable improvement over the base model on your evals.
- You understand how LoRA adapters plug into the base model and how to load/unload them.

