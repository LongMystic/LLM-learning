"""
05.2 – Train LoRA trên base model (TinyLlama).
Dùng dataset từ prepare_data.py; lưu adapter vào output_dir.
"""
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType

from prepare_data import prepare_dataset, DEFAULT_MODEL_NAME, MAX_LENGTH

# Cấu hình
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "experiments" / "lora_run"
LORA_R = 8
LORA_ALPHA = 16
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 2
LR = 2e-5
SAVE_STEPS = 50
MAX_STEPS = -1  # -1 = train theo epoch; trên CPU nên đặt 5–10 để test nhanh

def main():
    print("Load dataset...")
    dataset = prepare_dataset()
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_ds = dataset["train"]
    eval_ds = dataset["test"]

    use_cuda = torch.cuda.is_available()
    print("Load tokenizer & model...")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL_NAME,
        torch_dtype=torch.float16 if use_cuda else torch.float32,
        device_map="auto" if use_cuda else "cpu",
        trust_remote_code=True,
    )

    print("Gắn LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    use_fp16 = use_cuda
    max_steps = MAX_STEPS
    batch_size = BATCH_SIZE
    if not use_fp16:
        if max_steps == -1:
            max_steps = 2
            print("CPU: 2 bước test. Đổi MAX_STEPS nếu muốn train thêm.")
        else:
            print("Không có GPU; train trên CPU (rất chậm).")
        batch_size = 1

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        fp16=use_fp16,
        logging_steps=1,
        save_steps=SAVE_STEPS,
        max_steps=max_steps,
        eval_strategy="no",
        save_total_limit=2,
        load_best_model_at_end=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )
    print("Bắt đầu train...")
    trainer.train()
    trainer.save_model(str(OUTPUT_DIR / "final_adapter"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final_adapter"))
    print(f"Đã lưu adapter tại {OUTPUT_DIR / 'final_adapter'}")

if __name__ == "__main__":
    main()