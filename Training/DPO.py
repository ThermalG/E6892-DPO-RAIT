from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer, DPOConfig
from transformers import logging
import os
import torch
import numpy as np
import numpy.core.multiarray




logging.set_verbosity_error()

MODEL_PATH = "/insomnia001/depts/edu/users/qc2354/models/llama3.2-3B"
DATA_PATH = "/insomnia001/home/qc2354/RLfiles/Outputs/Clean_DPO/total_DPO_ready.json"
OUTPUT_DIR = "/insomnia001/depts/edu/users/qc2354/outputs/llama3.2-3B-DPO"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
print("DEBUG — Tokenizer type:", type(tokenizer))

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
model.gradient_checkpointing_enable()
ref_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
ref_model.gradient_checkpointing_enable()

# Load and format dataset
ds = load_dataset("json", data_files=DATA_PATH, split="train")

def prepare_dpo_format(example_batch):
    return {
        "prompt": example_batch["prompt"],
        "chosen": example_batch["selected"],
        "rejected": example_batch["rejected"]
    }

ds = ds.map(prepare_dpo_format, batched=True)

# Optional resume from checkpoint
resume_checkpoint = None
if os.path.exists(OUTPUT_DIR):
    checkpoints = [ckpt for ckpt in os.listdir(OUTPUT_DIR) if ckpt.startswith("checkpoint")]
    if checkpoints:
        resume_checkpoint = os.path.join(OUTPUT_DIR, sorted(checkpoints)[-1])
        print(f"Resuming from checkpoint: {resume_checkpoint}")

# DeepSpeed-compatible training args
dpo_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=3,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    deepspeed="ds_config.json",
    save_strategy="steps",
    save_steps=300,
    save_total_limit=3,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",
    beta=0.1,
    truncation_mode="keep_end",
    max_length=2048,
    label_pad_token_id=-100
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_args,      
    train_dataset=ds,
    processing_class = tokenizer
)

# Start training
trainer.train(resume_from_checkpoint=resume_checkpoint)

# Save model and tokenizer
model.save_pretrained(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")