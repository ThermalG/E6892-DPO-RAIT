from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch
import os
from transformers import logging

logging.set_verbosity_error()

MODEL_PATH = "/insomnia001/depts/edu/users/qc2354/models/llama3.2-3B"
DATA_PATH = "/insomnia001/home/qc2354/RLfiles/Outputs/Clean_Train/total_train_short_ready.json"
OUTPUT_DIR = "/insomnia001/depts/edu/users/qc2354/outputs/llama3.2-3B-SFT-4datasets"


llama_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
print("DEBUG — Tokenizer type:", type(llama_tokenizer))

llama_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
llama_model.gradient_checkpointing_enable()

def tokenize(example_batch):
    combined_inputs = [
        f"{prompt.strip()} {response.strip()}"
        for prompt, response in zip(example_batch["prompt"], example_batch["label"])
    ]
    tokenized = llama_tokenizer(
        combined_inputs,
        padding=True,          
        truncation=True,      
        max_length=2048      
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

ds = load_dataset("json", data_files=DATA_PATH, split="train")
ds = ds.map(tokenize, batched=True)
ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

# Check checkpoint
resume_checkpoint = None
if os.path.exists(OUTPUT_DIR):
    checkpoints = [ckpt for ckpt in os.listdir(OUTPUT_DIR) if ckpt.startswith("checkpoint")]
    if checkpoints:
        resume_checkpoint = os.path.join(OUTPUT_DIR, sorted(checkpoints)[-1])
        print(f"Resuming from checkpoint: {resume_checkpoint}")


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    deepspeed="./ds_config.json",
    save_strategy="steps",
    save_steps=300,  
    save_total_limit=3,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none"
)

trainer = Trainer(
    model=llama_model,
    args=training_args,
    train_dataset=ds,
    tokenizer=llama_tokenizer,
    data_collator=DataCollatorForLanguageModeling(llama_tokenizer, mlm=False)
)

trainer.train(resume_from_checkpoint=resume_checkpoint if resume_checkpoint else None)

llama_tokenizer.save_pretrained("/insomnia001/depts/edu/users/qc2354/outputs/llama3.2-3B-SFT-4data-final")
trainer.model.save_pretrained("/insomnia001/depts/edu/users/qc2354/outputs/llama3.2-3B-SFT-4data-final")

