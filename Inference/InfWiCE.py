#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""InfWiCE.py — Batched WiCE inference with certainty using Accelerate"""

import os
import json
import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
import argparse

# ────────── CLI ──────────
parser = argparse.ArgumentParser(description="InfMMLU_2.py: mc mode inference")
parser.add_argument("--model-path", type=str, default="/insomnia001/depts/edu/users/qc2354/models/llama3.2-3B")
parser.add_argument("--data-path",type=str, default="/insomnia001/home/qc2354/RLfiles/Data/R-Tuning-data/WiCE/wice_train.json")
parser.add_argument("--save-dir", type=str, default="/insomnia001/home/qc2354/RLfiles/Outputs/llama3.2_3B_inf_before_MMLU_train")
parser.add_argument("--batch-size", type=int, default=40)
parser.add_argument("--max-len", type=int, default=1024)
args = parser.parse_args()

# ────────── Accelerator / Device ──────────
acc = Accelerator()
device = acc.device

# ────────── Tokenizer & Model ──────────
tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=torch.float32,
    device_map={"": device}
).eval()
model.config.pad_token_id = tok.pad_token_id

STOP = tok(".")['input_ids'][0]
SURE = tok("sure")['input_ids'][0]
UNSURE = tok("unsure")['input_ids'][0]

choices = ["A", "B", "C"]
candidate_answer = ['supported.', 'partially_supported.', 'not_supported.']
label_map = {'supported': "A", 'partially_supported': "B", 'not_supported': "C"}

# ────────── Formatting ──────────
def format_question(ex):
    evidence = " ".join(ex["evidence"])
    claim = ex["claim"]
    block = f"Evidence: {evidence}\nClaim: {claim}\nQuestion: Does the evidence support the claim?"
    for i, ch in enumerate(choices):
        block += f"\n{ch}: {candidate_answer[i]}"
    block += "\nAnswer:"
    return block

# ────────── Batch Prediction ──────────
@torch.no_grad()
def batch_predict(batch_data):
    prompts = [format_question(ex) for ex in batch_data]
    tok_in = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                 max_length=args.max_len).to(device)

    outputs = model(**tok_in)
    logits = outputs.logits  # (batch, seq_len, vocab_size)
    last_tok_idx = (tok_in['attention_mask'].sum(dim=1) - 1).tolist()

    results = []
    for b, ex in enumerate(batch_data):
        logits_b = logits[b, last_tok_idx[b]]  # final token logits for this example

        # Map letter choices robustly
        letter_ids = {}
        for ch in choices:
            ids = tok(f" {ch}", add_special_tokens=False).input_ids
            if len(ids) == 1:
                letter_ids[ch] = ids[0]
            else:
                print(f"[ERROR] Choice '{ch}' maps to multiple tokens: {ids}")
        
        if len(letter_ids) != 3:
            print(f"[ERROR] Skipping due to invalid choice token mapping: {letter_ids}")
            continue

        try:
            three_logits = torch.tensor(
                [logits_b[letter_ids[ch]].item() for ch in choices],
                device=device
            )
        except KeyError as e:
            print(f"[FATAL] Missing token for choice: {e}")
            continue

        if torch.isnan(three_logits).any() or torch.isinf(three_logits).any():
            print(f"[ERROR] NaN/Inf in logits: {three_logits}\nPrompt: {prompts[b]}")
            continue

        probs = torch.softmax(three_logits, dim=0).detach().cpu().numpy()

        if np.isnan(probs).any() or np.isinf(probs).any():
            print(f"[ERROR] NaN/Inf in softmax output: {probs}\nPrompt: {prompts[b]}")
            continue

        idx_max = int(np.argmax(probs))
        pred_letter = choices[idx_max]
        pred_label = ["supported", "partially_supported", "not_supported"][idx_max]
        ref_letter = label_map[ex["label"]]

        results.append({
            "claim": ex["claim"],
            "predicted": pred_label,
            "reference_answer": ex["label"],
            "is_correct": int(pred_label == ex["label"]),
            "probs": dict(zip(choices, probs.tolist())),
            "prob_generated_token": float(probs[idx_max]),
            "prob_reference_token": float(probs[choices.index(ref_letter)]),
            "certainty_prompt": f"{format_question(ex)} {pred_letter}"
        })

    return results


# ────────── Certainty Estimation ──────────
@torch.no_grad()
def batch_certainty(results):
    prompts = [f"{r['certainty_prompt']}. Are you sure you accurately answered the question based on your internal knowledge? I am" for r in results]
    inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                 max_length=args.max_len).to(device)

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        pad_token_id=tok.pad_token_id,
        max_new_tokens=1,
        output_scores=True,
        return_dict_in_generate=True
    )

    logits = outputs.scores[0]  # (batch, vocab)
    for i, r in enumerate(results):
        pt = torch.softmax(logits[i], dim=0)
        sure_prob = pt[SURE].item()
        unsure_prob = pt[UNSURE].item()
        norm_certainty = sure_prob / (sure_prob + unsure_prob + 1e-8)
        r["self_assessed_certainty"] = norm_certainty
    return results

# ────────── Load & Distribute Data ──────────
if acc.is_main_process:
    with open(args.data_path) as f: data = json.load(f)
else:
    data = None
data = broadcast_object_list([data])[0]

# ────────── Shard for Distributed ──────────
my_samples = [data[i] for i in range(acc.process_index, len(data), acc.num_processes)]

# ────────── Run ──────────
results = []
for i in tqdm(range(0, len(my_samples), args.batch_size),
              desc=f"Rank {acc.process_index}"):
    batch = my_samples[i: i + args.batch_size]
    batch_results = batch_predict(batch)
    batch_results = batch_certainty(batch_results)
    results.extend(batch_results)
    torch.cuda.empty_cache()

# ────────── Save ──────────
os.makedirs(args.save_dir, exist_ok=True)
rank_file = os.path.join(args.save_dir, f"results.rank{acc.process_index}.json")
with open(rank_file, "w") as f:
    json.dump(results, f, indent=2)

acc.wait_for_everyone()
if acc.is_main_process:
    merged = []
    for r in range(acc.num_processes):
        fpath = os.path.join(args.save_dir, f"results.rank{r}.json")
        with open(fpath) as f:
            merged.extend(json.load(f))
        os.remove(fpath)

    final_path = os.path.join(args.save_dir, "WiCE_inference.json")
    with open(final_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\n✓ Saved merged results to {final_path}")

acc.end_training()
