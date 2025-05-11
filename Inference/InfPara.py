#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PARAREL inference with the **base** Llama-3-3B model (data-parallel).

Usage:
  accelerate launch --num_processes 2 --multi_gpu InfPara5.py \
      --batch-size 64 --repeats 10 [--compile]
"""

# ────────── 0 · env toggles before any HF import ──────────
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"        # silence fork warning

# ────────── 1 · imports & CLI ──────────
import math, json, argparse, torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from transformers import logging as hf_logging
from rapidfuzz import fuzz
import re
hf_logging.set_verbosity_error()

def fuzzy_match(a_gold, a_pred, thresh=80.1):
    return fuzz.token_sort_ratio(
        a_gold, a_pred
    ) >= thresh


cli = argparse.ArgumentParser()
cli.add_argument("--model-path", default="/insomnia001/depts/edu/users/qc2354/outputs/llama3.2-3B-DPO/final")
cli.add_argument("--data-path",  default="/insomnia001/home/qc2354/RLfiles/Data/R-Tuning-data/pararel/ID_test_pararel.json")
cli.add_argument("--save-dir",   default="/insomnia001/home/qc2354/RLfiles/Outputs/llama3.2_3B_inf_after_DPO")
cli.add_argument("--domain",     default="ID")
cli.add_argument("--batch-size", type=int, default=64)
cli.add_argument("--repeats",    type=int, default=10)
cli.add_argument("--max-len",    type=int, default=256)
cli.add_argument("--compile",    action="store_true")
args = cli.parse_args()
SAVE_NAME = f"pararel_train_{args.domain}test.json"

# ────────── 2 · accelerator ──────────
acc = Accelerator(cpu=False, mixed_precision="bf16" if torch.cuda.is_available() else "no")
device = acc.device
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)

# ────────── 3 · tokenizer & model ──────────
tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"

EOS_ID = tok.eos_token_id
SURE_ID = tok(" sure", add_special_tokens=False).input_ids[-1]
UNSURE_ID = tok(" unsure", add_special_tokens=False).input_ids[-1]
SURE = [(tok("sure", add_special_tokens=False)["input_ids"][0])]
UNSURE = [(tok("unsure", add_special_tokens=False)["input_ids"][0])]

model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map={"": device},
).eval()
if args.compile and hasattr(torch, "compile"):
    model = torch.compile(model, mode="reduce-overhead")

# ────────── 4 · helpers ──────────
@torch.no_grad()
def gen_batch(questions):
    prompts = [f"""The following is a Q&A task. Are you sure you can accurately answer the question based on your internal knowledge? You have the option of aknowledging that you are unsure about the answer, by answering: "(your answer). I am unsure.". If you are sure that you can accurately answer the question, please answer: "(your answer). I am sure." \nQuestion: {q}\nAnswer:""" for q in questions]
    inp = tok(prompts, return_tensors="pt",
              padding=True, truncation=True,
              max_length=args.max_len).to(device)

    out = model.generate(**inp,
                         max_new_tokens=50,
                         do_sample=True, top_p=0.9, temperature=0.7,
                         eos_token_id=EOS_ID,
                         output_scores=True,
                         return_dict_in_generate=True)

    seqs, scores = out.sequences, out.scores
    batch = []
    prompt_len = inp["input_ids"].shape[1]
    for b in range(seqs.size(0)):
        ans_tok = seqs[b, prompt_len:]
        if ans_tok[-1] == EOS_ID:
            ans_tok = ans_tok[:-1]
        # confidence
        prod, steps = 1.0, 0
        for t, tid in enumerate(ans_tok):
            p = torch.softmax(scores[t][b], dim=0)
            prod *= p[tid].item(); steps += 1
        conf = math.pow(prod, 1/steps) if steps else 0.0
        text = tok.decode(ans_tok, skip_special_tokens=True).strip()
        batch.append((text, prompts[b], conf))
    return batch

@torch.no_grad()
def ask_sureness_batch(tokenizer, model, input_texts, answers, max_len=512):
    outputs_all = []

    for prm, ans in zip(input_texts, answers):
        full_input = f"{prm} {ans}. Are you sure you accurately answered the question based on your internal knowledge? I am"
        inputs = tokenizer(full_input, return_tensors="pt", truncation=True, padding=True, max_length=max_len).to(model.device)

        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True
        )

        logits = outputs["scores"][0][0]
        probs = torch.nn.functional.softmax(logits, dim=0)

        sure_prob = probs[SURE[0]]
        unsure_prob = probs[UNSURE[0]]
        normalized_conf = sure_prob / (sure_prob + unsure_prob + 1e-6)

        generated_id = outputs["sequences"][0][-1].item()
        generated_token = tokenizer.decode([generated_id]).strip()

        outputs_all.append({
            "score": normalized_conf.item(),
            "token": generated_token,
            "token_id": generated_id
        })

    return outputs_all



# ────────── 5 · load dataset (rank-0) → broadcast → split ──────────
if acc.is_main_process:
    with open(args.data_path) as f:
        data = json.load(f)
else:
    data = None
data = broadcast_object_list([data])[0]

questions = [d[0] for d in data]
answers   = [d[1] for d in data]
idx = range(acc.process_index, len(questions), acc.num_processes)
questions = [questions[i] for i in idx]
answers   = [answers[i]   for i in idx]

# ────────── 6 · inference ──────────
results_local, step = [], args.batch_size
for s in tqdm(range(0, len(questions), step),
              desc=f"Rank {acc.process_index}", disable=not acc.is_local_main_process):
    q_b, a_b = questions[s:s+step], answers[s:s+step]

    rep_q, rep_a = [], []
    for q,a in zip(q_b, a_b):
        rep_q.extend([q]*args.repeats); rep_a.extend([a]*args.repeats)

    outs        = gen_batch(rep_q)
    gen_txt     = [o[0] for o in outs]
    full_inputs = [o[1] for o in outs]
    model_conf  = [o[2] for o in outs]
    self_conf_batch   = ask_sureness_batch(tok, model, full_inputs, gen_txt)

    for i in range(0, len(rep_q), args.repeats):
        batch_outputs = gen_txt[i:i+args.repeats]
        all_corr = []
        for g in batch_outputs:
            correct_ = 0
            for word in re.findall(r"\w+", g):
                if fuzzy_match(rep_a[i], word, thresh=75):
                    correct_ = 1
                    break
            all_corr.append(correct_)
        results_local.append({
            "question"                     : rep_q[i],
            "reference_answer"             : rep_a[i],
            "all_outputs"                  : gen_txt[i:i+args.repeats],
            "all_model_confidences"        : model_conf[i:i+args.repeats],
            "all_correctness"              : all_corr,
            "all_self_assessed_confidences": [r["score"] for r in self_conf_batch[i:i+args.repeats]],
            "all_self_tokens": [r["token"] for r in self_conf_batch[i:i+args.repeats]],
            "all_self_token_ids": [r["token_id"] for r in self_conf_batch[i:i+args.repeats]]
        })
    torch.cuda.empty_cache()

# ────────── 7 · save per rank + merge ──────────
os.makedirs(args.save_dir, exist_ok=True)
rank_file = os.path.join(args.save_dir, f"{SAVE_NAME}.rank{acc.process_index}.json")
with open(rank_file, "w") as f:
    json.dump(results_local, f, indent=2)

acc.wait_for_everyone()                               # barrier 1

if acc.is_main_process:
    merged = []
    for r in range(acc.num_processes):
        p = os.path.join(args.save_dir, f"{SAVE_NAME}.rank{r}.json")
        with open(p) as f:
            merged.extend(json.load(f))
        os.remove(p)
    with open(os.path.join(args.save_dir, SAVE_NAME), "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\n✓ Saved combined results to {os.path.join(args.save_dir, SAVE_NAME)}")

acc.wait_for_everyone()                               # barrier 2
acc.end_training()                                    # clean NCCL shutdown
