import os
import json
import argparse
import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from RLfiles.Scripts.UsableScripts.InfPara6 import ask_sureness_batch

parser = argparse.ArgumentParser(description="InfMMLU_2.py: mc mode inference")
parser.add_argument("--mode", choices=["mc"], default="mc")
parser.add_argument("--model-path", type=str, default="/insomnia001/depts/edu/users/qc2354/outputs/llama3.2-3B-DPO/final")
parser.add_argument("--data-path",type=str, default="/insomnia001/home/qc2354/RLfiles/Data/R-Tuning-data/MMLU/MMLU_ID_test.json")
parser.add_argument("--prompt-path", type=str, default="/insomnia001/home/qc2354/RLfiles/Data/R-Tuning-data/MMLU/MMLU_ID_prompt.json")
parser.add_argument("--save-dir", type=str, default="/insomnia001/home/qc2354/RLfiles/Outputs/llama3.2_3B_inf_after_DPO")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--max-len", type=int, default=1024)
args = parser.parse_args()

# accelerator
acc = Accelerator(cpu=False, mixed_precision="no")
device = acc.device

# tokenizer & model
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

choices = ["A", "B", "C", "D"]

# prompt constructor
def format_subject(subject):
    return " ".join(subject.split("_"))

def format_shots(prompt_data):
    block = ""
    for ex in prompt_data:
        q, *opts, ans = ex
        block += q + "\n"
        for i, o in enumerate(opts):
            block += f"{choices[i]}. {o}\n"
        block += f"Answer: {ans}\n\n"
    return block

def format_example(inst):
    q, *opts = inst
    block = q + "\n"
    for i, o in enumerate(opts[:-1]):
        block += f"{choices[i]}. {o}\n"
    block += "Answer:"
    return block

def gen_prompt(subject, inst, prompt_data):
    header = (f"The following are multiple choice questions about "
              f"{format_subject(subject)}.\n\n")
    return header + format_shots(prompt_data) + format_example(inst)
SURE_ID = tok(" sure", add_special_tokens=False).input_ids[-1]
UNSURE_ID = tok(" unsure", add_special_tokens=False).input_ids[-1]
SURE = [(tok("sure", add_special_tokens=False)["input_ids"][0])]
UNSURE = [(tok("unsure", add_special_tokens=False)["input_ids"][0])]
def ask_sureness_batch_batched(tokenizer, model, input_texts, answers, max_len=512):
    prompts = [
        f"{prm} {ans}. Are you sure you accurately answered the question based on your internal knowledge? I am"
        for prm, ans in zip(input_texts, answers)
    ]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_len
    ).to(model.device)

    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=1,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id
    )

    logits_batch = outputs["scores"][0]  # (batch_size, vocab_size)
    sequences = outputs["sequences"]     # (batch_size, seq_len)

    results = []
    for i in range(len(prompts)):
        logits = logits_batch[i]
        probs = torch.nn.functional.softmax(logits, dim=0)

        sure_prob = probs[SURE[0]]
        unsure_prob = probs[UNSURE[0]]
        normalized_conf = sure_prob / (sure_prob + unsure_prob + 1e-6)

        generated_id = sequences[i, -1].item()
        generated_token = tokenizer.decode([generated_id]).strip()

        results.append({
            "score": normalized_conf.item(),
            "token": generated_token,
            "token_id": generated_id
        })

    return results


# batch inference
@torch.no_grad()
def batch_inference(triples):
    prompts = [gen_prompt(s, inst, pd) for s, inst, pd in triples]
    tok_in = tok(prompts,
                 return_tensors="pt",
                 padding=True,
                 truncation=True,
                 max_length=args.max_len).to(device)

    outputs = model(**tok_in)
    logits = outputs.logits  # (batch, seq_len, vocab)
    last_tok_idx = (tok_in['attention_mask'].sum(dim=1) - 1).tolist()

    results = []
    for b, (subject, inst, pd) in enumerate(triples):
        logits_b = logits[b, last_tok_idx[b]]  # (vocab_size,)

        letter_ids = {} # robust token ID mapping
        for ch in choices:
            ids = tok(f" {ch}", add_special_tokens=False).input_ids
            if len(ids) == 1:
                letter_ids[ch] = ids[0]
            else:
                print(f"[ERROR] Choice '{ch}' does not map to single token: {ids}")
        
        if len(letter_ids) != 4:
            print(f"[ERROR] Skipping sample due to incomplete token mapping: {letter_ids}")
            continue

        try:
            four_logits = torch.tensor(
                [logits_b[letter_ids[ch]].item() for ch in choices],
                device=device
            )
        except KeyError as e:
            print(f"[FATAL] Missing token for choice: {e}")
            continue

        if torch.isnan(four_logits).any() or torch.isinf(four_logits).any():
            print(f"[ERROR] NaN/Inf in logits: {four_logits}\nPrompt: {prompts[b]}")
            continue

        probs = torch.softmax(four_logits, dim=0).detach().cpu().numpy()

        if np.isnan(probs).any() or np.isinf(probs).any():
            print(f"[ERROR] NaN/Inf in softmax output: {probs}\nPrompt: {prompts[b]}")
            continue

        idx_max = int(np.argmax(probs))
        pred_letter = choices[idx_max]
        prob_pred = float(probs[idx_max])
        ref_letter = inst[-1]
        ref_idx = choices.index(ref_letter)
        prob_ref = float(probs[ref_idx])

        results.append({
            "subject": subject,
            "question": inst[0],
            "options": inst[1:-1],
            "predicted": pred_letter,
            "reference_answer": ref_letter,
            "is_correct": int(pred_letter == ref_letter),
            "probs": dict(zip(choices, probs.tolist())),
            "prob_generated_token": prob_pred,
            "prob_reference_token": prob_ref
        })
        sureness_inputs = [gen_prompt(s, inst, pd) for s, inst, pd in triples]
        pred_answers = [res["predicted"] for res in results[-len(triples):]]  # only the current batch's results
        sureness_outputs = ask_sureness_batch(tok, model, sureness_inputs, pred_answers, max_len=args.max_len)
        for res, sure in zip(results[-len(triples):], sureness_outputs):
            res.update({
                "sureness_token": sure["token"],
                "sureness_token_id": sure["token_id"],
                "sureness_score": sure["score"]
            })
    return results

# load data
if acc.is_main_process:
    with open(args.data_path) as f: test_map = json.load(f)
    with open(args.prompt_path) as f: prompt_map = json.load(f)
else:
    test_map, prompt_map = None, None

test_map = broadcast_object_list([test_map])[0]
prompt_map = broadcast_object_list([prompt_map])[0]

# flatten & split
all_triples = []
for subject, insts in test_map.items():
    shots = prompt_map.get(subject)
    if shots is None:
        print(f"[WARN] no shots for {subject}")
        continue
    for inst in insts:
        all_triples.append((subject, inst, shots))

my_triples = [
    all_triples[i]
    for i in range(acc.process_index, len(all_triples), acc.num_processes)
]

# inference
results = []
for i in tqdm(range(0, len(my_triples), args.batch_size),
              desc=f"Rank {acc.process_index}",
              disable=not acc.is_local_main_process):
    batch = my_triples[i: i + args.batch_size]
    results.extend(batch_inference(batch))
    torch.cuda.empty_cache()

# save
os.makedirs(args.save_dir, exist_ok=True)
rank_file = os.path.join(args.save_dir,
                         f"results.rank{acc.process_index}.json")
with open(rank_file, "w") as f:
    json.dump(results, f, indent=2)

acc.wait_for_everyone()

if acc.is_main_process:
    merged = []
    for r in range(acc.num_processes):
        path = os.path.join(args.save_dir, f"results.rank{r}.json")
        with open(path) as f:
            merged.extend(json.load(f))
        os.remove(path)

    out_path = os.path.join(args.save_dir, "MMLU_ID_inference_test.json")
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\n✓ Saved merged results to {out_path}")

acc.end_training()