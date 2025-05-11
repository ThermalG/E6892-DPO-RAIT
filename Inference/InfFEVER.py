import os, json, argparse
import torch, numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list

os.environ["TOKENIZERS_PARALLELISM"] = "false" 

parser = argparse.ArgumentParser(description="InfMMLU_2.py: mc mode inference")
parser.add_argument("--model-path", type=str, default="/insomnia001/depts/edu/users/qc2354/models/llama3.2-3B")
parser.add_argument("--data-path",type=str, default="/insomnia001/home/qc2354/RLfiles/Data/R-Tuning-data/FEVER/fever_10k.json")
parser.add_argument("--save-dir", type=str, default="/insomnia001/home/qc2354/RLfiles/Outputs/llama3.2_3B_inf_before_MMLU_train")
parser.add_argument("--batch-size", type=int, default=40)
parser.add_argument("--max-len", type=int, default=1024)
args = parser.parse_args()


acc = Accelerator(cpu=False, mixed_precision="no")
device = acc.device

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


choices = ["A", "B", "C"]
label_to_letter = { 
    "SUPPORTED":        "A",
    "SUPPORTS":         "A",
    "REFUTED":          "B",
    "REFUTES":          "B",
    "NOT ENOUGH INFO":  "C"
}
letter_to_label = {v: k for k, v in label_to_letter.items()}


def gen_prompt(claim: str) -> str:
    return (
        f"Claim: {claim.strip()}\n"
        "A. Supports\n"
        "B. Refutes\n"
        "C. Not enough info\n"
        "Answer:"
    )


@torch.no_grad()
def batch_inference(triples):
    prompts = [gen_prompt(claim) for claim, _ in triples]
    tok_in = tok(prompts,
                 return_tensors="pt",
                 padding=True,
                 truncation=True,
                 max_length=args.max_len).to(device)

    out = model(**tok_in)
    logits = out.logits
    last_idx = (tok_in["attention_mask"].sum(dim=1) - 1).tolist()

    results = []
    for b, (claim, gold_label) in enumerate(triples):
        logits_b = logits[b, last_idx[b]]

        letter_ids = {}
        for ch in choices:
            ids = tok(f" {ch}", add_special_tokens=False).input_ids
            if len(ids) == 1:
                letter_ids[ch] = ids[0]
            else:
                print(f"[ERROR] Choice '{ch}' maps to multi-token {ids}, skipping.")

        if len(letter_ids) != 3:
            print(f"[ERROR] Skipping due to incomplete letter_ids: {letter_ids}")
            continue

        try:
            mc_logits = torch.tensor(
                [logits_b[letter_ids[ch]].item() for ch in choices],
                device=device
            )
        except KeyError as e:
            print(f"[ERROR] Missing token: {e}")
            continue

        if torch.isnan(mc_logits).any() or torch.isinf(mc_logits).any():
            continue

        probs = torch.softmax(mc_logits, dim=0).detach().cpu().numpy()
        if np.isnan(probs).any() or np.isinf(probs).any():
            continue

        idx_max = int(np.argmax(probs))
        pred_letter = choices[idx_max]
        prob_pred = float(probs[idx_max])


        norm_label = gold_label.strip().upper() 
        if "SUPPORT" in norm_label:
            ref_letter = "A"
        elif "REFUTE" in norm_label:
            ref_letter = "B"
        elif "NOT ENOUGH" in norm_label:
            ref_letter = "C"
        else:
            ref_letter = None
        prob_ref = float(probs[choices.index(ref_letter)]) if ref_letter else None

        results.append({
            "claim": claim,
            "reference_label": gold_label,
            "predicted_letter": pred_letter,
            "predicted_label": letter_to_label[pred_letter],
            "is_correct": int(pred_letter == ref_letter),
            "choice_probabilities": dict(zip(choices, probs.tolist())),
            "prob_generated_token": prob_pred,
            "prob_reference_token": prob_ref,
            "reference_letter": ref_letter
        })

    return results


if acc.is_main_process:
    records = json.load(open(args.data_path))
    data = [(r["claim"].strip(), r["label"].strip()) for r in records]
else:
    data = None
data = broadcast_object_list([data])[0]
my_data = [data[i] for i in range(acc.process_index, len(data), acc.num_processes)]


all_results = []
for i in tqdm(range(0, len(my_data), args.batch_size),
              desc=f"Rank {acc.process_index}",
              disable=not acc.is_local_main_process):
    batch = my_data[i: i + args.batch_size]
    all_results.extend(batch_inference(batch))
    torch.cuda.empty_cache()


os.makedirs(args.save_dir, exist_ok=True)
rank_path = os.path.join(args.save_dir, f"fever_rank{acc.process_index}.json")
with open(rank_path, "w") as f:
    json.dump(all_results, f, indent=2)

acc.wait_for_everyone()

if acc.is_main_process:
    merged = []
    for r in range(acc.num_processes):
        path = os.path.join(args.save_dir, f"fever_rank{r}.json")
        with open(path) as f:
            merged.extend(json.load(f))
        os.remove(path)
    outp = os.path.join(args.save_dir, "fever_inference.json")
    with open(outp, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"✓ Saved combined FEVER results to {outp}")

acc.end_training()
