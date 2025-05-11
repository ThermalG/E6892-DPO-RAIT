import json

with open ("/insomnia001/home/qc2354/RLfiles/Outputs/Clean_DPO/FEVER_DPO_ready.json", "r") as f:
    fever = json.load(f)
with open ("/insomnia001/home/qc2354/RLfiles/Outputs/Clean_DPO/MMLU_DPO_ready.json", "r") as f:
    mmlu = json.load(f)
with open ("/insomnia001/home/qc2354/RLfiles/Outputs/Clean_DPO/WiCE_DPO_ready.json", "r") as f:
    wice = json.load(f)
with open ("/insomnia001/home/qc2354/RLfiles/Outputs/Clean_DPO/pararel_DPO_ready.json", "r") as f:
    pararel = json.load(f)
total = fever + mmlu + wice + pararel
print(len(fever), len(mmlu), len(wice), len(pararel), len(total))
total = [ex for ex in total if len(ex["prompt"]+ex["selected"]+ex["rejected"]) <= 2048]
print(len(total))
with open ("/insomnia001/home/qc2354/RLfiles/Outputs/Clean_DPO/total_DPO_ready.json", "w") as f:
    json.dump(total, f, indent=2)
