import json
import random
with open ("/insomnia001/home/qc2354/RLfiles/Outputs/UsableData_Train/fever_inference2.json", "r") as f:
    inf = json.load(f)
with open ("/insomnia001/home/qc2354/RLfiles/Data/R-Tuning-data/FEVER/fever_10k.json", "r") as f:
    fever = json.load(f)
label_to_letter = {
    "SUPPORTED":        "A",
    "SUPPORTS":         "A",
    "REFUTED":          "B",
    "REFUTES":          "B",
    "NOT ENOUGH INFO":  "C"
}
my_list = []
choices = {"A", "B", "C"}
print(choices)

for i in range(len(inf)):
    claim = fever[i]["claim"]
    evidence = fever[i]["evidence"]
    label = label_to_letter[fever[i]["label"]]
    prompt = f"""The following is a multiple choice question about claim and evidence. Are you sure you can accurately answer the question based on your internal knowledge? You have the option of aknowledging that you are unsure about the answer, by answering: "(your answer). I am unsure.". If you are sure that you can accurately answer the question, please answer: "(your answer). I am sure." \nClaim: {claim} \nEvidence: {evidence}\nA: Supports\nB: Refutes\nC: Not enough information\nAnswer: """
    wrong_choices = choices - {label}
    wrong_choice = random.choice(list(wrong_choices))
    chosen = inf[i]["predicted_letter"]
    if inf[i]["is_correct"] == 1:
        my_list.append({"prompt": prompt, "selected": label+". I am sure.", "rejected": wrong_choice+". I am unsure."})
    elif inf[i]["is_correct"] == 0:
        my_list.append({"prompt": prompt, "selected": label+". I am unsure.", "rejected": chosen+". I am sure."})

print(my_list[0])
with open ("/insomnia001/home/qc2354/RLfiles/Outputs/Clean_DPO/FEVER_DPO_ready.json", "w") as f:
    json.dump(my_list, f, indent=2)
    