import json
import os
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
list = []
for i in range(len(inf)):
    claim = fever[i]["claim"]
    evidence = fever[i]["evidence"]
    label = label_to_letter[fever[i]["label"]]
    prompt = f"""The following is a multiple choice question about claim and evidence. Are you sure you can accurately answer the question based on your internal knowledge? You have the option of aknowledging that you are unsure about the answer, by answering: "(your answer). I am unsure.". If you are sure that you can accurately answer the question, please answer: "(your answer). I am sure." \nClaim: {claim} \nEvidence: {evidence}\nA: Supports\nB: Refutes\nC: Not enough information\nAnswer: """
    if inf[i]["is_correct"] == 1:
        list.append({"prompt": prompt, "label": label+". I am sure."})
    elif inf[i]["is_correct"] == 0:
        list.append({"prompt": prompt, "label": label+". I am unsure."})
print(list[0])
with open ("/insomnia001/home/qc2354/RLfiles/Outputs/Clean_Train/FEVER_train_ready.json", "w") as f:
    json.dump(list, f, indent=2)
