import json

choices = ["A", "B", "C"]
candidate_answer = ['supported.', 'partially_supported.', 'not_supported.']
label_map = {'supported': "A", 'partially_supported': "B", 'not_supported': "C"}

def format_question(ex):
    evidence = " ".join(ex["evidence"])
    claim = ex["claim"]
    block = f"""The following is a multiple choice question about evidence and claim. Are you sure you can accurately answer the question based on your internal knowledge? You have the option of aknowledging that you are unsure about the answer, by answering: "(your answer). I am unsure.". If you are sure that you can accurately answer the question, please answer: "(your answer). I am sure."  Evidence: {evidence}\nClaim: {claim}\nQuestion: Does the evidence support the claim?"""
    for i, ch in enumerate(choices):
        block += f"\n{ch}: {candidate_answer[i]}"
    block += "\nAnswer:"
    return block

with open("/insomnia001/home/qc2354/RLfiles/Data/R-Tuning-data/WiCE/wice_train.json", "r") as f:
    wice_data = json.load(f)
with open("/insomnia001/home/qc2354/RLfiles/Outputs/UsableData_Train/WiCE_inference.json", "r") as f:
    inference_data = json.load(f)

list = []
for i in range(len(wice_data)):
    prompt = format_question(wice_data[i])
    if inference_data[i]["is_correct"] == 1:
        label = label_map[wice_data[i]["label"]] + ". I am sure."
    elif inference_data[i]["is_correct"] == 0:
        label = label_map[wice_data[i]["label"]] + ". I am unsure."
    list.append({"prompt": prompt, "label": label})
print(list[0])

with open("/insomnia001/home/qc2354/RLfiles/Outputs/Clean_Train/WiCE_train_ready.json", "w") as f:
    json.dump(list, f, indent=2)
        