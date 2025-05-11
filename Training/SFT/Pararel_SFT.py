import json
import os
with open ("/insomnia001/home/qc2354/RLfiles/Outputs/UsableData_Train/pararel_IDtest_trimmed.json", "r") as f:
    inf = json.load(f)
print(len(inf))
list = []
for i in range(len(inf)):
    label = inf[i]["reference_answer"].capitalize()
    question = inf[i]["question"]
    prompt = f"""The following is a Q&A task. Are you sure you can accurately answer the question based on your internal knowledge? You have the option of aknowledging that you are unsure about the answer, by answering: "(your answer). I am unsure.". If you are sure that you can accurately answer the question, please answer: "(your answer). I am sure." \nQuestion: {question}\n Answer: """
    if sum(inf[i]["all_correctness"]) >= 5:
        list.append({"prompt": prompt, "label": label+". I am sure."})
    elif sum(inf[i]["all_correctness"]) < 5:
        list.append({"prompt": prompt, "label": label+". I am unsure."})
with open ("/insomnia001/home/qc2354/RLfiles/Outputs/Clean_Train/pararel_train_ready.json", "w") as f:
    json.dump(list, f, indent=2)

with open ("/insomnia001/home/qc2354/RLfiles/Outputs/Clean_Train/FEVER_train_ready.json", "r") as f:
    fever = json.load(f)
with open ("/insomnia001/home/qc2354/RLfiles/Outputs/Clean_Train/MMLU_train_ready.json", "r") as f:
    mmlu = json.load(f)
with open ("/insomnia001/home/qc2354/RLfiles/Outputs/Clean_Train/WiCE_train_ready.json", "r") as f:
    wice = json.load(f)
total = fever + mmlu + wice + list
with open ("/insomnia001/home/qc2354/RLfiles/Outputs/Clean_Train/total_train_ready.json", "w") as f:
    json.dump(total, f, indent=2)
print(len(fever), len(mmlu), len(wice), len(list), len(total))

print(len(total))


total = [ex for ex in total if len(ex["prompt"]+ex["label"]) <= 2048]
with open ("/insomnia001/home/qc2354/RLfiles/Outputs/Clean_Train/total_train_short_ready.json", "w") as f:
    json.dump(total, f, indent=2)

print(len(total))
