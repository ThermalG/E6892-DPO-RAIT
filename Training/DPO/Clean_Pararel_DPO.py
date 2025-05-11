import json
import os
import random
with open ("/insomnia001/home/qc2354/RLfiles/Outputs/UsableData_Train/pararel_IDtest_trimmed.json", "r") as f:
    inf = json.load(f)
print(len(inf))
my_list = []
for i in range(len(inf)):
    label = inf[i]["reference_answer"].capitalize()
    all_correctness = inf[i]["all_correctness"]
    all_output = inf[i]["all_outputs"]
    question = inf[i]["question"]
    wrong_choices = []
    for j in range(len(all_output)):
        if all_correctness[j] == 0:
            wrong_choices.append(all_output[j])
    prompt = f"""The following is a Q&A task. Are you sure you can accurately answer the question based on your internal knowledge? You have the option of aknowledging that you are unsure about the answer, by answering: "(your answer). I am unsure.". If you are sure that you can accurately answer the question, please answer: "(your answer). I am sure." \nQuestion: {question}\n Answer: """
    if sum(inf[i]["all_correctness"]) == 10:
        print("all correct")
    elif sum(inf[i]["all_correctness"]) >= 5:
        wrong_choice = random.choice(wrong_choices)
        my_list.append({"prompt": prompt, "selected": label+". I am sure.", "rejected": wrong_choice+" I am unsure."})
    elif sum(inf[i]["all_correctness"]) < 5:
        wrong_choice = random.choice(wrong_choices)
        my_list.append({"prompt": prompt, "selected": label+". I am unsure.", "rejected": wrong_choice+" I am sure."})
print(my_list[0])

with open ("/insomnia001/home/qc2354/RLfiles/Outputs/Clean_DPO/pararel_DPO_ready.json", "w") as f:
    json.dump(my_list, f, indent=2)
