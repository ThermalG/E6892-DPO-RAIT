import os
import json
import random
with open ("/insomnia001/home/qc2354/RLfiles/Data/R-Tuning-data/MMLU/MMLU_ID_train.json", "r") as f:
    mmlu_data = json.load(f)
with open ("/insomnia001/home/qc2354/RLfiles/Outputs/UsableData_Train/MMLU_ID_inference_train.json", "r") as f:
    inference_data = json.load(f)
print(mmlu_data['abstract_algebra'][0])
count = 0
my_list = []
choices = {" A", " B", " C", " D"}
for dic in inference_data:
        label = dic["reference_answer"]
        wrong_choices = choices - {label}
        wrong_choice = random.choice(list(wrong_choices))
        chosen = dic["predicted"]
        subject = dic["subject"]
        question = dic["question"]
        choiceA = dic["options"][0]
        choiceB = dic["options"][1]
        choiceC = dic["options"][2]
        choiceD = dic["options"][3]
        prompt = f"""The following is a multiple choice question about {subject}. Are you sure you can accurately answer the question based on your internal knowledge? You have the option of aknowledging that you are unsure about the answer, by answering: "(your answer). I am unsure.". If you are sure that you can accurately answer the question, please answer: "(your answer). I am sure." \nQuestion: {question} \nA: {choiceA}\nB: {choiceB}\n C: {choiceC}\n D: {choiceD}\n Answer: """
        if dic["is_correct"] == 1:
            my_list.append({"prompt": prompt, "selected": label+". I am sure.", "rejected": wrong_choice+". I am unsure."})
        elif dic["is_correct"] == 0:
            my_list.append({"prompt": prompt, "selected": label+". I am unsure.", "rejected": chosen+". I am sure."})

print(my_list[0])

with open ("/insomnia001/home/qc2354/RLfiles/Outputs/Clean_DPO/MMLU_DPO_ready.json", "w") as f:
    json.dump(my_list, f, indent=2)