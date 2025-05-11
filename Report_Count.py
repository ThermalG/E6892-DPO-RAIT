import json
import pandas as pd
import matplotlib.pyplot as plt
with open("/insomnia001/home/qc2354/RLfiles/Outputs/llama3.2_3B_inf_after_DPO/MMLU_ID_inference_test.json", "r") as f:
    dpo_mmlu_id = json.load(f)
with open("/insomnia001/home/qc2354/RLfiles/Outputs/llama3.2_3B_inf_after_4_train/MMLU_ID_inference_test.json", "r") as f:
    sft_mmlu_id = json.load(f)
with open("/insomnia001/home/qc2354/RLfiles/Outputs/llama3.2_3B_inf_before_MMLU_train/MMLU_ID_inference_test.json", "r") as f:
    before_mmlu_id = json.load(f)
with open("/insomnia001/home/qc2354/RLfiles/Outputs/llama3.2_3B_inf_before_MMLU_train/WiCE_test.json", "r") as f:
    before_wice = json.load(f)
with open("/insomnia001/home/qc2354/RLfiles/Outputs/llama3.2_3B_inf_after_DPO/WiCE_test.json", "r") as f:
    dpo_wice = json.load(f)
with open("/insomnia001/home/qc2354/RLfiles/Outputs/llama3.2_3B_inf_after_4_train/WiCE_test.json", "r") as f:
    sft_wice = json.load(f)

correct_dpo =0
good_sure = 0
good_unsure = 0
bad_sure = 0
bad_unsure = 0

def collect_data(data):
    correct_count =0
    good_sure = 0
    good_unsure = 0
    bad_sure = 0
    bad_unsure = 0
    sureness_token = set()
    for i in range(len(data)):
        sureness_token.add(data[i]["sureness_token"])
        if data[i]["is_correct"] == 1:
            correct_count += 1
        if data[i]["sureness_token"] == "sure" or data[i]["sureness_token"] == "yes" or data[i]["sureness_token"] == "certain": 
            data[i]["sureness_token"] = 1
            if data[i]["is_correct"] == 1:
                good_sure += 1
            if data[i]["is_correct"] == 0:
                bad_sure += 1
        elif data[i]["sureness_token"] == "unsure" or data[i]["sureness_token"] == "not" or data[i]["sureness_token"] == "uncertain": 
            data[i]["sureness_token"] = 0
            if data[i]["is_correct"] == 0:
                good_unsure += 1
            if data[i]["is_correct"] == 1:
                bad_unsure += 1
        
    current_dict = { 
        "correct_count": correct_count,
        "good_sure": good_sure,
        "good_unsure": good_unsure,
        "bad_sure": bad_sure,
        "bad_unsure": bad_unsure,
        "good_unsure_rate": good_unsure / (good_unsure + bad_unsure) if (good_unsure + bad_unsure) != 0 else 0,
        "good_sure&unsure_rate": (good_sure + good_unsure) / (good_sure + good_unsure + bad_sure + bad_unsure) if (good_sure + good_unsure + bad_sure + bad_unsure) != 0 else 0,
        "answer_accuracy": correct_count / len(data) if len(data) != 0 else 0,
        "unsure_rate": (good_unsure + bad_unsure) / len(data) if len(data) != 0 else 0
    }
    print(sureness_token)
    return current_dict

my_dict = {}
my_dict["dpo_mmlu_id"] = collect_data(dpo_mmlu_id)
my_dict["sft_mmlu_id"] = collect_data(sft_mmlu_id)
my_dict["base_mmlu_id"] = collect_data(before_mmlu_id)
my_dict["dpo_wice"] = collect_data(dpo_wice)
my_dict["sft_wice"] = collect_data(sft_wice)
my_dict["base_wice"] = collect_data(before_wice)
print(my_dict)



import matplotlib.pyplot as plt
import pandas as pd


rows = []
for name, stats in my_dict.items():
    model, dataset = name.split('_', 1)
    row = {'Model': model.upper(), 'Dataset': dataset.upper(), **stats}
    rows.append(row)

df = pd.DataFrame(rows)
df_MMLU = df[df["Dataset"] == "MMLU_ID"]
df_WiCE = df[df["Dataset"] == "WICE"]
plt.figure(figsize=(10, 6))
plt.bar(df_MMLU["Model"], df_MMLU["good_unsure_rate"], color=['blue', 'black', 'green'])
plt.xlabel("Model", fontsize=28)
plt.ylabel("Refusal Accuracy", fontsize=28)
plt.ylim(0.5, 0.7)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("good_unsure_rate_MMLU.png")
plt.show()
plt.figure(figsize=(10, 6))
plt.bar(df_WiCE["Model"], df_WiCE["good_unsure_rate"], color=['blue', 'black', 'green'])
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlabel("Model", fontsize=28)
plt.ylabel("Refusal Accuracy", fontsize=28)
plt.ylim(0.5, 0.7)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("good_unsure_rate_WiCE.png")
plt.show()
plt.close()

print(df)

