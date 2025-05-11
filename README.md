# E6892 Refusal Awareness Instruction Tuning for lightweight LLMs
A collaborative codebase for EECS 6892 grading

The current repository contains three parts: the inference folder, the training folder and the outputs folder.

The inference folder has scripts for generating the model's answers to the datasets' questions;

The training folder has scripts to clean the model's outputs for training format, both SFT and DPO training scripts, and the deepspeed training setting ds_config.json. 
## How to use
1. Clone repository;
2. download and decrompress ```datasets.zip``` into the same directory, available at our [Lion Drive](https://drive.google.com/drive/u/1/folders/1wjTKzzzDZjisEQlYadnN0MHoMN3MAFOB) (access with your Columbia Account, including the MMLU, FEVER, WiCE, and ParaRel dataset);
3. Install dependencies. You can achieve such by running the command below;
    ```
    pip install -r requirements.txt
    ```
4. Prepare a local LLM (we used LLaMA-3.2-3b) for inference and training (or use our ```download.py``` script to download with your own hugging face token);
5. Use the inference folder for generating answers to the datasets, for accelerated use on multiple GPUs, please use:
   ```
   accelerate launch --num_processes 2 --multi_gpu <inference_file_name.py>
   ```
   for accelerated use on single GPU, please use:
   ```
   accelerate launch <inference_file_name.py>
   ```
   Please modify the batch parameters according to your computational resources;
6. Use the DataCleaning scripts inside the training folder for adjusting the codes for training format (see Training/DPO and Training/SFT) and then use the merge scripts to combine them and shorten them for inference length (If you are using LLaMA-3.2-3B, you can directly use ```Outputs/Clean_DPO/total_DPO_ready.json``` for DPO training and ```Outputs/Clean_Train/total_train_short_ready.json``` for SFT training;
7. Use the DPO_train.py and SFT_train.py with
    ```
    deepspeed <train_file_name.py> ds_config.json
    ```
8. Examine our testing outcomes of the base model at Outputs/llama3.2_3B_inf_before_MMLU_train; the SFT model at Outputs/llama3.2_3B_inf_after_4_train; the DPO model at Outputs/llama3.2_3B_inf_after_DPO;
9. Replicating our results using the ```Report_Count.py```.
