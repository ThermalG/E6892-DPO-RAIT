from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="meta-llama/Llama-3.2-3B",  
    local_dir="/insomnia001/depts/edu/users/qc2354/models/llama3.2-3B",
    token="Your own Token",
    local_dir_use_symlinks=False  
)
