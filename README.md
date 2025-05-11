# E6892 Refusal Awareness Instruction Tuning for lightweight LLMs
A collaborative codebase for EECS 6892 grading


## How to use
1. Clone repository;
2. decrompress datasets.zip into the same directory;
3. Modify `token` argument in the test run notebook and paths in all inference and training scripts;
4. Install dependencies. You can achieve such by running the command below;
    ```
    pip install -r requirements.txt
    ```
5. Run using the command for optimal speed and results
   ```accelerate launch --num_processes 2 --multi_gpu path/to/the/script/you/want/to/run.py --batch-size 64 --repeats 10 [--compile]```
