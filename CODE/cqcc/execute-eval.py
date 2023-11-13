# nohup python execute-eval.py > log-eval.txt 2> err-eval.txt &

import os
import time
import subprocess

# variables
workers = 3
epochs = 100
no_best_epochs = 50
batch_size = 64
sampler = "block_shuffle_by_length"
lr_decay_factor = 0.5
lr_steplr_size = 10
lr_sheduler_type = 1
lr = 0.0003
gpu_device = 2

# models
module_models = [
    "cqcc-lcnn-attention-am", "cqcc-lcnn-attention-oc", "cqcc-lcnn-attention-p2s", "cqcc-lcnn-attention-sig",
    "cqcc-lcnn-fixed-am", "cqcc-lcnn-fixed-oc", "cqcc-lcnn-fixed-p2s", "cqcc-lcnn-fixed-sig",
    "cqcc-lcnn-lstmsum-am", "cqcc-lcnn-lstmsum-oc", "cqcc-lcnn-lstmsum-p2s", "cqcc-lcnn-lstmsum-sig"
]

# seeds 
seeds = [1, 10, 100, 1000, 10000, 100000]

def train(module_model, seed, log, *args, **kwargs):
    log.write(str(time.strftime("%Y/%m/%d %H:%M")) + f"{module_model}_{seed} \n")
    log.write(str(time.strftime("%Y/%m/%d %H:%M")) + " train \n")

    # evaluate
    inference_name = f"output_testset_{module_model}_{seed}"
    log.write(str(time.strftime("%Y/%m/%d %H:%M")) + " evaluate\n")
    evaluate = open(f'./evaluate_{module_model}_{seed}','w+')
    command = ["python", "evaluate.py", inference_name]
    response = subprocess.run(command, stdout=evaluate, text=True)
    evaluate.close()

# main
def main():
    log = open(f'./log-eval.txt','w+')
    for seed in seeds:
        for module_model in module_models:
            train(module_model, seed, log)
    log.close()

if __name__ == "__main__":
    main()
