# nohup python execute-eval-accents.py > log-eval.txt 2> err-eval.txt &

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
    "ldfcc-lcnn-attention-am", "ldfcc-lcnn-attention-oc", "ldfcc-lcnn-attention-p2s", "ldfcc-lcnn-attention-sig",
    "ldfcc-lcnn-fixed-am", "ldfcc-lcnn-fixed-oc", "ldfcc-lcnn-fixed-p2s", "ldfcc-lcnn-fixed-sig",
    "ldfcc-lcnn-lstmsum-am", "ldfcc-lcnn-lstmsum-oc", "ldfcc-lcnn-lstmsum-p2s", "ldfcc-lcnn-lstmsum-sig"
]

accents = [
    "arf", "vef", "pef", "clf", "cof",
    "arm", "vem", "pem", "clm", "com"
]

# seeds 
seeds = [1, 10, 100, 1000, 10000, 100000]

def train(module_model, accent, seed, log, *args, **kwargs):
    log.write(str(time.strftime("%Y/%m/%d %H:%M")) + f"{module_model}_{seed} \n")
    log.write(str(time.strftime("%Y/%m/%d %H:%M")) + " train \n")

    # evaluate
    inference_name = f"output_testset_{module_model}_{seed}"
    log.write(str(time.strftime("%Y/%m/%d %H:%M")) + " evaluate\n")
    evaluate = open(f'./evaluate_{module_model}_{accent}_{seed}','w+')
    command = ["python", "evaluate-accents.py", inference_name, accent]
    response = subprocess.run(command, stdout=evaluate, text=True)
    evaluate.close()

# main
def main():
    log = open(f'./log-eval.txt','w+')
    for seed in seeds:
        for module_model in module_models:
            for accent in accents:
                if seed == 100000 and module_model == "ldfcc-lcnn-attention-am":
                    continue
                train(module_model, accent, seed, log)
    log.close()

if __name__ == "__main__":
    main()
