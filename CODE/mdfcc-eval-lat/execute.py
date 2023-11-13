# nohup python execute.py > log.txt 2> err.txt &

import os
import time
import subprocess

# variables
workers = 3
epochs = 100
no_best_epochs = 30
batch_size = 64
sampler = "block_shuffle_by_length"
lr_decay_factor = 0.5
lr_steplr_size = 10
lr_sheduler_type = 1
lr = 0.0003
gpu_device = 0

# models
# module_models = [
#     "mdfcc-lcnn-attention-am", "mdfcc-lcnn-attention-oc", "mdfcc-lcnn-attention-p2s", "mdfcc-lcnn-attention-sig",
#     "mdfcc-lcnn-fixed-am", "mdfcc-lcnn-fixed-oc", "mdfcc-lcnn-fixed-p2s", "mdfcc-lcnn-fixed-sig",
#     "mdfcc-lcnn-lstmsum-am", "mdfcc-lcnn-lstmsum-oc", "mdfcc-lcnn-lstmsum-p2s", "mdfcc-lcnn-lstmsum-sig"
# ]

module_models = [
    "mdfcc-lcnn-attention-am"
]


# seeds 
# seeds = [1, 10, 100, 1000, 10000, 100000]
seeds = [1000]

def train(module_model, seed, log, *args, **kwargs):
    log.write(str(time.strftime("%Y/%m/%d %H:%M")) + f"{module_model}_{seed} \n")

    # inference
    log.write(str(time.strftime("%Y/%m/%d %H:%M")) + " inference\n")
    inference_name = f"output_testset_{module_model}_{seed}"
    model_name = f"trained_network_{module_model}_{seed}.pt"
    inference = open(f'./{inference_name}','w+')
    command = [
        "python", "main.py", "--inference", 
        "--model-forward-with-file-name", 
        "--module-model", module_model, 
        "--trained-model", model_name,
        "--num-workers", str(workers), 
        "--gpu-device", str(gpu_device)
    ]
    response = subprocess.run(command, stdout=inference, text=True)
    inference.close()

    # evaluate
    log.write(str(time.strftime("%Y/%m/%d %H:%M")) + " evaluate\n")
    evaluate = open(f'./evaluate_{module_model}_{seed}','w+')
    command = ["python", "evaluate.py", inference_name]
    response = subprocess.run(command, stdout=evaluate, text=True)
    evaluate.close()

# main
def main():
    log = open(f'./log.txt','w+')
    for seed in seeds:
        for module_model in module_models:
            train(module_model, seed, log)
    log.close()

if __name__ == "__main__":
    main()
