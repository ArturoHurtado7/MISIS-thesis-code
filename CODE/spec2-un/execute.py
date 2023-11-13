# nohup python execute.py > log.txt 2> err.txt &

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
    "spec2-lcnn-attention-am", "spec2-lcnn-attention-oc", "spec2-lcnn-attention-p2s", "spec2-lcnn-attention-sig",
    "spec2-lcnn-fixed-am", "spec2-lcnn-fixed-oc", "spec2-lcnn-fixed-p2s", "spec2-lcnn-fixed-sig",
    "spec2-lcnn-lstmsum-am", "spec2-lcnn-lstmsum-oc", "spec2-lcnn-lstmsum-p2s", "spec2-lcnn-lstmsum-sig"
]

# seeds 
# seeds = [1, 10, 100, 1000, 10000, 100000]
seeds = [1, 10, 100]
    
def train(module_model, seed, log, *args, **kwargs):
    log.write(str(time.strftime("%Y/%m/%d %H:%M")) + f"{module_model}_{seed} \n")
    log.write(str(time.strftime("%Y/%m/%d %H:%M")) + " train \n")
    train = open(f'./log/train_{module_model}_{seed}','w+')
    error = open(f'./log/error_{module_model}_{seed}','w+')
    command = [
        "python", "main.py", 
        "--model-forward-with-file-name",
        "--module-model", module_model, 
        "--num-workers", str(workers), 
        "--epochs", str(epochs), 
        "--no-best-epochs", str(no_best_epochs), 
        "--batch-size", str(batch_size), 
        "--sampler", sampler, 
        "--lr-decay-factor", str(lr_decay_factor), 
        "--lr-scheduler-type", str(lr_sheduler_type), 
        "--lr", str(lr), 
        "--seed", str(seed), 
        "--gpu-device", str(gpu_device), 
        "--lr-steplr-size", str(lr_steplr_size)
    ]
    response = subprocess.run(command, stdout=train, stderr=error, text=True)
    train.close()
    error.close()

    # inference
    log.write(str(time.strftime("%Y/%m/%d %H:%M")) + " inference\n")
    inference_name = f"output_testset_{module_model}_{seed}"
    inference = open(f'./{inference_name}','w+')
    command = [
        "python", "main.py", "--inference", 
        "--model-forward-with-file-name", 
        "--module-model", module_model, 
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

    # rename
    model_name = f"trained_network_{module_model}_{seed}.pt"
    os.rename("trained_network.pt", model_name)

# main
def main():
    log = open(f'./log.txt','w+')
    for seed in seeds:
        for module_model in module_models:
            train(module_model, seed, log)
    log.close()

if __name__ == "__main__":
    main()
