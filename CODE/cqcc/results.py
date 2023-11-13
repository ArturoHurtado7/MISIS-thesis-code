# nohup python results.py &
import os
import glob
import json

results = {}
dictionary = {
    "am": "AM-softmax", # Additive Margin softmax
    "oc": "OC-softmax", # One Class softmax - One-class Learning Towards Synthetic Voice Spoofing Detection
    "sig": "Sigmoid",
    "p2s": "MSE for P2SGrad", # Probability-to-Similarity Gradient - P2SGrad: Refined Gradients for Optimizing Deep Face Models
    "attention": "LCNN-attention",
    "fixed": "LCNN-trim-pad",
    "lstmsum": "LCNN-LSTM-sum",
}

for filename in glob.glob("./evaluate_*"):
    with open(filename, 'r') as f:
        # split by filename
        filename_split = filename.split('_')
        modelname = filename_split[1]
        seed = filename_split[-1]

        # split by modelname
        modelname_split = modelname.split('-')
        frontend = modelname_split[0]
        backend = modelname_split[-2]
        activation = modelname_split[-1]

        # create item
        item = {
            "model": modelname,
            "frontend": frontend.upper(),
            "backend": dictionary.get(backend, backend),
            "activation": dictionary.get(activation, activation),
            "seed": seed,
            "experiment": len(seed),
        }

        # add metrics
        for line in f:
            if ":" in line:
                metric, value = line.split(':', 1)
                value = value.strip().replace("%", "")
                if metric in ["mintDCF", "EER"]:
                    item[metric] = float(value)

        # group by model
        if results.get(modelname):
            results[modelname].append(item)
        else:
            results[modelname] = [item]

# order by eer
for result in results.values():
    result = sorted(result, key=lambda d: d['EER']) 
    for index, value in enumerate(result):
        value['index'] = index

# flatten results
list_results = []
for result in results.values():
    list_results.extend(result)

# export results
current_path = os.getcwd()
_, current_folder = current_path.rsplit('/', 1)
with open(f'{current_folder}.json', 'w') as fp:
    json.dump(list_results, fp, indent=4)
