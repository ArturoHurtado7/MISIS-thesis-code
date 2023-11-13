# nohup python results-accents.py &

import os
import glob
import json

results = {}
dictionary = {
    # Activation
    "am": "AM-softmax", # Additive Margin softmax
    "oc": "OC-softmax", # One Class softmax - One-class Learning Towards Synthetic Voice Spoofing Detection
    "sig": "Sigmoid",
    "p2s": "MSE for P2SGrad", # Probability-to-Similarity Gradient - P2SGrad: Refined Gradients for Optimizing Deep Face Models
    
    # Backend
    "attention": "LCNN-attention",
    "fixed": "LCNN-trim-pad",
    "lstmsum": "LCNN-LSTM-sum",
    
    # Nation
    "ar": "Argentina",
    "ve": "Venezuela",
    "pe": "Peru",
    "cl": "Chile",
    "co": "Colombia",
    
    # Gender
    "m": "Male",
    "f": "Female",
}

for filename in glob.glob("./evaluate_*"):
    with open(filename, 'r') as f:
        # split by filename
        _, modelname, accent, seed = filename.split('_')

        # split by modelname
        frontend, _, backend, activation = modelname.split('-')

        # accent
        nation = accent[0:2]
        gender = accent[-1]

        # create item
        item = {
            "model": modelname,
            "frontend": frontend.upper(),
            "backend": dictionary.get(backend, backend),
            "activation": dictionary.get(activation, activation),
            "seed": seed,
            "experiment": len(seed),
            "nation": dictionary.get(nation, nation),
            "gender": dictionary.get(gender, gender),
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
with open(f'{current_folder}_accent.json', 'w') as fp:
    json.dump(list_results, fp, indent=4)
