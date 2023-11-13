#!/usr/bin/python
""" 
Wrapper to parse the score file and compute EER and min tDCF

Usage:
python 00_evaluate.py log_output_testset

The log_output_testset is produced by the pytorch code, for
example, ./lfcc-lcnn-lstmsum-am/01/__pretrained/log_output_testset

It has information like:
...
Generating 71230,LA_E_9999427,0,43237,0, time: 0.005s
Output, LA_E_9999487, 0, 0.172325
...
(See README for the format of this log)

This script will extract the line starts with "Output, ..."

"""

import os
import sys
import numpy as np
from sandbox import eval_asvspoof

def parse_txt(file_path, accent):
    bonafide = []
    spoofed = []
    print('file_path: ', file_path)
    print('accent: ', accent)
    with open(file_path, 'r') as file_ptr:
        for line in file_ptr:
            if line.startswith('Output,'):
                #Output, LA_E_9999487, 0, 0.172325
                temp = line.split(',')
                flag = int(temp[2])
                name = temp[1]
                name_list = name.split("-")
                len_namelist = len(name_list)
                score = temp[-1].strip()
                if (len_namelist == 1 and accent in name_list[0] or
                    len_namelist == 2 and accent in name_list[1] or
                    len_namelist == 3 and accent in name_list[1] or
                    len_namelist == 4 and accent in name_list[2]):
                    if score.replace(".", "").replace("-","").isnumeric():
                        if flag:
                            bonafide.append(float(score))
                        else:
                            spoofed.append(float(score))
                    else:
                        print(f"file name: {name}")
                        print(f"score is not numeric: {score}")

    bonafide = np.array(bonafide)
    spoofed = np.array(spoofed)
    return bonafide, spoofed

if __name__ == "__main__":
    
    data_path = sys.argv[1]
    accent = sys.argv[2]
    bonafide, spoofed = parse_txt(data_path, accent)
    mintDCF, eer, threshold = eval_asvspoof.tDCF_wrapper(bonafide, spoofed)
    print("Score file: {:s}".format(data_path))
    print("mintDCF: {:1.4f}".format(mintDCF))
    print("EER: {:2.3f}%".format(eer * 100))
    print("Threshold: {:f}".format(threshold))