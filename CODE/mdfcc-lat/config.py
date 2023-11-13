#!/usr/bin/env python
import os

trn_set_name = 'asvspoof2019_trn'
val_set_name = 'asvspoof2019_val'

tmp = '/home/monitor/DATA/asvspoof2019_LA_LAT'

trn_list = tmp + '/scp/train.lst'
val_list = tmp + '/scp/val.lst'
input_dirs = [tmp + '/train_dev']
input_dims = [1]
input_exts = ['.wav']
input_reso = [1]
input_norm = [False]
output_dirs = []
output_dims = [1]
output_exts = ['.bin']
output_reso = [1]
output_norm = [False]
wav_samp_rate = 16000
truncate_seq = None
minimum_len = None
optional_argument = [tmp + '/protocol.txt']
test_set_name = 'asvspoof2019_test'
test_list = tmp + '/scp/test.lst'
test_input_dirs = [tmp + '/eval']
test_output_dirs = []
