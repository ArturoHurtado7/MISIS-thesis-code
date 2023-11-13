#!/usr/bin/env python

import torch
import torch.nn as nn
import core_scripts.other_tools.display as nii_display

class ParallelModel(nn.parallel.DistributedDataParallel):
    def __init__(self, model, *input, **kwargs):
        super(ParallelModel, self).__init__(*input, **kwargs)
        self.module = nn.DataParallel(model).module

    def forward(self, *input, **kwargs):
        return self.module.forward(*input, **kwargs)

    def prepare_mean_std(self, *input, **kwargs):
        return self.module.prepare_mean_std(*input, **kwargs)

    def normalize_input(self, *input, **kwargs):
        return self.module.normalize_input(*input, **kwargs)

    def normalize_target(self, *input, **kwargs):
        return self.module.normalize_target(*input, **kwargs)

    def denormalize_output(self, *input, **kwargs):
        return self.module.denormalize_output(*input, **kwargs)
