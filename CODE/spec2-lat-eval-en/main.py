#!/usr/bin/env python
import os
import sys
import torch
import importlib

import core_scripts.other_tools.display as display
import core_scripts.data_io.default_data_io as data_io
import core_scripts.other_tools.list_tools as list_tool
import core_scripts.config_parse.arg_parse as arg_parse
import core_scripts.op_manager.op_manager as op_manager
import core_scripts.nn_manager.nn_manager as nn_manager
import core_scripts.startup_config as startup_config


import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:734"


def training(args, config, device, base_model):
    params = {
        'batch_size':  args.batch_size,
        'shuffle':  args.shuffle,
        'num_workers': args.num_workers,
        'sampler': args.sampler
    }
    display.f_print('Params: %s \n' % (params))

    # Load file list and create data loader
    train_list = list_tool.read_list_from_text(config.trn_list)
    train_set = data_io.NIIDataSetLoader(
        config.trn_set_name, train_list, config.input_dirs, config.input_exts,
        config.input_dims, config.input_reso, config.input_norm, config.output_dirs,
        config.output_exts, config.output_dims, config.output_reso, config.output_norm, './', 
        params = params,
        truncate_seq = config.truncate_seq, 
        min_seq_len = config.minimum_len,
        save_mean_std = True,
        wav_samp_rate = config.wav_samp_rate, 
        global_arg = args
    )

    if config.val_list is None:
        validation_set = None
    else:
        validation_list = list_tool.read_list_from_text(config.val_list)
        validation_set = data_io.NIIDataSetLoader(
            config.val_set_name, validation_list, config.input_dirs, config.input_exts,
            config.input_dims, config.input_reso, config.input_norm, config.output_dirs,
            config.output_exts, config.output_dims, config.output_reso, config.output_norm, './',
            params = params,
            truncate_seq = config.truncate_seq, 
            min_seq_len = config.minimum_len,
            save_mean_std = False,
            wav_samp_rate = config.wav_samp_rate,
            global_arg = args
        )

    # initialize the model and loss function
    model = base_model.Model(
        train_set.get_in_dim(),
        train_set.get_out_dim(),
        args, 
        train_set.get_data_mean_std()
    )

    loss_wrapper = base_model.Loss(args)

    # initialize the optimizer
    optimizer_wrapper = op_manager.OptimizerWrapper(model, args)

    # if necessary, resume training
    if args.trained_model == "":
        checkpoint = None 
        display.f_print('No training checkpoint')
    else:
        checkpoint = torch.load(args.trained_model)
        display.f_print('With training checkpoint')

    # Training
    nn_manager.f_train_wrapper(
        args, model, loss_wrapper, device, optimizer_wrapper, train_set, validation_set, checkpoint
    )

def inference(args, config, device, base_model):
    params = {
        'batch_size':  args.batch_size,
        'shuffle': False,
        'num_workers': args.num_workers
    }
    display.f_print('Params: %s \n' % (params))

    if type(config.test_list) is list:
        test_list = config.test_list
    else:
        test_list = list_tool.read_list_from_text(config.test_list)

    test_set = data_io.NIIDataSetLoader(
        config.test_set_name, test_list, config.test_input_dirs, config.input_exts, 
        config.input_dims, config.input_reso, config.input_norm, config.test_output_dirs, 
        config.output_exts, config.output_dims, config.output_reso, config.output_norm, './',
        params = params,
        truncate_seq = None,
        min_seq_len = None,
        save_mean_std = False,
        wav_samp_rate = config.wav_samp_rate,
        global_arg = args
    )

    # initialize model
    model = base_model.Model(
        test_set.get_in_dim(),
        test_set.get_out_dim(),
        args
    )

    if args.trained_model == "":
        print("No model is loaded by ---trained-model for inference")
        print("By default, load %s%s" % (args.save_trained_name, args.save_model_ext))
        checkpoint = torch.load("%s%s" % (args.save_trained_name, args.save_model_ext))
    else:
        checkpoint = torch.load(args.trained_model)

    # do inference and output data
    nn_manager.f_inference_wrapper(
        args, model, device, test_set, checkpoint
    )

def main():
    
    torch.cuda.empty_cache()
    
    display.f_print_w_date("Start program", level='h')
    args = arg_parse.f_args_parsed()

    config = importlib.import_module(args.module_config)
    base_model = importlib.import_module(args.module_model)

    display.f_print("Load module: %s" % (config))
    display.f_print("Load module: %s" % (base_model))

    # initialization
    startup_config.set_random_seed(args.seed, args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        if args.gpu_device != 0:
            device = torch.device(f'cuda:{args.gpu_device}')
            fraction_to_reserve = 0.95
            torch.cuda.set_per_process_memory_fraction(
                fraction_to_reserve, device=device)
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    display.f_print("Random seed: %s" % (args.seed))
    display.f_print("GPU Device: %s" % (args.gpu_device))
    display.f_print("Device: %s" % (device))

    if args.inference:
        display.f_print("Inference")
        inference(args, config, device, base_model)
    else:
        display.f_print("Training")
        training(args, config, device, base_model)

    display.f_print_w_date("Finish program", level='h')

if __name__ == "__main__":
    main()
