'''
USAGE:
mprof run python bench_mem_pt_ov.py \
-d /home/rpanchum/upenn/data/sidd-tcga/val_data_sidd-tcga_4mod_244/data_processed_1.csv \
-r 4 -p OV-NNCF-PRUN06_ICX \
2>&1 | tee infer_logs/bench_mem_OV-NNCF-PRUN06_ICX.log

OR

bash bench_mem_prof.sh

'''
from pathlib import Path

import os
import sys
import time
import torch
import torch.onnx
from torch.autograd import Variable

import argparse
import pickle
from openvino.inference_engine import IECore

from GANDLF.compute import validate_network
from GANDLF.models import global_models_dict
from GANDLF.utils import (
    load_ov_model,
)

from readConfig_ma import readConfig

#from memory_profiler import memory_usage

#run_mode_list=( 0 1 2 3 4 5)
#fmwrk_mode_list=( 'PT-FP32' 'OV-FP32' 'OV-POT-INT8' 'OV-NNCF-INT8' 'OV-NNCF-PRUN06' 'OV-NNCF-QAT-Prun06-KD' )

parser = argparse.ArgumentParser(
    description='Benchmarking for the 3D ResUnet models')
parser.add_argument('-m', '--model_name',
                    help='The model name', default='resunet')
parser.add_argument('-r', '--run_mode', type=int, required=False,
                    help='Run Mode, choose from 0-5')
parser.add_argument('-d', '--data_csv',
                    help='The path to data csv containing path to images and labels.')
parser.add_argument('-p', '--parameters_file', required=False,
                    help='Config yaml file or the parameter file')
parser.add_argument('-o', '--output_dir', required=False, default='./output_benchmarking/',
                    help='Output directory to store segmenation results')


parser.add_argument('-v', '--verbose', required=False,
                    help='Whether to print verbose results')
args = parser.parse_args()

from tqdm import tqdm
import torchio

from load_data_inference import load_data

def load_torch_model(path, model, key = "model_state_dict"):
    main_dict = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(main_dict["model_state_dict"], strict=False)
    model.eval()
    return model


#@profile
def benchmark_pt_model(model, parameters, data_loader):
    st_time = time.time()
    epoch_valid_loss, epoch_valid_metric = validate_network(
                model, data_loader, scheduler=None, params=parameters, epoch=0, mode="validation")
    ed_time = time.time()
    print(f"***** Avg inference Time for the PyTorch Model is: {str((ed_time - st_time)/len(data_loader))}")

#@profile
def benchmark_ov_model(model, parameters, data_loader, input_blob, output_blob, model_type='FP32'):
    st_time = time.time()
    epoch_valid_loss, epoch_valid_metric = validate_network(
                model, data_loader, scheduler=None, params=parameters, epoch=0, mode="inference")
    ed_time = time.time()
    print(f"***** Avg inference Time for the OV {model_type} Model is: {str((ed_time - st_time)/len(data_loader))}")


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if args.run_mode == 0:
        parameters = readConfig(config_file="./configs/train_config.yaml")
        parameters['train_mode'] = False
        parameters['device'] = device

        model = global_models_dict[parameters["model"]["architecture"]](parameters=parameters)

        val_dataloader, parameters = load_data(args.data_csv, parameters, args.output_dir, train_mode=False)

        ##Original PyTorch Model
        orig_pth = "./models/0/resunet_best.pth.tar"
        model = load_torch_model(orig_pth, model)
        benchmark_pt_model(model, parameters, val_dataloader)

    elif args.run_mode > 0 :
        parameters = readConfig(config_file="./configs/train_config_ov.yaml")
        model = global_models_dict[parameters["model"]["architecture"]](parameters=parameters)
        val_dataloader, parameters = load_data(args.data_csv, parameters, args.output_dir, train_mode=False)

        parameters['train_mode'] = False
        parameters['device'] = device
        parameters['output_dir'] = args.output_dir
        parameters['channel_keys'] = ['1']
        parameters['problem_type'] = 'segmentation'
        parameters["weights"] = None
        parameters["class_weight"] = None
        parameters["label_keys"] = ['label']

        if args.run_mode == 1:
            ## Original OpenVINO FP32 Model
            fp32_ir_path = "./models/0/resunet_best.xml"
            exec_net, input_blob, output_blob = load_ov_model(fp32_ir_path)
            parameters["model"]["IO"] = [input_blob, output_blob]
            benchmark_ov_model(exec_net, parameters, val_dataloader, input_blob, output_blob, model_type='FP32')
        elif args.run_mode == 2:
            # # ###OpenVINO POT INT8 Model
            int8_ir_path = "./ov-irs/int8/resunet_best.xml"
            exec_net, input_blob, output_blob = load_ov_model(int8_ir_path)
            parameters["model"]["IO"] = [input_blob, output_blob]
            benchmark_ov_model(exec_net, parameters, val_dataloader, input_blob, output_blob, model_type='OV POT INT8')
        elif args.run_mode == 3:
            # # ###OpenVINO + NNCF QAT model
            int8_ir_path = "./ov-irs/nncf-int8-pruned/resunet_best.xml"
            exec_net, input_blob, output_blob = load_ov_model(int8_ir_path)
            parameters["model"]["IO"] = [input_blob, output_blob]
            benchmark_ov_model(exec_net, parameters, val_dataloader, input_blob, output_blob, model_type='NNCF QAT + OV INT8')
        elif args.run_mode == 4:
            # # # ###OpenVINO + NNCF Filter Pruned model
            int8_ir_path = "./ov-irs/fp32/prun5060-pruned/resunet_best.xml"
            exec_net, input_blob, output_blob = load_ov_model(int8_ir_path)
            parameters["model"]["IO"] = [input_blob, output_blob]
            benchmark_ov_model(exec_net, parameters, val_dataloader, input_blob, output_blob, model_type='NNCF Filter Pruned + OV FP32 --> Pruning level 0.6')
        elif args.run_mode == 5:
            int8_ir_path = "./ov-irs/kd-int8-pruned/resunet_best.xml"
            exec_net, input_blob, output_blob = load_ov_model(int8_ir_path)
            parameters["model"]["IO"] = [input_blob, output_blob]
            benchmark_ov_model(exec_net, parameters, val_dataloader, input_blob, output_blob, model_type='NNCF QAT + Filter Pruned 0.6 + KD')

if __name__ == '__main__':
    main()
