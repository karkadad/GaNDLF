# Usage: python run_unet_nncf.py -i ./3dresunet_exp_nncf/data_dir/resunet_best.pth.tar -c ./3dresunet_exp_nncf/nncf_config.json -p ./3dresunet_exp_nncf/data_dir/parameters.pkl -o 3dresunet_compressed.onnx -d ./3dresunet_exp_nncf/tcga-val-data-pre-ma-val.csv

import os
import argparse

import torch
import nncf  # Important - should be imported directly after torch

from nncf import NNCFConfig
from nncf.torch import create_compressed_model, register_default_init_args
from nncf.torch.checkpoint_loading import load_state

from readConfig import readConfig
from GANDLF.models import global_models_dict
from GANDLF.compute import train_network
from load_data_nncf import load_data

parser = argparse.ArgumentParser(
    description='Convert the NNCF PyTorch model to ONNX model.')
parser.add_argument('-i', '--nncf_model',
                    help='The NNCF compressed PyTorch model path.')
parser.add_argument('-o', '--onnx_model',
                    help='The exported ONNX model path.')
parser.add_argument('-d', '--data_csv',
                    help='The path to data csv containing path to images and labels.')
parser.add_argument('-c', '--nncf_config',
                    help="The NNCG config file")
parser.add_argument('-p', '--config_file', required=False, 
                    help='Config yaml file or the parameter file')
args = parser.parse_args()


parameters = readConfig(config_file=args.config_file)


print(parameters)

model = global_models_dict[parameters["model"]
                           ["architecture"]](parameters=parameters)

# # Provide data loaders for compression algorithm initialization, if necessary
init_loader, parameters = load_data(args.data_csv, parameters)
nncf_config = NNCFConfig.from_json(args.nncf_config) 
nncf_config = register_default_init_args(nncf_config, init_loader)

# load model
main_dict = torch.load(args.nncf_model, map_location=torch.device('cpu'))
model.load_state_dict(main_dict["model_state_dict"], strict=True)

# Create compressed model
compression_ctrl, compressed_model = create_compressed_model(model, nncf_config)

init_lr = 1e-4
compression_lr = init_lr / 10
optimizer = torch.optim.Adam(model.parameters(), lr=compression_lr)
train_network(compressed_model, init_loader, optimizer, parameters)

# # Direct PyTorch to OpenVINO IR conversion if necessary
# import mo_pytorch
# mo_pytorch.convert(compressed_model, input_shape=[1, 1, 128, 128, 128], model_name='resunet_qat')

compression_ctrl.export_model(args.onnx_model)

print("Onnx model is written to {0}.".format(args.onnx_model))