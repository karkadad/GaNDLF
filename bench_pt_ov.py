# Usage example (working): python benchmark_pt_ov.py -m 'resunet' -md './infer_models' \
#                          -ptm 3dresunet_pt -ovm 3dresunet_ov \
#                          -p ./3dunet_exp/data_dir/parameters.pkl \
#                          -d ./3dunet_exp/tcga-val-data-pre-ma-test.csv \
#                          -o ./3dunet_exp/test_data_dir -v False
'''

python bench_pt_ov.py \
-m 'resunet' \
-d /Share/ravi/upenn/data/sidd-tcga/val_data_sidd-tcga_4mod_244/data_processed_4.csv \
-p ./configs/train_config.yaml \
-o ./output_benchmarking/

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

parser = argparse.ArgumentParser(
    description='Benchmarking for the 3D ResUnet models')
parser.add_argument('-m', '--model_name',
                    help='The model name', default='resunet')
# parser.add_argument('-md', '--model_dir',
#                     help='The PyTorch or OpenVINO model root directory path.')
# parser.add_argument('-ptm', '--pytorch_model',
#                     help='The PyTorch model path.')
# parser.add_argument('-ovm', '--ov_model',
#                     help='The OpenVINO model path.')
parser.add_argument('-d', '--data_csv',
                    help='The path to data csv containing path to images and labels.')
parser.add_argument('-p', '--parameters_file', required=False,
                    help='Config yaml file or the parameter file')
parser.add_argument('-o', '--output_dir', required=False,
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

# def load_ov_model(path, is_onnx=False):
#     ie = IECore()
#     print(f'OV IR path:{path}')

#     if is_onnx:
#         net = ie.read_network(model=path.with_suffix(".onnx"))
#     else:
#         net = ie.read_network(model=path)

#     input_blob = next(iter(net.input_info))
#     out_blob = next(iter(net.outputs))

#     config = {}
#     config['CPU_THROUGHPUT_STREAMS'] = str('1')

#     exec_net = ie.load_network(network=net, device_name="CPU", config=config)
#     return exec_net, input_blob, out_blob

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

    # Load model parameters
    # with open(args.parameters_file, 'rb') as f:
    #     parameters = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    parameters = readConfig(config_file=args.parameters_file)
    parameters['train_mode'] = False
    parameters['device'] = device

    model = global_models_dict[parameters["model"]
                           ["architecture"]](parameters=parameters)

    val_dataloader, parameters = load_data(args.data_csv, parameters, args.output_dir, train_mode=True)

    # ##Original PyTorch Model
    # #orig_pth = os.path.join(args.model_dir, args.pytorch_model, args.model_name +"_best.pth.tar")
    # orig_pth = "/home/rpanchum/upenn/GaNDLF-dk-nncf/models/0/resunet_best.pth.tar"
    # model = load_torch_model(orig_pth, model)
    # benchmark_pt_model(model, parameters, val_dataloader)

    parameters = readConfig(config_file="./configs/train_config_ov.yaml")
    parameters['train_mode'] = False
    parameters['device'] = device
    parameters['output_dir'] = args.output_dir
    parameters['channel_keys'] = ['1']
    parameters['problem_type'] = 'segmentation'
    parameters["weights"] = None
    parameters["class_weight"] = None
    parameters["label_keys"] = ['label']

    ## Original OpenVINO FP32 Model
    fp32_ir_path = "./models/0/resunet_best.xml"
    exec_net, input_blob, output_blob = load_ov_model(fp32_ir_path)
    parameters["model"]["IO"] = [input_blob, output_blob]
    benchmark_ov_model(exec_net, parameters, val_dataloader, input_blob, output_blob, model_type='FP32')

    # # ###OpenVINO POT INT8 Model
    int8_ir_path = "./ov-irs/int8/resunet_best.xml"
    exec_net, input_blob, output_blob = load_ov_model(int8_ir_path)
    parameters["model"]["IO"] = [input_blob, output_blob]
    benchmark_ov_model(exec_net, parameters, val_dataloader, input_blob, output_blob, model_type='OV POT INT8')

    # # ###OpenVINO + NNCF QAT model
    int8_ir_path = "./ov-irs/nncf-int8-pruned/resunet_best.xml"
    exec_net, input_blob, output_blob = load_ov_model(int8_ir_path)
    parameters["model"]["IO"] = [input_blob, output_blob]
    benchmark_ov_model(exec_net, parameters, val_dataloader, input_blob, output_blob, model_type='NNCF QAT + OV INT8')

    # # # ###OpenVINO + NNCF Filter Pruned model
    int8_ir_path = "./ov-irs/fp32/prun5060-pruned/resunet_best.xml"
    exec_net, input_blob, output_blob = load_ov_model(int8_ir_path)
    parameters["model"]["IO"] = [input_blob, output_blob]
    benchmark_ov_model(exec_net, parameters, val_dataloader, input_blob, output_blob, model_type='NNCF Filter Pruned + OV FP32 --> Pruning level 0.6')

    # # # ###OpenVINO + NNCF QAT + Filter Pruned + KD model
    # #int8_ir_path = Path(os.path.join(args.model_dir, args.ov_model + '/QAT_FilterPruned_KD'))
    # exec_net, input_blob, output_blob = load_ov_model(Path(int8_ir_path / (args.model_name)))
    # benchmark_ov_model(exec_net, parameters, val_dataloader, input_blob, output_blob, model_type='NNCF QAT + Filter Pruned + KD')

    # # # ###OpenVINO + NNCF QAT + Filter Pruned --> Pruning level 0.6 + KD model
    # int8_ir_path = Path(os.path.join(args.model_dir, args.ov_model + '/QAT_FilterPruned_60_KD'))
    # exec_net, input_blob, output_blob = load_ov_model(Path(int8_ir_path / (args.model_name)))
    # benchmark_ov_model(exec_net, parameters, val_dataloader, input_blob, output_blob, model_type='NNCF QAT + Filter Pruned --> Pruning level 0.6 + KD')

    # # # ###OpenVINO + NNCF KD + Filter Pruned model
    # int8_ir_path = Path(os.path.join(args.model_dir, args.ov_model + '/KD_FilterPruned'))
    # exec_net, input_blob, output_blob = load_ov_model(Path(int8_ir_path / (args.model_name)))
    # benchmark_ov_model(exec_net, parameters, val_dataloader, input_blob, output_blob, model_type='NNCF KD + Filter Pruned')

    # # # ###OpenVINO + NNCF Filter Pruned model
    # int8_ir_path = Path(os.path.join(args.model_dir, args.ov_model + '/OfflineFilterPruned4050'))
    # exec_net, input_blob, output_blob = load_ov_model(Path(int8_ir_path / (args.model_name)))
    # benchmark_ov_model(exec_net, parameters, val_dataloader, input_blob, output_blob, model_type='NNCF Filter Pruned + OV FP32 --> Pruning level 0.5')



    # # # ###OpenVINO + NNCF Filter Pruned model
    # int8_ir_path = Path(os.path.join(args.model_dir, args.ov_model + '/OfflineFilterPruned5070'))
    # exec_net, input_blob, output_blob = load_ov_model(Path(int8_ir_path / (args.model_name)))
    # benchmark_ov_model(exec_net, parameters, val_dataloader, input_blob, output_blob, model_type='NNCF Filter Pruned + OV FP32 --> Pruning level 0.7')

    # # # ###OpenVINO + NNCF Filter Pruned model
    # int8_ir_path = Path(os.path.join(args.model_dir, args.ov_model + '/OfflineFilterPruned5080'))
    # exec_net, input_blob, output_blob = load_ov_model(Path(int8_ir_path / (args.model_name)))
    # benchmark_ov_model(exec_net, parameters, val_dataloader, input_blob, output_blob, model_type='NNCF Filter Pruned + OV FP32 --> Pruning level 0.8')

    # # # ###OpenVINO + NNCF Filter Pruned model
    # int8_ir_path = Path(os.path.join(args.model_dir, args.ov_model + '/OfflineFilterPruned5090'))
    # exec_net, input_blob, output_blob = load_ov_model(Path(int8_ir_path / (args.model_name)))
    # benchmark_ov_model(exec_net, parameters, val_dataloader, input_blob, output_blob, model_type='NNCF Filter Pruned + OV FP32 --> Pruning level 0.9')

if __name__ == '__main__':
    main()
