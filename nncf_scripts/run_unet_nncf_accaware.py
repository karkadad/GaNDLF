# Usage example: python run_unet_nncf_accaware.py -i ./3dresunet_exp_nncf/data_dir/resunet_best.pth.tar -c ./nncf_configs/quantization_pruning_kd.json -p ./3dresunet_exp_nncf/data_dir/parameters.pkl -o 3dresunet_compressed.onnx -dt ./3dresunet_exp_nncf/tcga-val-data-pre-ma-val.csv -dv ./3dresunet_exp_nncf/tcga-val-data-pre-ma-test.csv

import os
import argparse

import torch
import nncf  # Important - should be imported directly after torch

from nncf import NNCFConfig
from nncf.torch import create_compressed_model, register_default_init_args
from nncf.torch.checkpoint_loading import load_state
from nncf.common.accuracy_aware_training import create_accuracy_aware_training_loop

from readConfig_ma import readConfig
from GANDLF.models import global_models_dict
from GANDLF.compute import nncf_train_network
from GANDLF.compute import validate_network
from load_data_nncf import load_data

parser = argparse.ArgumentParser(
    description='Convert the NNCF PyTorch model to ONNX model.')
parser.add_argument('-i', '--nncf_model',
                    help='The NNCF compressed PyTorch model path.')
parser.add_argument('-o', '--onnx_model',
                    help='The exported ONNX model path.')
parser.add_argument('-dt', '--train_data_csv',
                    help='The path to train data csv containing path to images and labels.')
parser.add_argument('-dv', '--valid_data_csv',
                    help='The path to validation data csv containing path to images and labels.')
parser.add_argument('-c', '--nncf_config',
                    help="The NNCG config file")
parser.add_argument('-p', '--config_file', required=False,
                    help='Config yaml file or the parameter file')
args = parser.parse_args()


parameters = readConfig(config_file=args.config_file)


print(parameters)

def main():
    global parameters
    model = global_models_dict[parameters["model"]
                               ["architecture"]](parameters=parameters)

    # # Provide data loaders for compression algorithm initialization, if necessary
    train_dataloader, validation_dataloader, parameters = load_data(args.train_data_csv, args.valid_data_csv, parameters)
    nncf_config = NNCFConfig.from_json(args.nncf_config)

    def eval_fn(model, epoch=None):
        return validate_network(model, validation_dataloader, None, parameters)

    def configure_optimizers_fn():
        init_lr = 1e-4
        compression_lr = init_lr / 10
        optimizer = torch.optim.Adam(model.parameters(), lr=compression_lr)
        return optimizer, None

    # load model
    main_dict = torch.load(args.nncf_model, map_location=torch.device('cpu'))
    model.load_state_dict(main_dict["model_state_dict"], strict=False)

    optimizer, _ = configure_optimizers_fn()

    nncf_config = register_default_init_args(nncf_config, train_dataloader, model_eval_fn=eval_fn)

    # Create compressed model
    compression_ctrl, compressed_model = create_compressed_model(model, nncf_config)

    def validate_fn(compressed_model, epoch=None):
        return validate_network(compressed_model, validation_dataloader, None, parameters)

    # training function that trains the model for one epoch (full training dataset pass)
    # it is assumed that all the NNCF-related methods are properly called inside of
    # this function (like e.g. the step and epoch_step methods of the compression scheduler)
    def train_epoch_fn(compression_ctrl, compressed_model, optimizer, **kwargs):
        return nncf_train_network(compressed_model, train_dataloader, optimizer, compression_ctrl, parameters)

    acc_aware_training_loop = create_accuracy_aware_training_loop(nncf_config, compression_ctrl)
    model = acc_aware_training_loop.run(compressed_model,
                                        train_epoch_fn=train_epoch_fn,
                                        validate_fn=validate_fn,
                                        configure_optimizers_fn=configure_optimizers_fn,
                                        log_dir='nncf_accaware_outdir')

    # Direct PyTorch to OpenVINO IR conversion if necessary
    #import mo_pytorch
    #mo_pytorch.convert(compressed_model, input_shape=[1, 1, 128, 128, 128], model_name='resunet_qat')

    # save_format = 'onnx_{}'.format(args.opset_version)
    save_format = 'onnx_11'
   # Export to ONNX
    compression_ctrl.export_model(args.onnx_model, save_format=save_format)

    #compression_ctrl.export_model(args.onnx_model)

    checkpoint = {
        'model_state_dict': compressed_model.state_dict(),
        'compression_state': compression_ctrl.get_compression_state()
    }

    torch.save(checkpoint, 'nncf_accaware_outdir/resunet_nncf_accaware_model.pth')

    print("Onnx model is written to {0}.".format(args.onnx_model))

if __name__ == '__main__':
    main()