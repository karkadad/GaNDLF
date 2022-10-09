import onnx
import torch

from readConfig_ma import readConfig
from GANDLF.models import global_models_dict
from GANDLF.compute import validate_network

import torch.onnx
from torch.autograd import Variable
import numpy as np
from onnx import version_converter
from time import time

print('torch.__version__', torch.__version__)
print('onnx.__version__', onnx.__version__)

config_file_path = "../configs/train_config.yaml"
parameters = readConfig(config_file=config_file_path)

model_path = "/home/rpanchum/upenn/GaNDLF-dk-nncf/nncf_scripts/nncf_accaware_outdir/accuracy_aware_training/2022-09-11__18-56-52/acc_aware_checkpoint_best.pth"

model = global_models_dict[parameters["model"]["architecture"]](
    parameters=parameters
)


model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
model.eval()

dummy_input = Variable(torch.randn(1, 1, 240, 240, 144))
start = time()
torch_outputs = model(dummy_input)[-1]
end = time()
print('Pytorch model inference time:{}'.format((end-start)*1000))

onnx_model_name = "./nncf_compressed_models/nncf_quantized_finetuned_3dresunet_siddVALData_ep5.onnx"
# Convert to ONNX
torch.onnx.export(model, dummy_input, onnx_model_name, verbose=True, opset_version=11, training=False)


# Check model
model = onnx.load(onnx_model_name)
# converted_model = version_converter.convert_version(model, 9)
onnx.checker.check_model(model)
# print(onnx.helper.printable_graph(model.graph))

