from pathlib import Path
import requests, zipfile, io, os, csv, random, copy, shutil, sys, yaml, torch, pytest
import SimpleITK as sitk
import numpy as np
import pandas as pd

from pydicom.data import get_testdata_file

from GANDLF.data.ImagesFromDataFrame import ImagesFromDataFrame
from GANDLF.utils import *
from GANDLF.data.preprocessing import global_preprocessing_dict
from GANDLF.data.augmentation import global_augs_dict
from GANDLF.parseConfig import parseConfig
from GANDLF.training_manager import TrainingManager
from GANDLF.inference_manager import InferenceManager
from GANDLF.cli import main_run, preprocess_and_save, patch_extraction
from GANDLF.schedulers import global_schedulers_dict
from GANDLF.optimizers import global_optimizer_dict
from GANDLF.models import global_models_dict
from GANDLF.data.post_process import torch_morphological, fill_holes, get_mapped_label
from GANDLF.anonymize import run_anonymizer

device = "cpu"
## global defines
# pre-defined segmentation model types for testing
all_models_segmentation = [
    "lightunet",
    "lightunet_multilayer",
    "unet",
    "unet_multilayer",
    "deep_resunet",
    "fcn",
    "uinc",
    "msdnet",
]
# pre-defined regression/classification model types for testing
all_models_regression = [
    "densenet121",
    "vgg16",
    "resnet18",
    "resnet50",
    "efficientnetb0",
]
# pre-defined regression/classification model types for testing
all_models_classification = [
    "imagenet_vgg11",
    "imagenet_vgg11_bn",
    "imagenet_vgg13",
    "imagenet_vgg13_bn",
    "imagenet_vgg16",
    "imagenet_vgg16_bn",
    "imagenet_vgg19",
    "imagenet_vgg19_bn",
    "resnet18",
]

all_clip_modes = ["norm", "value", "agc"]
all_norm_type = ["batch", "instance"]

all_model_type = ["torch", "openvino"]

patch_size = {"2D": [128, 128, 1], "3D": [32, 32, 32]}

testingDir = Path(__file__).parent.absolute().__str__()
baseConfigDir = os.path.join(testingDir, os.pardir, "samples")
inputDir = os.path.join(testingDir, "data")
outputDir = os.path.join(testingDir, "data_output")
Path(outputDir).mkdir(parents=True, exist_ok=True)


"""
steps to follow to write tests:
[x] download sample data
[x] construct the training csv
[x] for each dir (application type) and sub-dir (image dimension), run training for a single epoch on cpu
  [x] separate tests for 2D and 3D segmentation
  [x] read default parameters from yaml config
  [x] for each type, iterate through all available segmentation model archs
  [x] call training manager with default parameters + current segmentation model arch
[ ] for each dir (application type) and sub-dir (image dimension), run inference for a single trained model per testing/validation split for a single subject on cpu
"""


def test_train_segmentation_rad_3d_quantization(device):
    print("01: Starting 3D Rad segmentation post training optimizations tests")
    # read and parse csv
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_segmentation_quantization.yaml", version_check_flag=False
    )
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_3d_rad_segmentation.csv"
    )
    parameters["modality"] = "rad"
    parameters["patch_size"] = patch_size["3D"]
    parameters["model"]["dimension"] = 3
    parameters["model"]["class_list"] = [0, 1]
    parameters["model"]["final_layer"] = "softmax"
    parameters["model"]["amp"] = True
    parameters["in_memory"] = True
    parameters["model"]["num_channels"] = len(parameters["headers"]["channelHeaders"])
    parameters["model"]["onnx_export"] = True
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    # loop through selected models and train for single epoch
    parameters["model"]["architecture"] = "unet"
    parameters["nested_training"]["testing"] = -5
    parameters["nested_training"]["validation"] = -5

    sanitize_outputDir()
    TrainingManager(
        dataframe=training_data,
        outputDir=outputDir,
        parameters=parameters,
        device=device,
        resume=False,
        reset=True,
    )

    parameters["model"]["type"] = "openvino"
    parameters["model"]["data_type"] = "INT8"
    parameters["model"]["optimization_mode"] = "post_training_quantization"
    parameters["model"]["quantization_mode"] = "DefaultQuantization"
    parameters["output_dir"] = outputDir  # this is in inference mode
    InferenceManager(
        dataframe=training_data,
        outputDir=outputDir,
        parameters=parameters,
        device=device,
    )

    print("passed")
