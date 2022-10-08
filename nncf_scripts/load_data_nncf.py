import os
import torch
from GANDLF.data.ImagesFromDataFrame import ImagesFromDataFrame
from GANDLF.utils import (
    populate_header_in_parameters,
    populate_channel_keys_in_params,
    parseTrainingCSV,
)

def load_data(train_data_csv, valid_data_csv, parameters, train_mode=True):
    train_data_full, train_headers = parseTrainingCSV(train_data_csv, train=train_mode)
    valid_data_full, valid_headers = parseTrainingCSV(valid_data_csv, train=train_mode)
    parameters = populate_header_in_parameters(parameters, train_headers)
    parameters = populate_header_in_parameters(parameters, valid_headers)

    training_data_for_torch = ImagesFromDataFrame(
        train_data_full, parameters, train=True
    )
    validation_data_for_torch = ImagesFromDataFrame(
        valid_data_full, parameters, train=False
    )
    parameters = populate_channel_keys_in_params(training_data_for_torch, parameters)
    parameters = populate_channel_keys_in_params(validation_data_for_torch, parameters)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    cwd = os.getcwd()

    parameters["device"] = device
    parameters["weights"], parameters["class_weights"] = None, None
    parameters["output_dir"] = f"{cwd}/nncf_scripts/nncf_accaware_outdir"
    # print(parameters)

    train_dataloader = torch.utils.data.DataLoader(
        training_data_for_torch,
        batch_size=parameters["batch_size"],
        shuffle=True,
        pin_memory=True,  # params["pin_memory_dataloader"], # this is going OOM if True - needs investigation
    )
    validation_dataloader = torch.utils.data.DataLoader(
        validation_data_for_torch,
        batch_size=1,
        shuffle=False,
        pin_memory=True,  # params["pin_memory_dataloader"], # this is going OOM if True - needs investigation
    )

    return train_dataloader, validation_dataloader, parameters
