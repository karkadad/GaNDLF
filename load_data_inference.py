import torch
from GANDLF.data.ImagesFromDataFrame import ImagesFromDataFrame
from GANDLF.utils import populate_header_in_parameters, populate_channel_keys_in_params, parseTrainingCSV

def load_data(data_csv, parameters, out_dir, train_mode=True):
    data_full, headers = parseTrainingCSV(data_csv, train=train_mode)
    parameters = populate_header_in_parameters(parameters, headers)

    inferenceDataForTorch = ImagesFromDataFrame(data_full, parameters, train=False)
    parameters = populate_channel_keys_in_params(inferenceDataForTorch, parameters)
    parameters['device'] = 'cpu'
    parameters["model"]["amp"] = None
    parameters["weights"], parameters["class_weights"] = None, None
    parameters["output_dir"] = out_dir
    print(parameters)


    infer_dataloader = torch.utils.data.DataLoader(
        inferenceDataForTorch,
        batch_size=parameters["batch_size"],
        shuffle=False,
        pin_memory=True  # params["pin_memory_dataloader"], # this is going OOM if True - needs investigation
    )

    return infer_dataloader, parameters