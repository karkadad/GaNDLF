import os, pathlib
import torch
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
import torchio

from GANDLF.utils import (
    get_date_time,
    get_filename_extension_sanitized,
    reverse_one_hot,
    resample_image,
)
from .step import step
from .loss_and_metric import get_loss_and_metrics

from openvino.inference_engine import IECore, IENetwork

from addict import Dict
from math import ceil

from compression.api import Metric, DataLoader
from compression.engines.ie_engine import IEEngine
from compression.graph import load_model, save_model
from compression.graph.model_utils import compress_model_weights
from compression.pipeline.initializer import create_pipeline
from compression.utils.logger import init_logger

class bcolors:
    """
    Just gives us some colors for the text
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class MyDataLoader(DataLoader):

    def __init__(self, config, params, valid_dataloader):

        super().__init__(config)
        # Initialize a few things
        total_epoch_valid_loss = 0
        total_epoch_valid_metric = {}
        average_epoch_valid_metric = {}

        self.params = params
        self.valid_dataloader = valid_dataloader
        self.features = []
        self.labels = []

        for metric in self.params["metrics"]:
            total_epoch_valid_metric[metric] = 0

        current_output_dir = self.params["output_dir"]  # this is in inference mode

        pathlib.Path(current_output_dir).mkdir(parents=True, exist_ok=True)

        idx = 0

        self.batch_size = 1

        for batch_idx, (subject) in enumerate(
            tqdm(self.valid_dataloader, desc="Looping over inference data")
        ):
            if self.params["verbose"]:
                print("== Current subject:", subject["subject_id"], flush=True)

            # ensure spacing is always present in params and is always subject-specific
            if "spacing" in subject:
                self.params["subject_spacing"] = subject["spacing"]
            else:
                self.params["subject_spacing"] = None

            # constructing a new dict because torchio.GridSampler requires torchio.Subject, which requires torchio.Image to be present in initial dict, which the loader does not provide
            subject_dict = {}
            label_ground_truth = None
            label_present = False
            # this is when we want the dataloader to pick up properties of GaNDLF's DataLoader, such as pre-processing and augmentations, if appropriate
            if "label" in subject:
                if subject["label"] != ["NA"]:
                    subject_dict["label"] = torchio.Image(
                        path=subject["label"]["path"],
                        type=torchio.LABEL,
                        tensor=subject["label"]["data"].squeeze(0),
                    )
                    label_present = True
                    label_ground_truth = subject_dict["label"]["data"]

            for key in self.params["channel_keys"]:
                subject_dict[key] = torchio.Image(
                    path=subject[key]["path"],
                    type=subject[key]["type"],
                    tensor=subject[key]["data"].squeeze(0),
                )

            grid_sampler = torchio.inference.GridSampler(
                torchio.Subject(subject_dict), self.params["patch_size"]
            )
            patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)

            current_patch = 0

            for patches_batch in patch_loader:
                if self.params["verbose"]:
                    print('!!!!!!!!!!!!!! Currently in seg quantize module !!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    print(
                        "=== Current patch:",
                        current_patch,
                        ", time : ",
                        get_date_time(),
                        ", location :",
                        patches_batch[torchio.LOCATION],
                        flush=True,
                    )
                current_patch += 1
                image = (
                    torch.cat(
                        [
                            patches_batch[key][torchio.DATA]
                            for key in self.params["channel_keys"]
                        ],
                        dim=1,
                    )
                    .float()
                    .to(self.params["device"])
                )
                if "value_keys" in self.params:
                    is_segmentation = False
                    label = label_ground_truth
                else:
                    label = patches_batch["label"][torchio.DATA]
                label = label.to(self.params["device"])
                if self.params["verbose"]:
                    print(
                        "=== Validation shapes : label:",
                        label.shape,
                        ", image:",
                        image.shape,
                        flush=True,
                    )

                self.features.append(image)
                self.labels.append(label)

                idx += 1


        self.items = np.arange(idx)

    @property
    def batch_num(self):
        return ceil(self.size / self.batch_size)

    @property
    def size(self):
        return self.items.shape[0]

    def __len__(self):
        return self.size

    def myPreprocess(self, image_filename, label_filename):
        """
        Custom code to preprocess input data
        For this example, we show how to process the brain tumor data.
        Change this to preprocess you data as necessary.
        """
        pass      

    def __getitem__(self, item):
        """
        Iterator to grab the data.
        If the data is too large to fit into memory, then
        you can have the item be a filename to load for the input
        and the label.

        In this example, we use the myPreprocess function above to
        do any custom preprocessing of the input.
        """

        # Load the iage and label files for this item
        image_features = self.features[self.items[item]]
        image_label = self.labels[self.items[item]]

        return (item, image_label), image_features


class MyMetric(Metric):
    def __init__(self, params):
        super().__init__()
        self.name = "custom Metric - Dice score"
        self._values = []
        self.round = 1
        self.params = params

    @property
    def value(self):
        """ Returns accuracy metric value for the last model output. """
        return {self.name: [self._values[-1]]}

    @property
    def avg_value(self):
        """ Returns accuracy metric value for all model outputs. """
        value = np.ravel(self._values).mean()
        print("Round #{}    Mean {} = {}".format(self.round, self.name, value))

        self.round += 1

        return {self.name: value}

    def update(self, outputs, labels):
        """ Updates prediction matches.

        Args:
            outputs: model output
            labels: annotations

        Put your post-processing code here.
        Put your custom metric code here.
        The metric gets appended to the list of metric values
        """

        def dice_score(pred, truth):
            """
            Sorensen Dice score
            Measure of the overlap between the prediction and ground truth masks
            """
            output = torch.from_numpy(pred)

            # one-hot encoding of 'label' will probably be needed for segmentation
            loss, metric_output = get_loss_and_metrics(output, truth, output, self.params)

            if len(output) > 1:
                output = output[0]

            print(metric_output)

            return metric_output


        metric_output = dice_score(outputs[0], labels[0])
        self._values.append(metric_output['dice'])

    def reset(self):
        """ Resets collected matches """
        self._values = []

    @property
    def higher_better(self):
        """Attribute whether the metric should be increased"""
        return True

    def get_attributes(self):
        return {self.name: {"direction": "higher-better", "type": ""}}

def validate_network(
    model, valid_dataloader, scheduler, params, epoch=0, mode="validation", input_name=None, output_name=None, frmwk=None
):
    model_config = Dict({
        "model_name": '3dresunet',
        "model": '/home/bduser/dkarkada/3dunet_karkadad_GaNDLF/GaNDLF/3dresunet_OV_FP32/3dresunet.xml',
        "weights": '/home/bduser/dkarkada/3dunet_karkadad_GaNDLF/GaNDLF/3dresunet_OV_FP32/3dresunet.bin'
    })
    dataset_config = {
    "images": "image",
    "labels": "label"
}

    engine_config = Dict({
        "device": "CPU",
        "stat_requests_number": 4,
        "eval_requests_number": 4
    })

    default_quantization_algorithm = [
        {
            "name": "DefaultQuantization",
            "params": {
                "target_device": "CPU",
                "preset": "performance",
                #"stat_subset_size": 10
            }
        }
    ]
        
    accuracy_aware_quantization = False

    model = load_model(model_config)

    # construct the model object (requires cpu since we're passing [padded] whole brains)
    #model = create_model_object(flplan=flplan, data_object=data, model_device=device)

    data_loader = MyDataLoader(dataset_config, params, valid_dataloader)
    metric = MyMetric(params)


    engine = IEEngine(engine_config, data_loader, metric)

    if accuracy_aware_quantization:
        # https://docs.openvinotoolkit.org/latest/_compression_algorithms_quantization_accuracy_aware_README.html
        print(bcolors.BOLD + "Accuracy-aware quantization method" + bcolors.ENDC)
        pipeline = create_pipeline(accuracy_aware_quantization_algorithm, engine)
    else:
        print(bcolors.BOLD + "Default quantization method" + bcolors.ENDC)
        pipeline = create_pipeline(default_quantization_algorithm, engine)


    metric_results_FP32 = pipeline.evaluate(model)

    compressed_model = pipeline.run(model)
    save_model(compressed_model, args.int8_directory)

    metric_results_INT8 = pipeline.evaluate(compressed_model)

    print(bcolors.BOLD + "\nFINAL RESULTS" + bcolors.ENDC)

    # print metric value
    if metric_results_FP32:
        for name, value in metric_results_FP32.items():
            print(bcolors.OKGREEN + "{: <27s} FP32: {}".format(name, value) + bcolors.ENDC)

    if metric_results_INT8:
        for name, value in metric_results_INT8.items():
            print(bcolors.OKBLUE + "{: <27s} INT8: {}".format(name, value) + bcolors.ENDC)


    print(bcolors.BOLD + "\nThe INT8 version of the model has been saved to the directory ".format(args.int8_directory) + \
        bcolors.HEADER + "{}\n".format(args.int8_directory) + bcolors.ENDC)

    return 0.0, metric_results_INT8
