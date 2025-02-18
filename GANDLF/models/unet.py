# -*- coding: utf-8 -*-
"""
Implementation of UNet
"""

from GANDLF.models.seg_modules.DownsamplingModule import DownsamplingModule
from GANDLF.models.seg_modules.EncodingModule import EncodingModule
from GANDLF.models.seg_modules.DecodingModule import DecodingModule
from GANDLF.models.seg_modules.UpsamplingModule import UpsamplingModule
from GANDLF.models.seg_modules.in_conv import in_conv
from GANDLF.models.seg_modules.out_conv import out_conv
from .modelBase import ModelBase
import sys
from GANDLF.utils.generic import checkPatchDivisibility


class unet(ModelBase):
    """
    This is the standard U-Net architecture : https://arxiv.org/pdf/1606.06650.pdf. The 'residualConnections' flag controls residual connections, the
    Downsampling, Encoding, Decoding modules are defined in the seg_modules file. These smaller modules are basically defined by 2 parameters, the input
    channels (filters) and the output channels (filters), and some other hyperparameters, which remain constant all the modules. For more details on the
    smaller modules please have a look at the seg_modules file.
    """

    def __init__(
        self,
        parameters: dict,
        residualConnections=False,
    ):
        self.network_kwargs = {"res": residualConnections}
        super(unet, self).__init__(parameters)

        if not (checkPatchDivisibility(parameters["patch_size"])):
            sys.exit(
                "The patch size is not divisible by 16, which is required for",
                parameters["model"]["architecture"],
            )

        self.ins = in_conv(
            input_channels=self.n_channels,
            output_channels=self.base_filters,
            conv=self.Conv,
            dropout=self.Dropout,
            norm=self.Norm,
            network_kwargs=self.network_kwargs,
        )
        self.ds_0 = DownsamplingModule(
            input_channels=self.base_filters,
            output_channels=self.base_filters * 2,
            conv=self.Conv,
            norm=self.Norm,
        )
        self.en_1 = EncodingModule(
            input_channels=self.base_filters * 2,
            output_channels=self.base_filters * 2,
            conv=self.Conv,
            dropout=self.Dropout,
            norm=self.Norm,
            network_kwargs=self.network_kwargs,
        )
        self.ds_1 = DownsamplingModule(
            input_channels=self.base_filters * 2,
            output_channels=self.base_filters * 4,
            conv=self.Conv,
            norm=self.Norm,
        )
        self.en_2 = EncodingModule(
            input_channels=self.base_filters * 4,
            output_channels=self.base_filters * 4,
            conv=self.Conv,
            dropout=self.Dropout,
            norm=self.Norm,
            network_kwargs=self.network_kwargs,
        )
        self.ds_2 = DownsamplingModule(
            input_channels=self.base_filters * 4,
            output_channels=self.base_filters * 8,
            conv=self.Conv,
            norm=self.Norm,
        )
        self.en_3 = EncodingModule(
            input_channels=self.base_filters * 8,
            output_channels=self.base_filters * 8,
            conv=self.Conv,
            dropout=self.Dropout,
            norm=self.Norm,
            network_kwargs=self.network_kwargs,
        )
        self.ds_3 = DownsamplingModule(
            input_channels=self.base_filters * 8,
            output_channels=self.base_filters * 16,
            conv=self.Conv,
            norm=self.Norm,
        )
        self.en_4 = EncodingModule(
            input_channels=self.base_filters * 16,
            output_channels=self.base_filters * 16,
            conv=self.Conv,
            dropout=self.Dropout,
            norm=self.Norm,
            network_kwargs=self.network_kwargs,
        )
        self.us_3 = UpsamplingModule(
            input_channels=self.base_filters * 16,
            output_channels=self.base_filters * 8,
            conv=self.Conv,
            interpolation_mode=self.linear_interpolation_mode,
        )
        self.de_3 = DecodingModule(
            input_channels=self.base_filters * 16,
            output_channels=self.base_filters * 8,
            conv=self.Conv,
            norm=self.Norm,
            network_kwargs=self.network_kwargs,
        )
        self.us_2 = UpsamplingModule(
            input_channels=self.base_filters * 8,
            output_channels=self.base_filters * 4,
            conv=self.Conv,
            interpolation_mode=self.linear_interpolation_mode,
        )
        self.de_2 = DecodingModule(
            input_channels=self.base_filters * 8,
            output_channels=self.base_filters * 4,
            conv=self.Conv,
            norm=self.Norm,
            network_kwargs=self.network_kwargs,
        )
        self.us_1 = UpsamplingModule(
            input_channels=self.base_filters * 4,
            output_channels=self.base_filters * 2,
            conv=self.Conv,
            interpolation_mode=self.linear_interpolation_mode,
        )
        self.de_1 = DecodingModule(
            input_channels=self.base_filters * 4,
            output_channels=self.base_filters * 2,
            conv=self.Conv,
            norm=self.Norm,
            network_kwargs=self.network_kwargs,
        )
        self.us_0 = UpsamplingModule(
            input_channels=self.base_filters * 2,
            output_channels=self.base_filters,
            conv=self.Conv,
            interpolation_mode=self.linear_interpolation_mode,
        )
        self.de_0 = DecodingModule(
            input_channels=self.base_filters * 2,
            output_channels=self.base_filters * 2,
            conv=self.Conv,
            norm=self.Norm,
            network_kwargs=self.network_kwargs,
        )
        self.out = out_conv(
            input_channels=self.base_filters * 2,
            output_channels=self.n_classes,
            conv=self.Conv,
            norm=self.Norm,
            network_kwargs=self.network_kwargs,
            final_convolution_layer=self.final_convolution_layer,
            sigmoid_input_multiplier=self.sigmoid_input_multiplier,
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor
            Should be a 5D Tensor as [batch_size, channels, x_dims, y_dims, z_dims].

        Returns
        -------
        x : Tensor
            Returns a 5D Output Tensor as [batch_size, n_classes, x_dims, y_dims, z_dims].

        """
        x1 = self.ins(x)
        x2 = self.ds_0(x1)
        x2 = self.en_1(x2)
        x3 = self.ds_1(x2)
        x3 = self.en_2(x3)
        x4 = self.ds_2(x3)
        x4 = self.en_3(x4)
        x5 = self.ds_3(x4)
        x5 = self.en_4(x5)

        x = self.us_3(x5)
        x = self.de_3(x, x4)
        x = self.us_2(x)
        x = self.de_2(x, x3)
        x = self.us_1(x)
        x = self.de_1(x, x2)
        x = self.us_0(x)
        x = self.de_0(x, x1)
        x = self.out(x)
        return x


class resunet(unet):
    """
    This is the standard U-Net architecture with residual connections : https://arxiv.org/pdf/1606.06650.pdf.
    The 'residualConnections' flag controls residual connections The Downsampling, Encoding, Decoding modules are defined in the seg_modules file.
    These smaller modules are basically defined by 2 parameters, the input channels (filters) and the output channels (filters),
    and some other hyperparameters, which remain constant all the modules. For more details on the smaller modules please have a look at the seg_modules file.
    """

    def __init__(self, parameters: dict):
        super(resunet, self).__init__(parameters, residualConnections=True)
