import torch
from torch import nn
import numpy as np
from collections import OrderedDict

from .base import Concatenate, get_padding, skip_none_sequential

# Adapted from https://github.com/imagirom/ConfNets
class EncoderDecoderSkeleton(nn.Module):
    """
    Base class for Networks with Encoder Decoder Structure, such as UNet.
    To use, inherit from this and implement a selection of the construct_* methods.
    To add side-outputs, use a wrapper
    """
    def __init__(self, depth):
        super(EncoderDecoderSkeleton, self).__init__()
        self.depth = depth
        # construct all the layers
        self.initial_module = self.construct_input_module()
        self.encoder_modules = nn.ModuleList(
            [self.construct_encoder_module(i) for i in range(depth)])
        self.skip_modules = nn.ModuleList(
            [self.construct_skip_module(i) for i in range(depth)])
        self.downsampling_modules = nn.ModuleList(
            [self.construct_downsampling_module(i) for i in range(depth)])
        self.upsampling_modules = nn.ModuleList(
            [self.construct_upsampling_module(i) for i in range(depth)])
        self.decoder_modules = nn.ModuleList(
            [self.construct_decoder_module(i) for i in range(depth)])
        self.merge_modules = nn.ModuleList(
            [self.construct_merge_module(i) for i in range(depth)])
        self.base_module = self.construct_base_module()
        self.final_module = self.construct_output_module()

    def forward(self, input):
        encoded_states = []
        current = self.initial_module(input)
        for encode, downsample in zip(self.encoder_modules, self.downsampling_modules):
            current = encode(current)
            encoded_states.append(current)
            current = downsample(current)
        current = self.base_module(current)
        for encoded_state, upsample, skip, merge, decode in reversed(list(zip(
                encoded_states, self.upsampling_modules, self.skip_modules, self.merge_modules, self.decoder_modules))):
            current = upsample(current)
            encoded_state = skip(encoded_state)
            current = merge(current, encoded_state)
            current = decode(current)
        current = self.final_module(current)
        return current

    def construct_input_module(self):
        return nn.Identity()

    def construct_encoder_module(self, depth):
        return nn.Identity()

    def construct_decoder_module(self, depth):
        return self.construct_encoder_module(depth)

    def construct_downsampling_module(self, depth):
        return nn.Identity()

    def construct_upsampling_module(self, depth):
        return nn.Identity()

    def construct_skip_module(self, depth):
        return nn.Identity()

    def construct_merge_module(self, depth):
        return Concatenate()

    def construct_base_module(self):
        return nn.Identity()

    def construct_output_module(self):
        return nn.Identity()


class UNet(EncoderDecoderSkeleton):
    def __init__(self,
                 depth,
                 in_channels,
                 out_channels,
                 fmaps,
                 dim=2,
                 scale_factor=2,
                 norm_type=None,
                 activation='ReLU',
                 final_activation=None):
        """
        :param depth: Depth of the UNet model (how many downscaling levels in the hierarchy)

        :param in_channels: Number of input channels

        :param out_channels: Number of output channels

        :param fmaps: list or tuple indicating the number of channels / feature maps used for the convolutional
                layers in the encoder and decoder modules of the UNet. The length of the list should be (depth + 1).
                For example, if depth 2 and `fmaps` = (16, 32, 64) then the encoder/decoder modules at level=0 (i.e.
                at the highest resolution) will have 16 channels, the encoder/decoder modules at level=1 will have
                32 channels and the base module will have 64 channels.

        :param dim: dimension of the input. 1 for 1D vectors, 2 for images, 3 for volumetric images

        :param scale_factor: How much to downscale at each level of the UNet. It can be a in integer or a list
                with length `depth` (specifying the downscaling factor for each level).

        :param norm_type: Class of a normalization layer from torch.nn, for example nn.BatchNorm2d

                Remark about torch.nn.GroupNorm: in this case, you first need to specify the argument `num_groups`,
                making sure that fmaps % num_groups == 0 for all the `fmaps` in every level of the UNet.
                For example, for `num_groups`= 16, you can use the following code:
                    >>>  from functools import partial
                    >>>  num_groups = 16
                    >>>  group_norm = partial(torch.nn.GroupNorm, 16)
                    >>>  unet_model = UNet(..., norm_type=group_norm, ...)

        :param activation: Class of an activation layer from torch.nn, for example torch.nn.ReLU

        :param final_activation: Activation layer to apply just before of the output, for example torch.nn.Softmax
                It can also be None and in that case no activation is applied (it can be useful if you use a loss
                like `torch.nn.BCEWithLogitsLoss`, which combines a Sigmoid layer and the BCELoss
                in one single class to avoid outputs with infinite values.
        """
        self.dim = dim

        # Get conv_type
        conv_type = f'Conv{self.dim}d'
        conv_type = getattr(nn, conv_type)
        self.conv_type = conv_type

        # Validate norm_type
        if norm_type is None:
            norm_type = lambda in_channels: None
        assert callable(norm_type), \
                f'norm_type has to be callable or None'
        self.norm_type = norm_type

        # Validate activation
        if isinstance(activation, nn.Module):
            activation_module = activation
            activation = lambda: activation_module
        assert callable(activation), \
            f'activation has to be nn.Module or callable.'
        self.activation = activation

        # Build final activation:
        if final_activation is not None:
            assert callable(final_activation), \
                'final activation has to be None or callable'
        self.final_activation = final_activation

        # shorthand dictionary for conv_type, norm_type and activation, e.g. for the initialization of blocks
        self.conv_norm_act_dict = dict(conv_type=self.conv_type, norm_type=self.norm_type,
                                       activation=self.activation)

        # parse scale factor
        if isinstance(scale_factor, int):
            scale_factor = [scale_factor, ] * depth
        scale_factors = scale_factor
        normalized_factors = []
        for scale_factor in scale_factors:
            assert isinstance(scale_factor, (int, list, tuple))
            if isinstance(scale_factor, int):
                scale_factor = self.dim * [scale_factor, ]
            assert len(scale_factor) == self.dim
            normalized_factors.append(tuple(scale_factor))
        self.scale_factors = tuple(normalized_factors)

        # compute input size divisibility constraints
        divisibility_constraint = np.ones(len(self.scale_factors[0]))
        for scale_factor in self.scale_factors:
            divisibility_constraint *= np.array(scale_factor)
        self.divisibility_constraint = list(divisibility_constraint.astype(int))

        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert isinstance(fmaps, (list, tuple))
        self.fmaps = fmaps
        assert len(self.fmaps) == self.depth + 1

        self.merged_fmaps = [2 * n for n in self.fmaps]

        # Build all the layers:
        super(UNet, self).__init__(depth)

        # delete attributes that are only relevant for construction and might lead to errors when model is saved
        del self.conv_type
        del self.norm_type
        del self.activation
        del self.conv_norm_act_dict

    def forward(self, input_):
        input_dim = len(input_.shape)
        assert all(input_.shape[-i] % self.divisibility_constraint[-i] == 0 for i in range(1, input_dim - 1)), \
            f'Input shape {input_.shape[2:]} not suited for downsampling with factors {self.scale_factors}.' \
            f'Lengths of spatial axes must be multiples of {self.divisibility_constraint}.'
        return super(UNet, self).forward(input_)

    def construct_layer(self, f_in, f_out, kernel_size=3):
        return skip_none_sequential(OrderedDict([
            ('conv', self.conv_type(f_in, f_out, kernel_size=kernel_size, padding=get_padding(kernel_size))),
            ('norm', self.norm_type(f_out)),
            ('activation', self.activation())
        ]))

    def construct_encoder_module(self, depth):
        f_in = self.in_channels if depth == 0 else self.fmaps[depth - 1]
        f_out = self.fmaps[depth]
        return nn.Sequential(
            self.construct_layer(f_in, f_out),
            self.construct_layer(f_out, f_out)
        )

    def construct_decoder_module(self, depth):
        f_in = self.merged_fmaps[depth]
        f_intermediate = self.fmaps[depth]
        # do not reduce to f_out yet - this is done in the output module
        f_out = f_intermediate if depth == 0 else self.fmaps[depth - 1]
        return nn.Sequential(
            self.construct_layer(f_in, f_intermediate),
            self.construct_layer(f_intermediate, f_out)
        )

    def construct_base_module(self):
        f_in = self.fmaps[self.depth - 1]
        f_intermediate = self.fmaps[self.depth]
        f_out = self.fmaps[self.depth - 1]
        return nn.Sequential(
            self.construct_layer(f_in, f_intermediate),
            self.construct_layer(f_intermediate, f_out)
        )

    def construct_merge_module(self, depth):
        return Concatenate()

    def construct_output_module(self):
        if self.final_activation is not None:
            return skip_none_sequential(OrderedDict([
                ('final_conv', self.conv_type(self.fmaps[0], self.out_channels, kernel_size=1)),
                ('final_activation', self.final_activation())
            ]))
        else:
            return self.conv_type(self.fmaps[0], self.out_channels, kernel_size=1)

    def construct_skip_module(self, depth):
        return nn.Identity()

    def construct_downsampling_module(self, depth):
        scale_factor = self.scale_factors[depth]
        maxpool = getattr(nn, f'MaxPool{self.dim}d')
        return maxpool(kernel_size=scale_factor,
                       stride=scale_factor,
                       padding=0)

    def construct_upsampling_module(self, depth):
        scale_factor = self.scale_factors[depth]
        if scale_factor[0] == 1:
            assert scale_factor[1] == scale_factor[2]
        return nn.Upsample(scale_factor=scale_factor)



