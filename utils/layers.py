import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from .functs import to_2tuple
from .norm import build_norm_layer
from .conv import build_conv_layer

from neuralop.layers.spectral_convolution import SpectralConv

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
    
    def drop_path(self, x, drop_prob: float = 0., training: bool = False):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
        This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
        the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
        See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
        changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
        'survival rate' as the argument.
        """
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training)


class AdaptivePadding(nn.Module):
    """Applies padding adaptively to the input.

    This module can make input get fully covered by filter
    you specified. It support two modes "same" and "corner". The
    "same" mode is same with "SAME" padding mode in TensorFlow, pad
    zero around input. The "corner"  mode would pad zero
    to bottom right.

    Args:
        kernel_size (int | tuple): Size of the kernel. Default: 1.
        stride (int | tuple): Stride of the filter. Default: 1.
        dilation (int | tuple): Spacing between kernel elements.
            Default: 1.
        padding (str): Support "same" and "corner", "corner" mode
            would pad zero to bottom right, and "same" mode would
            pad zero around input. Default: "corner".

    Example:
        >>> kernel_size = 16
        >>> stride = 16
        >>> dilation = 1
        >>> input = torch.rand(1, 1, 15, 17)
        >>> adap_pad = AdaptivePadding(
        >>>     kernel_size=kernel_size,
        >>>     stride=stride,
        >>>     dilation=dilation,
        >>>     padding="corner")
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
        >>> input = torch.rand(1, 1, 16, 17)
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
    """

    def __init__(self, kernel_size=1, stride=1, dilation=1, padding='corner'):
        super().__init__()
        assert padding in ('same', 'corner')

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
    

    def get_pad_shape(self, input_shape):
        """Calculate the padding size of input.

        Args:
            input_shape (:obj:`torch.Size`): arrange as (H, W).

        Returns:
            Tuple[int]: The padding size along the
            original H and W directions
        """
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h +
                    (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w +
                    (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w

    def forward(self, x):
        """Add padding to `x`

        Args:
            x (Tensor): Input tensor has shape (B, C, H, W).

        Returns:
            Tensor: The tensor with adaptive padding
        """
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == 'corner':
                x = F.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == 'same':
                x = F.pad(x, [
                    pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                    pad_h - pad_h // 2
                ])
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding.

    We use a conv layer to implement PatchEmbed.

    Args:
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        conv_type (str): The type of convolution
            to generate patch embedding. Default: "Conv2d".
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int): The slide stride of embedding conv.
            Default: 16.
        padding (int | tuple | string): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int): The dilation rate of embedding conv. Default: 1.
        bias (bool): Bias of embed conv. Default: True.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        input_size (int | tuple | None): The size of input, which will be
            used to calculate the out size. Only works when `dynamic_size`
            is False. Default: None.
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=768,
                 conv_type='Conv2d',
                 kernel_size=16,
                 stride=16,
                 padding='corner',
                 dilation=1,
                 bias=True,
                 norm_cfg=None,
                 input_size=None,
                 init_cfg=None):
        super().__init__()

        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adaptive_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of conv
            padding = 0
        else:
            self.adaptive_padding = None
        padding = to_2tuple(padding)

        self.projection = build_conv_layer(
            dict(type=conv_type),
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

        if input_size:
            input_size = to_2tuple(input_size)
            # `init_out_size` would be used outside to
            # calculate the num_patches
            # e.g. when `use_abs_pos_embed` outside
            self.init_input_size = input_size
            if self.adaptive_padding:
                pad_h, pad_w = self.adaptive_padding.get_pad_shape(input_size)
                input_h, input_w = input_size
                input_h = input_h + pad_h
                input_w = input_w + pad_w
                input_size = (input_h, input_w)

            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            h_out = (input_size[0] + 2 * padding[0] - dilation[0] *
                     (kernel_size[0] - 1) - 1) // stride[0] + 1
            w_out = (input_size[1] + 2 * padding[1] - dilation[1] *
                     (kernel_size[1] - 1) - 1) // stride[1] + 1
            self.init_out_size = (h_out, w_out)
        else:
            self.init_input_size = None
            self.init_out_size = None



    def forward(self, x):
        """
        Args:
            x (Tensor): Has shape (B, C, H, W). In most case, C is 3.

        Returns:
            tuple: Contains merged results and its spatial shape.

            - x (Tensor): Has shape (B, out_h * out_w, embed_dims)
            - out_size (tuple[int]): Spatial shape of x, arrange as
              (out_h, out_w).
        """

        if self.adaptive_padding:
            x = self.adaptive_padding(x)

        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size
    
    
class ConvLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super(ConvLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, height, width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1,
                                   stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        i_x, f_x, g_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t
        o_t = torch.sigmoid(o_x + o_h)
        h_new = o_t * torch.tanh(c_new)
        return h_new, c_new


class ConvLSTMFNOCell(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super(ConvLSTMFNOCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                SpectralConv2d(in_channel, num_hidden * 4, [10,10]),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_h = nn.Sequential(
                SpectralConv2d(num_hidden, num_hidden * 4, [10,10]),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_o = nn.Sequential(
                SpectralConv2d(num_hidden * 2, num_hidden, [10,10]),
                nn.LayerNorm([num_hidden, height, width])
            )
        else:
            self.conv_x = nn.Sequential(
                SpectralConv2d(in_channel, num_hidden * 4, [10,10]),
            )
            self.conv_h = nn.Sequential(
                SpectralConv2d(num_hidden, num_hidden * 4, [10,10]),
            )
            self.conv_o = nn.Sequential(
                SpectralConv2d(num_hidden * 2, num_hidden, [10,10]),
            )
        self.conv_last = SpectralConv2d(num_hidden * 2, num_hidden, [10,10])

    def forward(self, x_t, h_t, c_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        i_x, f_x, g_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t
        o_t = torch.sigmoid(o_x + o_h)
        h_new = o_t * torch.tanh(c_new)
        return h_new, c_new
    
class SpectralConv2d(SpectralConv):
        
    def forward(self, x, indices=0):
        batchsize, channels, height, width = x.shape

        x = torch.fft.rfft2(x.float(), norm=self.fft_norm, dim=(-2, -1))

        # The output will be of size (batch_size, self.out_channels,
        # x.size(-2), x.size(-1)//2 + 1)
        out_fft = torch.zeros(
            [batchsize, self.out_channels, height, width // 2 + 1],
            dtype=x.dtype,
            device=x.device,
        )

        slices0 = (
            slice(None),  # Equivalent to: [:,
            slice(None),  # ............... :,
            slice(self.n_modes[0] // 2),  # :half_n_modes[0],
            slice(self.n_modes[1]),  #      :half_n_modes[1]]
        )
        slices1 = (
            slice(None),  # Equivalent to:        [:,
            slice(None),  # ...................... :,
            slice(-self.n_modes[0] // 2, None),  # -half_n_modes[0]:,
            slice(self.n_modes[1]),  # ......      :half_n_modes[1]]
        )

        """Upper block (truncate high frequencies)."""
        out_fft[slices0] = self._contract(
            x[slices0], self._get_weight(indices)[slices1], separable=self.separable
        )

        """Lower block"""
        out_fft[slices1] = self._contract(
            x[slices1], self._get_weight(indices)[slices0], separable=self.separable
        )

        if self.output_scaling_factor is not None:
            width = round(width * self.output_scaling_factor[indices][0])
            height = round(height * self.output_scaling_factor[indices][1])

        x = torch.fft.irfft2(
            out_fft, s=(height, width), dim=(-2, -1), norm=self.fft_norm
        )

        if self.bias is not None:
            x = x + self.bias[indices, ...]

        return x
