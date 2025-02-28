import torch
import torch.nn as nn

from pde.utils.layers import ConvLSTMCell

class ConvLSTM_Model(nn.Module):
    r"""ConvLSTM Model

    Implementation of `Convolutional LSTM Network: A Machine Learning Approach
    for Precipitation Nowcasting <https://arxiv.org/abs/1506.04214>`_.

    """

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(ConvLSTM_Model, self).__init__()
        T, C, H, W = configs['in_shape']

        self.configs = configs
        self.frame_channel = self.configs['patch_size'] * self.configs['patch_size'] * C
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        height = H // self.configs['patch_size']
        width = W // self.configs['patch_size']

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                ConvLSTMCell(in_channel, num_hidden[i], height, width, self.configs['filter_size'],
                                       self.configs['stride'], self.configs['layer_norm'])
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true, **kwargs):
        # [batch, length, channel, height, width]
        device = frames_tensor.device

        batch = frames_tensor.shape[0]
        height = frames_tensor.shape[3]
        width = frames_tensor.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(device)
            h_t.append(zeros)
            c_t.append(zeros)

        for t in range(self.configs['pre_seq_length'] + self.configs['aft_seq_length'] - 1):
            # reverse schedule sampling
            if t < self.configs['pre_seq_length']:
                net = frames_tensor[:, t]
            else:
                print(1)
                net = mask_true[:, t - self.configs['pre_seq_length']] * frames_tensor[:, t] + (1 - mask_true[:, t - self.configs['pre_seq_length']]) * x_gen

            h_t[0], c_t[0] = self.cell_list[0](net, h_t[0], c_t[0])

            for i in range(1, self.num_layers):
                h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i])

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, channel, height, width]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()

        return next_frames