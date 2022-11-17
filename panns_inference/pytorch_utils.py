"""This pytorch_utils.py contains functions from:
https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/pytorch/pytorch_utils.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)


class Interpolator(nn.Module):
    def __init__(self, ratio, interpolate_mode='nearest'):
        """Interpolate the sound event detection result along the time axis.

        Args:
            ratio: int
            interpolate_mode: str

        """
        super(Interpolator, self).__init__()

        if interpolate_mode == 'nearest':
            self.interpolator = NearestInterpolator(ratio)

        elif interpolate_mode == 'linear':
            self.interpolator = LinearInterpolator(ratio)

    def forward(self, x):
        """Interpolate the sound event detection result along the time axis.
        
        Args:
            x: (batch_size, time_steps, classes_num)

        Returns:
            (batch_size, new_time_steps, classes_num)
        """
        return self.interpolator(x)


class NearestInterpolator(nn.Module):
    def __init__(self, ratio):
        """Nearest interpolate the sound event detection result along the time axis.

        Args:
            ratio: int
        """
        super(NearestInterpolator, self).__init__()

        self.ratio = ratio

    def forward(self, x):
        """Interpolate the sound event detection result along the time axis.
        
        Args:
            x: (batch_size, time_steps, classes_num)

        Returns:
            upsampled: (batch_size, new_time_steps, classes_num)
        """
        (batch_size, time_steps, classes_num) = x.shape
        upsampled = x[:, :, None, :].repeat(1, 1, self.ratio, 1)
        upsampled = upsampled.reshape(batch_size, time_steps * self.ratio, classes_num)
        return upsampled


class LinearInterpolator(nn.Module):
    def __init__(self, ratio):
        """Linearly interpolate the sound event detection result along the time axis.

        Args:
            ratio: int
        """
        super(LinearInterpolator, self).__init__()

        self.ratio = ratio
    
        self.weight = torch.zeros(ratio * 2 + 1)

        for i in range(ratio):
            self.weight[i] = i / ratio

        for i in range(ratio, ratio * 2 + 1):
            self.weight[i] = 1. - (i - ratio) / ratio

        self.weight = self.weight[None, None, :]

    def forward(self, x):
        """Interpolate the sound event detection result along the time axis.
        
        Args:
            x: (batch_size, time_steps, classes_num)

        Returns:
            upsampled: (batch_size, new_time_steps, classes_num)
        """
        batch_size, time_steps, classes_num = x.shape
        x = x.transpose(1, 2).reshape(batch_size * classes_num, 1, time_steps)

        upsampled = F.conv_transpose1d(
            input=x, 
            weight=self.weight, 
            bias=None, 
            stride=self.ratio, 
            padding=self.ratio, 
            output_padding=0
        )
        new_time_steps = upsampled.shape[-1]
        upsampled = upsampled.reshape(batch_size, classes_num, new_time_steps)
        upsampled = upsampled.transpose(1, 2)
        return upsampled


def pad_framewise_output(framewise_output, frames_num):
    """Pad framewise_output to the same length as input frames.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    pad = framewise_output[:, -1 :, :].repeat(1, frames_num - framewise_output.shape[1], 1)
    """tensor for padding"""

    output = torch.cat((framewise_output, pad), dim=1)
    """(batch_size, frames_num, classes_num)"""

    return output


def do_mixup(x, mixup_lambda):
    out = x[0::2].transpose(0, -1) * mixup_lambda[0::2] + \
        x[1::2].transpose(0, -1) * mixup_lambda[1::2]
    return out.transpose(0, -1)