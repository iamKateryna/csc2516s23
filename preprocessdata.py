import torch
from torch import Tensor
from torchvision import transforms
from typing import Tuple

from constants import DATA_STATS


def clip_and_normalize(
    inputs: Tensor, 
    key: str
    ) -> Tensor:
    
    """Clips and normalizes inputs with the stats corresponding to `key`.

      Args:
        inputs: Inputs to clip and normalize.
        key: Key describing the inputs.

      Returns:
        Clipped and normalized input.
    """
    min_val, max_val, mean, std = DATA_STATS[key]
    inputs = torch.clamp(inputs, min_val, max_val)
    inputs = inputs - mean # batch size x H*W
    return torch.nan_to_num(torch.div(inputs, std), nan=0.0, posinf=0.0, neginf=0.0).reshape(-1, 64, 64) #divide inputs by std, replace NaN with 0, batch size x H x W

def random_crop_input_and_output_images(
input_img: Tensor,
    output_img: Tensor,
    sample_size: int,
    num_in_channels: int,
    num_out_channels: int,
) -> Tuple[Tensor, Tensor]:
    """Randomly axis-align crop input and output image tensors.

    Args:
        input_img: tensor with dimensions HWC.
        output_img: tensor with dimensions HWC.
        sample_size: side length (square) to crop to.
        num_in_channels: number of channels in input_img.
        num_out_channels: number of channels in output_img.
    Returns:
        input_img: tensor with dimensions HWC.
        output_img: tensor with dimensions HWC.
    """
    combined = torch.cat([input_img, output_img], dim=0) # num_channels x batch size x H x W
    transform = transforms.RandomCrop((sample_size, sample_size)) #RandomCrop Crops 2 last dims
    combined = transform(combined) # num_channels x batch size x sample_size x sample_size
    #print(combined.shape)
    input_img = combined[  0: num_in_channels, :] # num_channels (12, only input) x batch size x sample_size x sample_size
    output_img = combined[ -num_out_channels:, :] # num_channels (1, only output) x batch size x sample_size x sample_size
    return input_img, output_img


def center_crop_input_and_output_images(
    input_img: Tensor,
    output_img: Tensor,
    sample_size: int,
) -> Tuple[Tensor, Tensor]:
    """Center crops input and output image tensors.

    Args:
        input_img: tensor with dimensions HWC.
        output_img: tensor with dimensions HWC.
        sample_size: side length (square) to crop to.
    Returns:
        input_img: tensor with dimensions HWC.
        output_img: tensor with dimensions HWC.
    """
    central_fraction = sample_size / input_img.shape[0]
    input_img = transforms.CenterCrop(input_img, central_fraction)
    output_img = transforms.CenterCrop(output_img, central_fraction)
    return input_img, output_img