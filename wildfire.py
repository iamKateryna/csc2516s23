import torch
from torch.utils.data import Dataset

from preprocessdata import clip_and_normalize, random_crop_input_and_output_images

from constants import INPUT_FEATURES, OUTPUT_FEATURES

class WildfireDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset['FireMask'])

    def __getitem__(self, index):

        # clip all data to avoid extreme values, (unreasonable, spanning an extensive dynamic range).
        # The clipping values are either based on physical knowledge or set to the 0.1% and 99.9% percentiles.
        # Means and standard deviations are calculated after clipping.

        x = [clip_and_normalize(self.dataset.get(key), key) for key in INPUT_FEATURES]
        x = [feature[index] for feature in x] # 12 x 64 x 64, list of Tensors

        inputs_stacked = torch.stack(x, dim = 0) # num_channels x H x W, Tensor
        
        y = [clip_and_normalize(self.dataset.get(key).reshape(-1, 64, 64), key) for key in OUTPUT_FEATURES]
        y = y[0][index].reshape((-1, 64,64)) #1 x H x W, Tensor      
        y = y.type(torch.int32) 
        y[y < 0] = 0
    
        if self.transform:
            # random crops (crop all data to 32x32km regions)
            input_img, output_img = random_crop_input_and_output_images(inputs_stacked, y, sample_size, num_in_channels, 1)
            return input_img, output_img

        return inputs_stacked, y