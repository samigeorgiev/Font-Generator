import os
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = [
    'FontsDataset',
]


class FontsDataset(Dataset):
    '''Fonts Datasets loader from folder
    '''

    def __init__(self, path_to_folder, transform=None):
        '''
        Parameters:
        -----------
        path_to_folder : string, directory with all the images
        transform : torch transform, optional transform
        to be applied on a sample.

        '''

        super(FontsDataset, self).__init__()

        self.path_to_folder = path_to_folder
        self.transform = transform

        self._img_names = [name for name in os.listdir(path_to_folder) \
             if name[0] != '.' and '.png' in name]


    def __len__(self):
        return len(self._img_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path_to_img = os.path.join(self.path_to_folder, self._img_names[idx])
        image = np.asarray(Image.open(path_to_img).convert('L'))
        image = np.expand_dims(image, axis=2)

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'name': self._img_names[idx]}
