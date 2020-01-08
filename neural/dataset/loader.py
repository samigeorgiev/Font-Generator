from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

from . import CONFIG
from .dataset import FontsDataset

__all__ = [
    'FontsLoader',
]


class FontsLoader:
    '''Fonts Samples Loader utility class
    '''

    @staticmethod
    def _get_transform():
        return Compose([
            ToTensor(),
        ])

    @staticmethod
    def _get_set():
        return FontsDataset(CONFIG.PATH_TO_FOLDER,
                transform=FontsLoader._get_transform())

    @staticmethod
    def get_set_loader(batch_size=CONFIG.BATCH_SIZE,
                        shuffle=CONFIG.SHUFFLE,
                        num_workers=CONFIG.NUM_WORKERS):

        return DataLoader(FontsLoader._get_set(),
                batch_size, shuffle, num_workers=num_workers)
