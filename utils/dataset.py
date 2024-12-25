import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from icecream import ic
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, raw_data_folder, ground_truth_csv, transform=None):
        self.raw_data_folder = raw_data_folder
        self.transform = transform
        self.df = pd.read_csv(ground_truth_csv, sep=' ', names=['filename', 'label'])

    def __getitem__(self, idx):
        filename, label = self.df.iloc[idx]
        file_path = os.path.join(self.raw_data_folder, filename)
        img = self.transform(file_path)
        return img, label, file_path

    def __len__(self):
        return len(self.df)


if __name__ == '__main__':
    from icecream import ic
    from utils import config
    from utils.image_process import preprocess_image
    import os

    root_data_folder = config['root_data_folder']
    raw_data_folder = os.path.join(root_data_folder, config['raw_data_folder'])
    annotation_data_folder = os.path.join(root_data_folder, config['annotations'])
    ground_truth_csv = os.path.join(root_data_folder, config['ground_truth_csv'])
    dataset = CustomDataset(raw_data_folder, ground_truth_csv, transform=preprocess_image)
    img, label, filename = dataset[1]
    ic(img.shape, label, filename)
