from torchvision import transforms
from torch.utils.data import DataLoader
from utils import config
import os
from utils.dataset import CustomDataset
from utils.image_process import preprocess_image


def get_dataloader(batch_size=1, shuffle=True, num_workers=0):
    transform = preprocess_image

    root_data_folder = config['root_data_folder']
    raw_data_folder = os.path.join(root_data_folder, config['raw_data_folder'])
    ground_truth_csv = os.path.join(root_data_folder, config['ground_truth_csv'])
    dataset = CustomDataset(raw_data_folder, ground_truth_csv, transform=transform)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


if __name__ == '__main__':
    from icecream import ic
    dataloader = get_dataloader(shuffle=False)
    for img, label, filename in dataloader:
        ic(img.shape, label, filename)

