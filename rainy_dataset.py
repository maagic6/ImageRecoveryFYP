# code for loading dataset from csv file

import os
import pandas as pd
from torch.utils.data import Dataset
from scipy import io

class RainyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform = None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # rename variables later
        img_path1 = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image1 = io.imread(img_path1)
        img_path2 = os.path.join(self.root_dir, self.annotations.iloc[index, 1])
        image2 = io.imread(img_path2)

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return (image1, image2)