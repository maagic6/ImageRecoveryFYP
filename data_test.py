# code for testing and printing tensor sizes of dataset

from rainy_dataset import RainyDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

dataset = RainyDataset(csv_file='None', root_dir='None', transform= transforms.ToTensor())

# create DataLoader
train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# iterate over the DataLoader
for x, y in train_dataloader:
    print(x.shape, y.shape)