''' IGNORE

import torchvision, os
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

data_path = 'mnist/'
if not os.path.exists(os.path.join(data_path, "MNIST")):
    dataset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=torchvision.transforms.ToTensor())
else:
    dataset = torchvision.datasets.MNIST(root=data_path, train=True, download=False, transform=torchvision.transforms.ToTensor())
train_dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
x, y = next(iter(train_dataloader))
print('input shape:', x.shape)
print('labels:', y)

plt.imshow(torchvision.utils.make_grid(x)[0], cmap='Greys');
plt.show()'''