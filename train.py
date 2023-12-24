import torch, torchvision, os
from torch import nn
from torch.utils.data import DataLoader
from diffusers import DDIMScheduler
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from modules.ClassConditionedUNet import ClassConditionedUNet

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

data_path = 'mnist/'
if not os.path.exists(os.path.join(data_path, "MNIST")):
    dataset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=torchvision.transforms.ToTensor())
else:
    dataset = torchvision.datasets.MNIST(root=data_path, train=True, download=False, transform=torchvision.transforms.ToTensor())

noise_scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

n_epochs = 10
net = ClassConditionedUNet().to(device)
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(net.parameters(), lr=1e-3) 

# losses
losses = []

# training loop
for epoch in range(n_epochs):
    for x, y in tqdm(train_dataloader):
        
        x = x.to(device) * 2 - 1 
        y = y.to(device)
        noise = torch.randn_like(x)
        timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
        noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

        pred = net(noisy_x, timesteps, y)
        loss = loss_fn(pred, noise)

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())

    avg_loss = sum(losses[-100:])/100
    print(f'Finished epoch {epoch}')
    print(f'Average of the last 100 loss values: {avg_loss:05f}')

torch.save(net.state_dict(), 'test_model.pth')
plt.plot(losses)