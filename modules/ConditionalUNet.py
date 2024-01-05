import torch
from torch import nn
from diffusers import UNet2DModel

class ConditionalUNet(nn.Module):
  def __init__(self, num_channels=1):
    super().__init__()
    self.model = UNet2DModel(
        sample_size = 28,
        in_channels = num_channels + 1, # additional input channel for condition image
        out_channels = 1,
        layers_per_block = 2,
        block_out_channels =(32, 64, 64), 
        down_block_types = ( 
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ), 
        up_block_types = (
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
          ),
    )

  def forward(self, x, t, condition_image):
    # x and class cond concatenated together along dimension 1
    net_input = torch.cat((x, condition_image), 1)
    # feed to UNet alongside the timestep and return prediction
    return self.model(net_input, t).sample