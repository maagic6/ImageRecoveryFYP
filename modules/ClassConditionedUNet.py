import torch
from torch import nn
from diffusers import UNet2DModel

class ClassConditionedUNet(nn.Module):
  def __init__(self, num_classes=10, class_emb_size=4):
    super().__init__()
    self.class_emb = nn.Embedding(num_classes, class_emb_size)
    self.model = UNet2DModel(
        sample_size=28,
        in_channels=1 + class_emb_size, # additional input channels for class conditioning
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(32, 64, 64), 
        down_block_types=( 
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ), 
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
          ),
    )

  # Our forward method now takes the class labels as an additional argument
  def forward(self, x, t, class_labels):
    # Shape of x:
    bs, ch, w, h = x.shape
    
    # class conditioning in right shape to add as additional input channels
    class_cond = self.class_emb(class_labels) # Map to embedding dimension
    class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(bs, class_cond.shape[1], w, h)
    # x is shape (bs, 1, 28, 28) and class_cond is now (bs, 4, 28, 28)

    # x and class cond concatenated together along dimension 1
    net_input = torch.cat((x, class_cond), 1) # (bs, 5, 28, 28)

    # Feed this to the UNet alongside the timestep and return the prediction
    return self.model(net_input, t).sample # (bs, 1, 28, 28)