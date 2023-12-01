import torch
from torch import nn
from .beta_schedules import *
    
class DDIM_Sampler(nn.Module): #DDIM sampler class for forward process
    
    def __init__(self, num_timesteps=100, train_timesteps=1000, clip_sample=True, schedule='linear'):
        super().__init__()