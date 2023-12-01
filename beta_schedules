import torch

torch.pi = torch.acos(torch.zeros(1)).item() * 2

def get_beta_schedule(variant, timesteps):
    if variant=='linear':
        return linear_beta_schedule(timesteps)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)