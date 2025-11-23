import torch

def disturbance_process(n, t, std=0.1, decay_rate=0.1):
    w = std * torch.randn(n)      # Random Gaussian
    w *= torch.exp(torch.tensor(-decay_rate * t))  # Exponential decay
    return w