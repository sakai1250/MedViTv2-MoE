"""
WavKAN: Wavelet Kolmogorov-Arnold Networks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class WavKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_wavelets=8, wavelet_type='mexican_hat'):
        super(WavKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_wavelets = num_wavelets
        self.wavelet_type = wavelet_type
        
        # Weights for the wavelet basis combination
        # Flattened shape for F.linear: (output_dim, input_dim * num_wavelets)
        self.weights = nn.Parameter(torch.Tensor(output_dim, input_dim * num_wavelets))
        
        # Learnable translation and scale for wavelets
        self.translation = nn.Parameter(torch.Tensor(input_dim, num_wavelets))
        self.scale = nn.Parameter(torch.Tensor(input_dim, num_wavelets))
        
        # Base weight (linear residual)
        self.base_weight = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(input_dim)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        nn.init.uniform_(self.translation, -1, 1)
        nn.init.uniform_(self.scale, 0.1, 1.0)
        
    def forward(self, x):
        # x: (batch, input_dim)
        base_out = self.base_weight(x)
        
        # x_expanded: (batch, input_dim, 1)
        x_expanded = x.unsqueeze(-1)
        
        # t: (batch, input_dim, num_wavelets)
        t = (x_expanded - self.translation) / self.scale
        
        if self.wavelet_type == 'mexican_hat':
            basis = (1 - t**2) * torch.exp(-0.5 * t**2)
        elif self.wavelet_type == 'morlet':
            basis = torch.cos(5 * t) * torch.exp(-0.5 * t**2)
        elif self.wavelet_type == 'dog':
            basis = -t * torch.exp(-0.5 * t**2)
        else:
            raise ValueError(f"Unknown wavelet type: {self.wavelet_type}")
            
        # basis: (batch, input_dim, num_wavelets) -> (batch, input_dim * num_wavelets)
        basis_flat = basis.view(x.size(0), -1)
        
        # Efficient linear combination
        kan_out = F.linear(basis_flat, self.weights)
        
        return base_out + kan_out

class WavKAN(nn.Module):
    def __init__(self, layers_hidden, num_wavelets=8, wavelet_type='mexican_hat'):
        super(WavKAN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers_hidden) - 1):
            self.layers.append(WavKANLayer(
                layers_hidden[i], 
                layers_hidden[i+1], 
                num_wavelets=num_wavelets,
                wavelet_type=wavelet_type
            ))
            
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x