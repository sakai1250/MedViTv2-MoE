import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexGaborConv2d(nn.Module):
    """
    Adaptive Complex-Wavelet Convolution Layer (AC-Conv).
    Implements learnable Gabor wavelets inspired by DTCWT (Dual-Tree Complex Wavelet Transform).
    
    Features:
    - Learnable Mother Wavelet: sigma, omega, theta, psi are learnable parameters.
    - Complex-valued convolution: Maintains phase information internally.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, dilation=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation

        # Number of filters to generate
        # We assume depthwise-like structure if groups == in_channels
        self.num_kernels = out_channels * (in_channels // groups)

        # Learnable Parameters for the Mother Wavelet
        # Sigma (Scale): Controls the width of the Gaussian envelope
        self.sigma = nn.Parameter(torch.ones(self.num_kernels))
        # Omega (Frequency): Controls the frequency of the carrier
        self.omega = nn.Parameter(torch.rand(self.num_kernels) * math.pi)
        # Theta (Orientation): Controls the direction
        self.theta = nn.Parameter(torch.rand(self.num_kernels) * 2 * math.pi)
        # Phase (Shift): Controls the phase shift
        self.psi = nn.Parameter(torch.rand(self.num_kernels) * 2 * math.pi)
        # Amplitude weight
        self.weight = nn.Parameter(torch.ones(self.num_kernels))

        # Coordinate grid
        self.register_buffer('grid', self._build_grid(kernel_size))

    def _build_grid(self, size):
        coords = torch.arange(size, dtype=torch.float32)
        coords -= (size - 1) / 2.0
        y, x = torch.meshgrid(coords, coords, indexing='ij')
        return torch.stack([x, y], dim=0)

    def get_kernel(self):
        # grid: (2, K, K)
        x = self.grid[0]
        y = self.grid[1]
        
        # Reshape params for broadcasting: (N, 1, 1)
        sigma = self.sigma.view(-1, 1, 1)
        omega = self.omega.view(-1, 1, 1)
        theta = self.theta.view(-1, 1, 1)
        psi = self.psi.view(-1, 1, 1)
        weight = self.weight.view(-1, 1, 1)

        # Rotate coordinates
        x_rot = x * torch.cos(theta) + y * torch.sin(theta)
        y_rot = -x * torch.sin(theta) + y * torch.cos(theta)

        # Gaussian Envelope
        # Add epsilon to sigma to avoid division by zero
        sigma_sq = torch.square(sigma) + 1e-6
        envelope = torch.exp(-(torch.square(x_rot) + torch.square(y_rot)) / (2 * sigma_sq))

        # Complex Carrier
        # Real part (Cosine)
        real_part = weight * envelope * torch.cos(omega * x_rot + psi) # 
        # Imaginary part (Sine)
        imag_part = weight * envelope * torch.sin(omega * x_rot + psi)

        # Reshape to (Out, In/Groups, K, K)
        k_shape = (self.out_channels, self.in_channels // self.groups, self.kernel_size, self.kernel_size)
        kernel_real = real_part.view(*k_shape)
        kernel_imag = imag_part.view(*k_shape)
        
        return kernel_real, kernel_imag

    def forward(self, x):
        kernel_real, kernel_imag = self.get_kernel()
        
        # Handle complex input or real input
        if torch.is_complex(x):
            x_real, x_imag = x.real, x.imag
        else:
            x_real = x
            x_imag = torch.zeros_like(x)

        # Complex Convolution
        out_real = F.conv2d(x_real, kernel_real, stride=self.stride, padding=self.padding, groups=self.groups, dilation=self.dilation) - \
                   F.conv2d(x_imag, kernel_imag, stride=self.stride, padding=self.padding, groups=self.groups, dilation=self.dilation)
        
        out_imag = F.conv2d(x_real, kernel_imag, stride=self.stride, padding=self.padding, groups=self.groups, dilation=self.dilation) + \
                   F.conv2d(x_imag, kernel_real, stride=self.stride, padding=self.padding, groups=self.groups, dilation=self.dilation)

        return torch.complex(out_real, out_imag)

class ACKAN(nn.Module):
    """
    Adaptive Complex-Wavelet KAN (AC-KAN) Module.
    Integrates Channel-wise Basis Selection and Multi-Resolution analysis.
    Optimized to use single grouped convolution instead of loop.
    """
    def __init__(self, channels, groups=4, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.groups = groups
        
        assert channels % groups == 0, "Channels must be divisible by groups"
        self.group_channels = channels // groups
        
        # Optimized: Use a single layer with groups=channels (depthwise across all)
        # This is mathematically equivalent to the loop if we initialize correctly.
        self.layer = ComplexGaborConv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            groups=channels # Depthwise across all channels
        )
        
        # Channel-wise Basis Selection: Initialize parameters by group slices
        with torch.no_grad():
            # Get parameter views reshaped to (groups, group_channels)
            # Original params are (channels,), which is (groups * group_channels,)
            
            # Helper to get a view for specific parameter
            def get_view(param):
                return param.view(groups, self.group_channels)
            
            omega_view = get_view(self.layer.omega)
            sigma_view = get_view(self.layer.sigma)
            theta_view = get_view(self.layer.theta)
            
            for i in range(groups):
                if i == 0: # Low Frequency
                    omega_view[i].uniform_(0, math.pi / 4)
                    sigma_view[i].uniform_(2.0, 3.0)
                elif i == 1: # High Frequency
                    omega_view[i].uniform_(math.pi / 2, math.pi)
                    sigma_view[i].uniform_(0.5, 1.0)
                elif i == 2: # Oriented Edges (Horizontal/Vertical)
                    omega_view[i].uniform_(math.pi / 4, math.pi / 2)
                    # Bias theta towards 0 or 90 degrees
                    thetas = (torch.randint(0, 2, (self.group_channels,)).float() * math.pi / 2) + torch.randn(self.group_channels) * 0.1
                    theta_view[i].copy_(thetas)
                else: # Random / Texture
                    omega_view[i].uniform_(math.pi / 4, 3 * math.pi / 4)
                    theta_view[i].uniform_(0, 2 * math.pi)

    def forward(self, x):
        # x: (B, C, H, W)
        # Apply Complex Wavelet (Single optimized kernel)
        out_complex = self.layer(x)
        
        # Compute Magnitude for Shift-Invariance
        # |z| = sqrt(real^2 + imag^2)
        out_mag = torch.abs(out_complex) 
        
        return out_mag