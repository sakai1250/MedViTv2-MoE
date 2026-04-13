"""
FM-KAN: Frequency Modulation Kolmogorov-Arnold Network

Core idea: Use FM synthesis basis functions where one sine wave modulates
the frequency of another, providing rich harmonic structure through
Bessel function expansion.

Basis function:
    φ_k(x) = sin(ω_k · x + α_k · sin(ω'_k · x + ψ_k)) · exp(-β_k · x²)

Parameters (all learnable):
    ω_k   : Carrier frequency
    ω'_k  : Modulator frequency
    α_k   : Modulation index (controls complexity)
    ψ_k   : Phase shift
    β_k   : Gaussian window width

Key property (Jacobi-Anger expansion):
    sin(ωx + α sin(ω'x)) = Σ_n J_n(α) · sin((ω + nω')x)
    
    A single basis function generates infinite frequency components
    controlled by just the modulation index α.

Medical imaging strengths:
    - Parameter-efficient multi-frequency representation
    - Chirp-like behavior captures contrast-varying boundaries
    - Natural band splitting via Bessel functions (multi-scale features)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FMKANLayer(nn.Module):
    """
    Single layer of FM-KAN.
    
    Maps input_dim -> output_dim using FM synthesis basis functions.
    Each (input, output) pair has `num_basis` FM basis functions,
    and the output is a weighted sum of these basis evaluations.
    """
    def __init__(self, input_dim, output_dim, num_basis=8):
        super(FMKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_basis = num_basis

        self.layernorm = nn.LayerNorm(input_dim)

        # Learnable FM parameters: shape (input_dim, num_basis)
        # ω_k: Carrier frequency
        self.omega = nn.Parameter(torch.empty(input_dim, num_basis))
        # ω'_k: Modulator frequency
        self.omega_prime = nn.Parameter(torch.empty(input_dim, num_basis))
        # α_k: Modulation index (controls harmonic richness)
        self.alpha = nn.Parameter(torch.empty(input_dim, num_basis))
        # ψ_k: Phase shift
        self.psi = nn.Parameter(torch.empty(input_dim, num_basis))
        # β_k: Gaussian window width (ensure positive via softplus)
        self.beta = nn.Parameter(torch.empty(input_dim, num_basis))

        # Output mixing weights: (input_dim, output_dim, num_basis)
        self.coeffs = nn.Parameter(torch.empty(input_dim, output_dim, num_basis))

        # Per-input scaling
        self.w = nn.Parameter(torch.ones(input_dim, output_dim))

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize FM parameters for diverse frequency coverage.
        """
        # Carrier frequencies: spread across different bands
        nn.init.uniform_(self.omega, 0.5, 3.0 * math.pi)
        # Modulator frequencies: generally lower than carrier
        nn.init.uniform_(self.omega_prime, 0.1, 2.0 * math.pi)
        # Modulation index: start small for stable training
        nn.init.uniform_(self.alpha, 0.0, 1.0)
        # Phase: uniform random
        nn.init.uniform_(self.psi, 0.0, 2.0 * math.pi)
        # Gaussian window: moderate width
        nn.init.uniform_(self.beta, 0.1, 1.0)
        # Output coefficients
        nn.init.xavier_normal_(self.coeffs)
        # Scaling
        nn.init.normal_(self.w, mean=1.0, std=0.1)

    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            y: (batch_size, output_dim)
        """
        B, I = x.shape

        # Normalize input
        x = self.layernorm(x)
        x = torch.tanh(x)  # Map to [-1, 1]

        # x: (B, I) -> (B, I, 1) for broadcasting with (I, K) params
        x_expanded = x.unsqueeze(-1)  # (B, I, 1)

        # FM Synthesis basis: φ_k(x) = sin(ω_k·x + α_k·sin(ω'_k·x + ψ_k)) · exp(-β_k·x²)
        
        # Modulator signal: sin(ω'_k · x + ψ_k)
        modulator = torch.sin(self.omega_prime.unsqueeze(0) * x_expanded + self.psi.unsqueeze(0))
        # (B, I, K)
        
        # Carrier with FM: sin(ω_k · x + α_k · modulator)
        carrier_phase = self.omega.unsqueeze(0) * x_expanded + self.alpha.unsqueeze(0) * modulator
        fm_signal = torch.sin(carrier_phase)
        # (B, I, K)

        # Gaussian window: exp(-β_k · x²), β_k > 0 via softplus
        beta_positive = F.softplus(self.beta)  # Ensure positive
        gaussian_window = torch.exp(-beta_positive.unsqueeze(0) * x_expanded.pow(2))
        # (B, I, K)

        # Windowed FM basis
        basis = fm_signal * gaussian_window  # (B, I, K)

        # Weighted combination: coeffs (I, O, K) * w (I, O)
        effective_coeffs = self.coeffs * self.w.unsqueeze(-1)  # (I, O, K)

        # Output: y = sum_over_k(basis * coeffs) summed over input dims
        # basis: (B, I, K) -> einsum with effective_coeffs: (I, O, K) -> (B, O)
        y = torch.einsum('bik,iok->bo', basis, effective_coeffs)

        return y


class FMKAN(nn.Module):
    """
    FM-KAN: Multi-layer Frequency Modulation KAN.
    
    Drop-in replacement for ORKAN/FasterKAN with the same interface:
        FMKAN([input_dim, hidden_dim, output_dim])
    
    Args:
        layers_hidden: List of layer dimensions, e.g. [64, 192, 64]
        num_basis: Number of FM basis functions per (input, output) pair
    """
    def __init__(self, layers_hidden, num_basis=8):
        super(FMKAN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers_hidden) - 1):
            self.layers.append(
                FMKANLayer(
                    layers_hidden[i],
                    layers_hidden[i + 1],
                    num_basis=num_basis
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
