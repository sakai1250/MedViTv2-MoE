"""
DR-KAN: Dynamic Rational Kolmogorov-Arnold Network

Core idea: The rational function coefficients are not static but dynamically
generated from the input via a lightweight coefficient generator (SE-Net style).
This makes the nonlinearity adaptive per-sample.

Static RKAN:   R(x) = P(x; a_fixed) / Q(x; b_fixed)
DR-KAN:        R(x) = P(x; a_base + Δa(x)) / Q(x; b_base + Δb(x))

The coefficient generator uses Global Average Pooling + small MLP to produce
per-sample coefficient adjustments, similar to how Dynamic Convolution or
SE-Net generates channel-wise attention.

Key advantages:
    - Input-adaptive nonlinearity: different samples use different rational functions
    - Padé-like approximation with dynamic poles
    - Lightweight: coefficient generator adds minimal parameters
    - Medical imaging: adapts feature extraction per pathology type
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicRationalKANLayer(nn.Module):
    """
    Dynamic Rational KAN Layer.
    
    The rational function R(x) = P(x)/Q(x) has base coefficients that are
    modulated by input-dependent adjustments from a lightweight generator.
    """
    def __init__(self, input_dim, output_dim, degree_p=5, degree_q=4, reduction=4):
        super(DynamicRationalKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree_p = degree_p
        self.degree_q = degree_q

        self.layernorm = nn.LayerNorm(input_dim)

        # Base (static) coefficients — same as RKAN
        self.base_numerator = nn.Parameter(
            torch.empty(input_dim, output_dim, degree_p + 1)
        )
        self.base_denominator = nn.Parameter(
            torch.empty(input_dim, degree_q)
        )

        # Per-(input, output) scaling factor
        self.w = nn.Parameter(torch.ones(input_dim, output_dim))

        # Dynamic coefficient generator (SE-Net style)
        # Input: (B, input_dim) -> bottleneck -> coefficients
        num_dynamic_coeffs = (degree_p + 1) + degree_q
        r = max(1, input_dim // reduction)
        self.coeff_generator = nn.Sequential(
            nn.Linear(input_dim, r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(r, num_dynamic_coeffs, bias=False),
            nn.Tanh()  # Bound adjustments to [-1, 1]
        )

        # Learnable scaling for dynamic adjustments (initialized small)
        self.dynamic_scale = nn.Parameter(torch.ones(1) * 0.1)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_normal_(self.base_numerator)
        nn.init.normal_(self.base_denominator, std=0.1)
        nn.init.normal_(self.w, mean=1.0, std=0.1)
        # Initialize coefficient generator weights small
        for m in self.coeff_generator:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            y: (batch_size, output_dim)
        """
        B, I = x.shape

        # Generate dynamic coefficient adjustments from input
        # Use the raw input (before normalization) for conditioning
        delta = self.coeff_generator(x)  # (B, num_coeffs)
        delta = delta * self.dynamic_scale  # Scale down adjustments

        # Split into numerator and denominator adjustments
        delta_p = delta[:, :self.degree_p + 1]  # (B, degree_p + 1)
        delta_q = delta[:, self.degree_p + 1:]  # (B, degree_q)

        # Normalize input
        x = self.layernorm(x)
        x = torch.tanh(x)

        # Compute powers of x: x^0, x^1, ..., x^max_degree
        max_degree = max(self.degree_p, self.degree_q)
        powers = [torch.ones(B, I, device=x.device, dtype=x.dtype)]
        if max_degree > 0:
            powers.append(x)
        for d in range(2, max_degree + 1):
            powers.append(powers[-1] * x)
        X = torch.stack(powers, dim=2)  # (B, I, max_degree + 1)

        # ---------------------------------------------------------
        # Dynamic Denominator: Q(x) = 1 + |Σ (b_base + Δb) * x^j|
        # ---------------------------------------------------------
        X_q = X[:, :, 1:self.degree_q + 1]  # (B, I, degree_q)
        # Effective denominator coefficients: base + dynamic adjustment
        # base: (I, degree_q), delta_q: (B, degree_q) -> broadcast over I
        eff_denom = self.base_denominator.unsqueeze(0) + delta_q.unsqueeze(1)  # (B, I, degree_q)
        Q_sum = (X_q * eff_denom).sum(dim=2)  # (B, I)
        Q = 1.0 + torch.abs(Q_sum)  # (B, I), always > 0

        # ---------------------------------------------------------
        # Numerator: P(x) uses base coefficients (static per-output)
        # Dynamic adjustment applied via scaling the rational basis
        # ---------------------------------------------------------
        X_p = X[:, :, :self.degree_p + 1]  # (B, I, degree_p + 1)

        # Rational basis: R(x) = X_p / Q(x)
        R = X_p / Q.unsqueeze(-1)  # (B, I, degree_p + 1)

        # Apply dynamic numerator modulation:
        # delta_p: (B, degree_p + 1) -> modulates each basis function globally
        # R_modulated = R * (1 + delta_p)  broadcast over I
        R_mod = R * (1.0 + delta_p.unsqueeze(1))  # (B, I, degree_p + 1)

        # Flatten for linear projection
        R_flat = R_mod.reshape(B, I * (self.degree_p + 1))

        # Merge scaling w into numerator coefficients
        effective_coeffs = self.base_numerator * self.w.unsqueeze(-1)

        # Weight matrix: (O, I * (p+1))
        weight_matrix = effective_coeffs.permute(1, 0, 2).reshape(
            self.output_dim, I * (self.degree_p + 1)
        )

        # Linear projection
        y = F.linear(R_flat, weight_matrix)

        return y


class DRKAN(nn.Module):
    """
    DR-KAN: Dynamic Rational KAN (Multi-layer).

    Drop-in replacement for RKAN/ORKAN/FasterKAN with the same interface:
        DRKAN([input_dim, hidden_dim, output_dim])

    Args:
        layers_hidden: List of layer dimensions, e.g. [64, 192, 64]
        degree_p: Numerator polynomial degree (default: 5)
        degree_q: Denominator polynomial degree (default: 4)
        reduction: Bottleneck reduction ratio for coefficient generator
    """
    def __init__(self, layers_hidden, degree_p=5, degree_q=4, reduction=4):
        super(DRKAN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers_hidden) - 1):
            self.layers.append(
                DynamicRationalKANLayer(
                    layers_hidden[i],
                    layers_hidden[i + 1],
                    degree_p=degree_p,
                    degree_q=degree_q,
                    reduction=reduction
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
