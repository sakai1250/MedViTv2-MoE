"""
RationalKAN: Rational Function based Kolmogorov-Arnold Network

Core idea: Use pure rational functions P(x)/Q(x) as basis functions,
where both numerator and denominator are learnable polynomials.
This is similar to Padé approximation, providing high approximation power
with relatively few parameters.

Basis function:
    R(x) = P(x) / Q(x)
    P(x) = Σ_{i=0}^{p} a_i · x^i   (numerator polynomial)
    Q(x) = 1 + |Σ_{j=1}^{q} b_j · x^j|  (denominator, strictly positive)

Key properties:
    - Padé-like approximation: captures poles and singularities
    - More expressive than polynomial basis for the same degree
    - Positive denominator guarantees numerical stability
    - No orthogonal polynomial overhead (simpler than ORKAN)

Medical imaging strengths:
    - Efficient representation of sharp transitions (tissue boundaries)
    - Adaptive nonlinearity via learnable poles
    - Compact parameterization with high expressiveness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RationalKANLayer(nn.Module):
    """
    Single RationalKAN layer mapping input_dim -> output_dim.
    
    Uses rational function basis R(x) = P(x) / Q(x) where:
    - P(x) is a degree-p polynomial (numerator)
    - Q(x) = 1 + |sum of degree-q terms| (denominator, always > 0)
    """
    def __init__(self, input_dim, output_dim, degree_p=5, degree_q=4):
        super(RationalKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree_p = degree_p
        self.degree_q = degree_q

        self.layernorm = nn.LayerNorm(input_dim)

        # Numerator coefficients: a_i for P(x) = Σ a_i * x^i
        # Shape: (input_dim, output_dim, degree_p + 1)
        self.numerator_coeffs = nn.Parameter(
            torch.empty(input_dim, output_dim, degree_p + 1)
        )

        # Denominator coefficients: b_j for Q(x) = 1 + |Σ b_j * x^j|
        # Shared across output dims for efficiency (like ORKAN's shared denominator)
        # Shape: (input_dim, degree_q)  — no constant term (starts from x^1)
        self.denominator_coeffs = nn.Parameter(
            torch.empty(input_dim, degree_q)
        )

        # Per-(input, output) scaling factor
        self.w = nn.Parameter(torch.ones(input_dim, output_dim))

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_normal_(self.numerator_coeffs)
        nn.init.normal_(self.denominator_coeffs, std=0.1)
        nn.init.normal_(self.w, mean=1.0, std=0.1)
        self._cached_Q = None  # Cache Q for pole regularization

    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            y: (batch_size, output_dim)
        """
        B, I = x.shape

        # Normalize input to [-1, 1]
        x = self.layernorm(x)
        x = torch.tanh(x)

        # Compute powers of x: x^0, x^1, ..., x^max_degree
        # Shape: (B, I, max_degree + 1)
        max_degree = max(self.degree_p, self.degree_q)
        powers = [torch.ones(B, I, device=x.device, dtype=x.dtype)]  # x^0
        if max_degree > 0:
            powers.append(x)  # x^1
        for d in range(2, max_degree + 1):
            powers.append(powers[-1] * x)  # x^d
        X = torch.stack(powers, dim=2)  # (B, I, max_degree + 1)

        # ---------------------------------------------------------
        # Denominator Q(x) = 1 + |Σ_{j=1}^{q} b_j * x^j|
        # ---------------------------------------------------------
        # X_q: powers x^1 through x^q -> (B, I, degree_q)
        X_q = X[:, :, 1:self.degree_q + 1]
        # denominator_coeffs: (I, degree_q)
        Q_sum = torch.einsum('bik,ik->bi', X_q, self.denominator_coeffs)  # (B, I)
        Q = 1.0 + torch.abs(Q_sum)  # (B, I), always > 0

        # Cache Q for pole regularization
        self._cached_Q = Q

        # ---------------------------------------------------------
        # Numerator P(x) = Σ_{i=0}^{p} a_i * x^i  (per output dim)
        # ---------------------------------------------------------
        # X_p: powers x^0 through x^p -> (B, I, degree_p + 1)
        X_p = X[:, :, :self.degree_p + 1]

        # ---------------------------------------------------------
        # Rational basis: R(x) = X_p / Q(x)
        # ---------------------------------------------------------
        R = X_p / Q.unsqueeze(-1)  # (B, I, degree_p + 1)

        # Flatten R for linear projection
        R_flat = R.reshape(B, I * (self.degree_p + 1))

        # Merge scaling w into numerator coefficients
        # effective = numerator_coeffs * w.unsqueeze(-1)  -> (I, O, p+1)
        effective_coeffs = self.numerator_coeffs * self.w.unsqueeze(-1)

        # Flatten to weight matrix: (O, I * (p+1))
        weight_matrix = effective_coeffs.permute(1, 0, 2).reshape(
            self.output_dim, I * (self.degree_p + 1)
        )

        # Linear projection: y = R_flat @ weight_matrix.T -> (B, O)
        y = F.linear(R_flat, weight_matrix)

        return y

    def pole_regularization(self):
        """
        Compute pole regularization loss: L_pole = -mean(log(Q))
        
        Encourages Q(x) to be large, pushing poles away from the input
        distribution. This stabilizes training and improves AUC by preventing
        sharp/unstable rational function responses near data points.
        
        Returns 0 if forward() has not been called yet.
        """
        if self._cached_Q is None:
            return torch.tensor(0.0)
        # -mean(log(Q)): lower Q -> higher penalty
        return -torch.mean(torch.log(self._cached_Q))


class RKAN(nn.Module):
    """
    RationalKAN: Multi-layer Rational KAN.

    Drop-in replacement for ORKAN/OKAN/FasterKAN/FMKAN with the same interface:
        RKAN([input_dim, hidden_dim, output_dim])

    Args:
        layers_hidden: List of layer dimensions, e.g. [64, 192, 64]
        degree_p: Numerator polynomial degree (default: 5)
        degree_q: Denominator polynomial degree (default: 4)
    """
    def __init__(self, layers_hidden, degree_p=5, degree_q=4):
        super(RKAN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers_hidden) - 1):
            self.layers.append(
                RationalKANLayer(
                    layers_hidden[i],
                    layers_hidden[i + 1],
                    degree_p=degree_p,
                    degree_q=degree_q
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def pole_regularization(self):
        """
        Aggregate pole regularization loss across all layers.
        """
        total = torch.tensor(0.0)
        for layer in self.layers:
            reg = layer.pole_regularization()
            if reg.device != total.device:
                total = total.to(reg.device)
            total = total + reg
        return total / len(self.layers)
