
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class OrthogonalRationalKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree_m=5, degree_n=4, order=3):
        super(OrthogonalRationalKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree_m = degree_m  # Numerator degree
        self.degree_n = degree_n  # Denominator degree
        
        self.layernorm = nn.LayerNorm(input_dim)
        
        # Initialize coefficients for numerator (alpha) and denominator (beta)
        # alpha: (input_dim, output_dim, degree_m + 1)
        # beta: (input_dim, degree_n + 1) -> Shared Denominator per input dimension
        self.alpha_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree_m + 1))
        self.beta_coeffs = nn.Parameter(torch.empty(input_dim, degree_n + 1))
        
        # Scaling factor w
        self.w = nn.Parameter(torch.ones(input_dim, output_dim))

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_normal_(self.alpha_coeffs)
        nn.init.normal_(self.beta_coeffs, std=0.1) # Initialize beta small
        nn.init.normal_(self.w, mean=1.0, std=0.1)

    def chebyshev_polynomials(self, x, degree):
        # x: (batch_size, input_dim) - normalized to [-1, 1]
        # Returns: (batch_size, input_dim, degree + 1)
        B, I = x.shape
        T_list = [torch.ones(B, I, device=x.device, dtype=x.dtype)] # U_0(x) = 1
        if degree > 0:
            T_list.append(2 * x) # U_1(x) = 2x (Chebyshev 2nd Kind)
        for i in range(2, degree + 1):
            T_next = 2 * x * T_list[-1] - T_list[-2] # U_n(x) = 2xU_{n-1}(x) - U_{n-2}(x)
            T_list.append(T_next)
        T = torch.stack(T_list, dim=2)
        return T

    def forward(self, x):
        # x: (batch_size, input_dim)
        B, I = x.shape
        # Normalize x to [-1, 1] using tanh
        x = self.layernorm(x)
        x = torch.tanh(x) 

        # Generate Chebyshev polynomials T_n(x)
        # T shape: (B, I, max_degree + 1)
        max_degree = max(self.degree_m, self.degree_n)
        T = self.chebyshev_polynomials(x, max_degree)
        
        # ---------------------------------------------------------
        # Shared Denominator Architecture (FastORKAN)
        # ---------------------------------------------------------
        
        # Denominator Q(x)
        # beta: (I, degree_n + 1)
        # T_n: (B, I, degree_n + 1)
        # Q_prime = sum(beta * T) -> (B, I)
        T_n = T[:, :, :self.degree_n + 1]
        Q_prime = torch.einsum('bik,ik->bi', T_n, self.beta_coeffs)
        Q = 1 + torch.abs(Q_prime) # (B, I)
        
        # Numerator Basis: T_m(x)
        # T_m: (B, I, degree_m + 1)
        T_m = T[:, :, :self.degree_m + 1]
        
        # Rational Basis R(x) = T_m(x) / Q(x)
        # R: (B, I, degree_m + 1)
        R = T_m / Q.unsqueeze(-1)
        
        # Flatten R to (B, I * (m+1)) for Linear projection
        R_flat = R.reshape(B, I * (self.degree_m + 1))
        
        # Prepare weights for Linear projection
        # alpha: (I, O, m+1) -> permute to (O, I, m+1) -> reshape to (O, I * (m+1))
        # w: (I, O) -> permute (O, I) -> expand and reshape
        
        output_dim = self.output_dim
        m_plus_1 = self.degree_m + 1
        
        # Merge w into alpha for efficient computation: w * alpha
        # w: (I, O)
        # alpha: (I, O, m+1)
        # effective_alpha = alpha * w.unsqueeze(-1) -> (I, O, m+1)
        effective_alpha = self.alpha_coeffs * self.w.unsqueeze(-1)
        
        # Flatten effective_alpha to (O, I*(m+1))
        weight_matrix = effective_alpha.permute(1, 0, 2).reshape(output_dim, I * m_plus_1)
        
        # Linear projection: y = R_flat @ weight_matrix.T
        # y: (B, O)
        y = F.linear(R_flat, weight_matrix)
        
        return y

class ORKAN(nn.Module):
    def __init__(self, layers_hidden, degree_m=10, degree_n=8):
        super(ORKAN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers_hidden) - 1):
            self.layers.append(
                OrthogonalRationalKANLayer(
                    layers_hidden[i],
                    layers_hidden[i+1],
                    degree_m=degree_m,
                    degree_n=degree_n
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class OrthogonalKANLayer(nn.Module):
    """
    Orthogonal KAN Layer using only Chebyshev polynomials (no Rational denominator).
    Compared to OrthogonalRationalKANLayer, the denominator Q(x) is removed,
    so the basis functions are purely T_m(x) (Chebyshev polynomials of the 2nd kind).
    """
    def __init__(self, input_dim, output_dim, degree=11):
        super(OrthogonalKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree

        self.layernorm = nn.LayerNorm(input_dim)

        # Coefficients for Chebyshev basis: alpha
        # alpha: (input_dim, output_dim, degree + 1)
        self.alpha_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))

        # Scaling factor w
        self.w = nn.Parameter(torch.ones(input_dim, output_dim))

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_normal_(self.alpha_coeffs)
        nn.init.normal_(self.w, mean=1.0, std=0.1)

    def chebyshev_polynomials(self, x, degree):
        # x: (batch_size, input_dim) - normalized to [-1, 1]
        # Returns: (batch_size, input_dim, degree + 1)
        B, I = x.shape
        T_list = [torch.ones(B, I, device=x.device, dtype=x.dtype)]  # U_0(x) = 1
        if degree > 0:
            T_list.append(2 * x)  # U_1(x) = 2x
        for i in range(2, degree + 1):
            T_next = 2 * x * T_list[-1] - T_list[-2]  # U_n(x) = 2xU_{n-1}(x) - U_{n-2}(x)
            T_list.append(T_next)
        T = torch.stack(T_list, dim=2)
        return T

    def forward(self, x):
        # x: (batch_size, input_dim)
        B, I = x.shape
        # Normalize x to [-1, 1] using tanh
        x = self.layernorm(x)
        x = torch.tanh(x)

        # Generate Chebyshev polynomials T(x)
        # T shape: (B, I, degree + 1)
        T = self.chebyshev_polynomials(x, self.degree)

        # No denominator Q(x) — use T directly as basis
        # T: (B, I, degree + 1)

        # Flatten T to (B, I * (degree+1)) for Linear projection
        T_flat = T.reshape(B, I * (self.degree + 1))

        # Merge w into alpha for efficient computation: w * alpha
        # effective_alpha: (I, O, degree+1)
        effective_alpha = self.alpha_coeffs * self.w.unsqueeze(-1)

        # Flatten effective_alpha to (O, I*(degree+1))
        weight_matrix = effective_alpha.permute(1, 0, 2).reshape(self.output_dim, I * (self.degree + 1))

        # Linear projection: y = T_flat @ weight_matrix.T
        # y: (B, O)
        y = F.linear(T_flat, weight_matrix)

        return y


class OKAN(nn.Module):
    """Orthogonal KAN (no Rational denominator) multi-layer wrapper."""
    def __init__(self, layers_hidden, degree=10):
        super(OKAN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers_hidden) - 1):
            self.layers.append(
                OrthogonalKANLayer(
                    layers_hidden[i],
                    layers_hidden[i+1],
                    degree=degree
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
