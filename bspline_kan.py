import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt


# ---------------------------------------------------------------------------
# Internal: stateless B-spline chunk computation (used with gradient ckpt)
# ---------------------------------------------------------------------------
def _spline_chunk(xc, grid, coeff, base_weight, spline_scale, spline_order):
    """
    Compute B-spline KAN output for a spatial chunk.
    xc          : (chunk, C)
    grid        : (G,) — extended knot grid
    coeff       : (C, num_bases)
    base_weight : (C,)
    spline_scale: (C,)
    Returns     : (chunk, C)
    """
    xg = xc.unsqueeze(-1)  # (chunk, C, 1)

    # Degree-0 indicator basis
    bases = ((xg >= grid[:-1]) & (xg < grid[1:])).to(xc.dtype)

    # Cox–de Boor recursion
    for k in range(1, spline_order + 1):
        left  = (xg - grid[:-(k + 1)]) / (grid[k:-1]   - grid[:-(k + 1)] + 1e-8)
        right = (grid[k + 1:] - xg)    / (grid[k + 1:] - grid[1:(-k)]    + 1e-8)
        bases = left * bases[..., :-1] + right * bases[..., 1:]

    base   = F.silu(xc) * base_weight                    # (chunk, C)
    spline = (bases * coeff).sum(-1) * spline_scale       # (chunk, C)
    return base + spline


class BSplineKAN(nn.Module):
    """
    B-Spline KAN for spatial feature maps (B, C, H, W).
    Drop-in replacement for ACKAN in LocalityFeedForward.

    Memory strategy
    ---------------
    The intermediate tensor `bases` has shape (chunk, C, num_bases).
    Chunking the spatial dimension keeps *peak* usage bounded.
    During training, gradient-checkpoint is applied per chunk so that
    intermediate activations are recomputed on the backward pass instead
    of being stored — trading compute for memory.

    Parameters
    ----------
    spatial_chunk : int
        Number of spatial locations per chunk (default 1024).
        Smaller → less VRAM, more kernel launches.
    use_ckpt : bool
        Enable gradient checkpointing per chunk (default True).
    """
    def __init__(self, channels, grid_size=5, spline_order=3,
                 grid_range=(-1.0, 1.0), kernel_size=3,
                 spatial_chunk=1024, use_ckpt=True):
        super().__init__()
        self.channels = channels
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.kernel_size = kernel_size
        self.spatial_chunk = spatial_chunk
        self.use_ckpt = use_ckpt

        # Extended knot grid
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.linspace(
            grid_range[0] - spline_order * h,
            grid_range[1] + spline_order * h,
            grid_size + 2 * spline_order + 1,
        )
        self.register_buffer('grid', grid)

        num_bases = grid_size + spline_order
        self.coeff        = nn.Parameter(torch.zeros(channels, num_bases))
        self.base_weight  = nn.Parameter(torch.ones(channels))
        self.spline_scale = nn.Parameter(torch.full((channels,), 0.1))

        # if kernel_size > 1:
        #     self.dw_conv = nn.Conv2d(
        #         channels, channels,
        #         kernel_size=kernel_size,
        #         padding=kernel_size // 2,
        #         groups=channels,
        #         bias=False,
        #     )
        # else:
        #     self.dw_conv = None
        self.dw_conv = None

        self._init_params()

    def _init_params(self):
        nn.init.normal_(self.coeff, mean=0.0, std=0.1)
        nn.init.ones_(self.base_weight)
        nn.init.constant_(self.spline_scale, 0.1)
        if self.dw_conv is not None:
            nn.init.kaiming_uniform_(self.dw_conv.weight, a=math.sqrt(5))

    def _process_chunk(self, xc):
        """Wrapper so gradient checkpoint can be applied."""
        return _spline_chunk(
            xc, self.grid, self.coeff,
            self.base_weight, self.spline_scale, self.spline_order,
        )

    def forward(self, x):
        B, C, H, W = x.shape

        if self.dw_conv is not None:
            x = self.dw_conv(x)

        # (B, C, H, W) → (N, C)
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
        N = x_flat.shape[0]

        chunks = []
        for i in range(0, N, self.spatial_chunk):
            xc = x_flat[i : i + self.spatial_chunk]  # (chunk, C)
            if self.use_ckpt and self.training:
                out_c = ckpt.checkpoint(self._process_chunk, xc, use_reentrant=False)
            else:
                out_c = self._process_chunk(xc)
            chunks.append(out_c)

        out = torch.cat(chunks, dim=0)                          # (N, C)
        return out.reshape(B, H, W, C).permute(0, 3, 1, 2)     # (B, C, H, W)
