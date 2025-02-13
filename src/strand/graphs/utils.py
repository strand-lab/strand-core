"""Utility functions for graph computations."""

import torch


def tensor_inv(x):
    """Compute element-wise inverse, handling zeros.
    
    Args:
        x: torch.Tensor - Input tensor
        
    Returns:
        torch.Tensor - Element-wise inverse with zeros preserved
    """
    mask = (x != 0)
    xinv = torch.ones_like(x)
    xinv[mask] = 1/x[mask]
    return xinv


def sparse_diagonal(values):
    """Create a sparse diagonal matrix from a 1D tensor of values.
    
    Args:
        values: torch.Tensor - 1D tensor of diagonal values
        
    Returns:
        torch.sparse_coo_tensor - Sparse diagonal matrix
    """
    indices = torch.arange(len(values), device=values.device)
    indices = torch.stack([indices, indices])
    size = (len(values), len(values))
    return torch.sparse_coo_tensor(indices, values, size, device=values.device) 