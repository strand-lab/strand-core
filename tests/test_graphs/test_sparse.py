"""Tests for sparse matrix implementations."""

import numpy as np
import pytest
import torch

from strand.graphs import SparseAdjacencyMatrix


def test_init():
    """Test initialization of SparseAdjacencyMatrix."""
    # Test with edge list
    edge_index = torch.tensor([[0, 1, 1], [1, 0, 2]], dtype=torch.long)
    adj = SparseAdjacencyMatrix(edge_index)
    assert adj.num_nodes == 3
    assert adj.edge_index.size(1) == 3
    
    # Test with explicit num_nodes
    adj = SparseAdjacencyMatrix(edge_index, num_nodes=4)
    assert adj.num_nodes == 4
    
    # Test with device
    if torch.cuda.is_available():
        adj = SparseAdjacencyMatrix(edge_index, device='cuda')
        assert adj.device == torch.device('cuda')


def test_basic_operations():
    """Test basic matrix operations."""
    edge_index = torch.tensor([[0, 1], [1, 0]])
    adj = SparseAdjacencyMatrix(edge_index)
    
    # Test transpose
    assert adj.is_symmetric()
    assert torch.equal(adj.T.edge_index, adj.edge_index)
    
    # Test matmul
    x = torch.randn(2, 3)
    result = adj @ x
    assert result.shape == (2, 3)


def test_component_analysis():
    """Test connected component analysis."""
    # Create disconnected graph: two triangles
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 0, 3, 4, 4, 5, 5, 3],
        [1, 2, 0, 0, 1, 2, 4, 5, 3, 3, 4, 5]
    ])
    adj = SparseAdjacencyMatrix(edge_index)
    
    # Test component finding
    components, largest = adj.find_connected_components()
    assert len(components) == 2  # Two components
    assert len(largest) == 3  # Each component has 3 nodes
    
    # Test component sizes
    sizes = adj.get_component_sizes()
    assert torch.equal(sizes, torch.tensor([3, 3]))
    
    # Test connectivity check
    assert not adj.is_connected()
    
    # Test largest component extraction
    largest_comp = adj.get_largest_component()
    assert largest_comp.num_nodes == 3


def test_spectral_embedding():
    """Test spectral embedding computation."""
    # Create a simple graph
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 0],
        [1, 2, 0, 0, 1, 2]
    ])
    adj = SparseAdjacencyMatrix(edge_index)
    
    # Test 2D embedding
    emb_2d = adj.spectral_embedding(dim=2)
    assert emb_2d.shape == (3, 2)
    
    # Test 3D embedding
    emb_3d = adj.spectral_embedding(dim=3)
    assert emb_3d.shape == (3, 3)
    
    # Test normalized vs unnormalized
    emb_norm = adj.spectral_embedding(normalized=True)
    emb_unnorm = adj.spectral_embedding(normalized=False)
    assert not torch.allclose(emb_norm, emb_unnorm)


def test_visualization():
    """Test visualization methods (no display)."""
    import matplotlib.pyplot as plt
    
    # Create a simple graph
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 0],
        [1, 2, 0, 0, 1, 2]
    ])
    adj = SparseAdjacencyMatrix(edge_index)
    
    # Test basic visualization
    plt.close('all')
    fig = adj.visualize(with_labels=True)
    plt.close('all')
    
    # Test spectral visualization
    plt.close('all')
    fig = adj.visualize_spectral(dim=2)
    plt.close('all')
    
    # Test 3D spectral visualization
    plt.close('all')
    fig = adj.visualize_spectral(dim=3)
    plt.close('all')


def test_self_loops():
    """Test self-loop handling."""
    edge_index = torch.tensor([[0, 1], [1, 0]])
    adj = SparseAdjacencyMatrix(edge_index)
    
    # Add self loops
    adj_self = adj.add_self_loops()
    assert adj_self.edge_index.size(1) == 4  # Original edges + 2 self loops
    
    # Remove self loops
    adj_no_self = adj_self.remove_self_loops()
    assert torch.equal(adj_no_self.edge_index, adj.edge_index)


def test_multiply():
    """Test element-wise multiplication."""
    a = SparseAdjacencyMatrix([[1, 2], [3, 4]])
    b = SparseAdjacencyMatrix([[5, 6], [7, 8]])
    result = a.multiply(b)
    expected = np.array([[5, 12], [21, 32]])
    assert np.array_equal(result.to_dense(), expected) 