"""Tests for graph utility functions."""

import numpy as np
import pytest
from scipy import sparse

from strand.graphs.utils import tensor_inv, sparse_diagonal


def test_tensor_inv():
    """Test tensor inverse function."""
    # Test with dense array
    x = np.array([[2.0, 0.0], [4.0, 5.0]])
    expected = np.array([[0.5, 0.0], [0.25, 0.2]])
    result = tensor_inv(x)
    assert np.allclose(result, expected)
    
    # Test with sparse matrix
    x_sparse = sparse.csr_matrix(x)
    result_sparse = tensor_inv(x_sparse)
    assert sparse.issparse(result_sparse)
    assert np.allclose(result_sparse.toarray(), expected)


def test_sparse_diagonal():
    """Test sparse diagonal matrix creation."""
    diag = np.array([1, 2, 3])
    result = sparse_diagonal(diag)
    assert sparse.issparse(result)
    expected = np.diag(diag)
    assert np.array_equal(result.toarray(), expected) 