"""Examples demonstrating the usage of sparse matrix functionality."""

import numpy as np
from strand.graphs import SparseAdjacencyMatrix
from strand.graphs.utils import tensor_inv, sparse_diagonal


def main():
    """Run examples of sparse matrix operations."""
    # Create a simple adjacency matrix
    adj_data = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ])
    
    adj = SparseAdjacencyMatrix(adj_data)
    print("Adjacency matrix:")
    print(adj.to_dense())
    
    # Create weights for edges
    weights = np.array([
        [0, 0.5, 0, 0.3],
        [0.5, 0, 0.2, 0],
        [0, 0.2, 0, 0.4],
        [0.3, 0, 0.4, 0]
    ])
    
    weighted_adj = SparseAdjacencyMatrix(weights)
    print("\nWeighted adjacency matrix:")
    print(weighted_adj.to_dense())
    
    # Demonstrate matrix operations
    result = adj.multiply(weighted_adj)
    print("\nElement-wise multiplication result:")
    print(result.to_dense())
    
    # Create diagonal degree matrix
    degrees = np.sum(adj_data, axis=1)
    degree_matrix = sparse_diagonal(degrees)
    print("\nDegree matrix:")
    print(degree_matrix.toarray())


if __name__ == "__main__":
    main() 