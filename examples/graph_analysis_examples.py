"""Examples demonstrating advanced graph analysis features."""

import torch
import matplotlib.pyplot as plt
from strand.graphs import SparseAdjacencyMatrix


def create_example_graph():
    """Create an example graph with two components."""
    # Create two triangles connected by a single edge
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 0, 3, 4, 4, 5, 5, 3, 2, 3],  # Source nodes
        [1, 2, 0, 0, 1, 2, 4, 5, 3, 3, 4, 5, 3, 2]   # Target nodes
    ])
    return SparseAdjacencyMatrix(edge_index)


def component_analysis_example():
    """Demonstrate connected component analysis."""
    print("\n=== Component Analysis Example ===")
    
    # Create a graph with multiple components
    adj = create_example_graph()
    print(f"Created graph: {adj}")
    
    # Find all components
    components, largest = adj.find_connected_components()
    print(f"\nFound {len(components)} components")
    print("Component sizes:", [len(comp) for comp in components])
    print("Largest component nodes:", largest.tolist())
    
    # Get component sizes
    sizes = adj.get_component_sizes()
    print("\nComponent sizes (sorted):", sizes.tolist())
    
    # Check connectivity
    print("\nIs graph connected?", adj.is_connected())
    
    # Extract largest component
    largest_comp = adj.get_largest_component()
    print(f"\nLargest component: {largest_comp}")
    
    # Visualize original and largest component
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    adj.visualize(title="Full Graph")
    
    plt.subplot(122)
    largest_comp.visualize(title="Largest Component")
    
    plt.tight_layout()
    plt.show()


def spectral_analysis_example():
    """Demonstrate spectral embedding and visualization."""
    print("\n=== Spectral Analysis Example ===")
    
    # Create a larger graph for better visualization
    n = 50  # number of nodes
    # Create a random sparse graph
    edge_prob = 0.1
    rand_adj = torch.rand(n, n) < edge_prob
    rand_adj = rand_adj & ~torch.eye(n).bool()  # Remove self-loops
    rand_adj = rand_adj | rand_adj.t()  # Make symmetric
    edge_index = rand_adj.nonzero().t()
    
    adj = SparseAdjacencyMatrix(edge_index)
    print(f"\nCreated random graph: {adj}")
    
    # Compute spectral embeddings
    print("\nComputing spectral embeddings...")
    emb_2d = adj.spectral_embedding(dim=2, normalized=True)
    emb_3d = adj.spectral_embedding(dim=3, normalized=True)
    
    # Visualize using built-in method
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    adj.visualize_spectral(dim=2, title="2D Spectral Embedding")
    
    plt.subplot(122)
    adj.visualize_spectral(dim=3, title="3D Spectral Embedding")
    
    plt.tight_layout()
    plt.show()
    
    # Example: Color nodes by degree
    degrees = adj.get_out_degrees()
    plt.figure(figsize=(12, 5))
    adj.visualize_spectral(
        dim=2, 
        color_by=degrees,
        title="2D Embedding (colored by degree)"
    )
    plt.show()


def main():
    """Run all examples."""
    component_analysis_example()
    spectral_analysis_example()


if __name__ == "__main__":
    main() 