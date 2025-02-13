"""Sparse matrix implementations for efficient graph computations."""

import torch
from ..graphs.utils import tensor_inv, sparse_diagonal


class SparseAdjacencyMatrix:
    """A class to handle sparse adjacency matrices for graph operations.
    Internally stores data as a sparse tensor for efficiency.
    """
    
    def __init__(self, data, num_nodes=None, device=None):
        """Initialize from either edge list or sparse tensor.
        
        Args:
            data: Either torch.Tensor edge_index (2 x E) or torch.sparse_coo_tensor
            num_nodes: Optional, number of nodes. Inferred if not provided
            device: Optional, device to store tensors on
        """
        self.device = device if device is not None else torch.device('cpu')
        
        if isinstance(data, torch.Tensor) and data.size(0) == 2:  # Edge list format
            edge_index = data.to(self.device)
            
            # Infer num_nodes if not provided
            if num_nodes is None:
                num_nodes = int(edge_index.max().item() + 1) if edge_index.numel() > 0 else 0
            self.num_nodes = num_nodes
            
            # Convert to sparse tensor
            values = torch.ones(edge_index.size(1), device=self.device)
            self._sparse_tensor = torch.sparse_coo_tensor(
                edge_index, 
                values,
                size=(self.num_nodes, self.num_nodes)
            ).coalesce()
            
        elif torch.is_tensor(data) and data.is_sparse:  # Sparse tensor format
            self._sparse_tensor = data.coalesce().to(self.device)
            self.num_nodes = self._sparse_tensor.size(0)
            
        else:
            raise ValueError("Input must be either edge_index tensor (2 x E) or sparse tensor")
    
    @property
    def edge_index(self):
        """Get edge_index representation (2 x E tensor)"""
        return self._sparse_tensor.indices()
    
    def normalize(self):
        """Return degree-normalized version using D_out^(-1/2) A D_in^(-1/2) normalization"""
        # Calculate normalized degree matrices
        d_in = torch.sqrt(self.get_in_degrees())
        d_out = torch.sqrt(self.get_out_degrees())
        
        # Create diagonal normalization matrices
        D_in_inv = sparse_diagonal(tensor_inv(d_in))
        D_out_inv = sparse_diagonal(tensor_inv(d_out))
        
        # Normalize adjacency: D_out^(-1/2) A D_in^(-1/2)
        norm_adj = D_out_inv @ self._sparse_tensor @ D_in_inv
        
        return SparseAdjacencyMatrix(norm_adj, self.num_nodes, self.device)
    
    def subsample_nodes(self, node_indices, sort=True):
        """Create new SparseAdjacencyMatrix with only specified nodes.
        
        Args:
            node_indices: Indices of nodes to keep
            sort: If True, sort node_indices for consistent node ordering
        """
        if sort:
            node_indices = torch.sort(node_indices)[0]
            
        # Create mask of valid nodes
        node_mask = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)
        node_mask[node_indices] = True
        
        # Find edges where both endpoints are in sampled nodes
        edge_mask = node_mask[self.edge_index[0]] & node_mask[self.edge_index[1]]
        
        # Create new edges with remapped indices
        mapping = -torch.ones(self.num_nodes, dtype=torch.long, device=self.device)
        mapping[node_indices] = torch.arange(len(node_indices), device=self.device)
        new_edges = mapping[self.edge_index[:, edge_mask]]
        
        return SparseAdjacencyMatrix(new_edges, len(node_indices), self.device)
    
    def to_dense(self):
        """Convert to dense tensor representation"""
        return self._sparse_tensor.to_dense()
    
    def get_in_degrees(self):
        """Get in-degrees of all nodes"""
        ones = torch.ones(self.num_nodes, device=self.device)
        return self._sparse_tensor.t() @ ones
        
    def get_out_degrees(self):
        """Get out-degrees of all nodes"""
        ones = torch.ones(self.num_nodes, device=self.device)
        return self._sparse_tensor @ ones
        
    def is_symmetric(self):
        """Check if the adjacency matrix is symmetric"""
        A = self._sparse_tensor.coalesce()
        At = A.t().coalesce()
        return torch.equal(A.indices(), At.indices()) and torch.equal(A.values(), At.values())
        
    def add_self_loops(self):
        """Add self-loops to the adjacency matrix"""
        diag = sparse_diagonal(torch.ones(self.num_nodes, device=self.device))
        new_adj = self._sparse_tensor + diag
        return SparseAdjacencyMatrix(new_adj, self.num_nodes, self.device)
        
    def remove_self_loops(self):
        """Remove self-loops from the adjacency matrix"""
        indices = self._sparse_tensor.indices()
        values = self._sparse_tensor.values()
        mask = indices[0] != indices[1]
        new_adj = torch.sparse_coo_tensor(
            indices[:, mask], 
            values[mask], 
            self._sparse_tensor.size()
        ).coalesce()
        return SparseAdjacencyMatrix(new_adj, self.num_nodes, self.device)
    
    def to_sparse_tensor(self):
        """Return sparse tensor representation"""
        return self._sparse_tensor
    
    def to(self, device):
        """Move adjacency matrix to specified device"""
        if device == self.device:
            return self
        return SparseAdjacencyMatrix(self._sparse_tensor, self.num_nodes, device)
    
    @property
    def device(self):
        """Get current device"""
        return self._device
    
    @device.setter 
    def device(self, device):
        """Set device"""
        self._device = device

    def __matmul__(self, other):
        """Support matrix multiplication with @ operator"""
        if isinstance(other, SparseAdjacencyMatrix):
            return self._sparse_tensor @ other._sparse_tensor
        elif isinstance(other, torch.Tensor):
            return self._sparse_tensor @ other
        else:
            raise TypeError(f"Unsupported operand type for @: {type(other)}")

    def __rmatmul__(self, other):
        """Support right multiplication with @ operator"""
        if isinstance(other, torch.Tensor):
            return other @ self._sparse_tensor
        else:
            raise TypeError(f"Unsupported operand type for @: {type(other)}")

    @property
    def T(self):
        """Support transpose operation"""
        return SparseAdjacencyMatrix(self._sparse_tensor.t(), self.num_nodes, self.device)
    
    def t(self):
        """Support transpose operation (alias for .T)"""
        return self.T
    
    def find_connected_components(self):
        """Find all connected components in the graph using iterative depth-first search.
        Works with both directed (strongly connected) and undirected graphs.
        
        Returns:
            components (list): List of torch.Tensor containing node indices for each component
            largest_component (torch.Tensor): Node indices of the largest component
        """
        # For directed graphs, we need to consider both directions
        if not self.is_symmetric():
            # Create undirected version by adding transpose
            undirected = self._sparse_tensor + self._sparse_tensor.t()
            edges = undirected.coalesce().indices()
        else:
            edges = self.edge_index

        # Create adjacency list representation for faster traversal
        adj_list = [[] for _ in range(self.num_nodes)]
        for i in range(edges.size(1)):
            src, dst = edges[:, i]
            adj_list[src.item()].append(dst.item())

        # Iterative DFS
        def iterative_dfs(start_node, visited):
            component = []
            stack = [start_node]
            
            while stack:
                node = stack.pop()
                if not visited[node]:
                    visited[node] = True
                    component.append(node)
                    # Add unvisited neighbors to stack
                    stack.extend(n for n in adj_list[node] if not visited[n])
            
            return component

        # Find all components
        visited = [False] * self.num_nodes
        components = []
        
        for node in range(self.num_nodes):
            if not visited[node]:
                component = iterative_dfs(node, visited)
                components.append(torch.tensor(component, device=self.device))

        # Find largest component
        largest_component = max(components, key=len)
        
        return components, largest_component

    def get_largest_component(self):
        """Return a new SparseAdjacencyMatrix containing only the largest component.
        
        Returns:
            SparseAdjacencyMatrix: Subgraph of the largest connected component
        """
        _, largest_component = self.find_connected_components()
        return self.subsample_nodes(largest_component)

    def get_component_sizes(self):
        """Get the sizes of all connected components.
        
        Returns:
            torch.Tensor: Sorted tensor of component sizes (descending order)
        """
        components, _ = self.find_connected_components()
        sizes = torch.tensor([len(comp) for comp in components], device=self.device)
        return torch.sort(sizes, descending=True)[0]

    def is_connected(self):
        """Check if the graph is connected (single component).
        
        Returns:
            bool: True if graph is connected, False otherwise
        """
        components, _ = self.find_connected_components()
        return len(components) == 1
    
    def spectral_embedding(self, dim=2, normalized=True):
        """Compute spectral embedding of the graph using top eigenvectors.
        
        Args:
            dim (int): Dimension of embedding (2 or 3)
            normalized (bool): Whether to use normalized adjacency matrix
        
        Returns:
            torch.Tensor: Node embeddings of shape (num_nodes, dim)
        """
        import scipy.sparse as sp
        import scipy.sparse.linalg as splinalg
        
        # Get symmetric version if not already symmetric
        if not self.is_symmetric():
            adj = self._sparse_tensor + self._sparse_tensor.t()
        else:
            adj = self._sparse_tensor
        
        # Convert to scipy sparse format
        indices = adj.indices().cpu().numpy()
        values = adj.values().cpu().numpy()
        adj_scipy = sp.coo_matrix(
            (values, (indices[0], indices[1])),
            shape=(self.num_nodes, self.num_nodes)
        ).tocsr()
        
        if normalized:
            # Compute D^(-1/2) A D^(-1/2) normalization
            degrees = torch.sparse.sum(adj, dim=1).to_dense()
            deg_sqrt_inv = torch.where(
                degrees > 0, 
                degrees.pow(-0.5), 
                torch.zeros_like(degrees)
            )
            
            # Convert to scipy diagonal matrix
            D_sqrt_inv = sp.diags(deg_sqrt_inv.cpu().numpy())
            adj_norm = D_sqrt_inv @ adj_scipy @ D_sqrt_inv
        else:
            adj_norm = adj_scipy
        
        # Compute top eigenvectors using scipy's eigsh
        # k=dim+1 because first eigenvector is usually constant
        try:
            eigenvalues, eigenvectors = splinalg.eigsh(
                adj_norm, k=dim+1, which='LA'
            )
        except splinalg.ArpackNoConvergence as e:
            # If no convergence, return partial results
            eigenvalues = e.eigenvalues
            eigenvectors = e.eigenvectors
            print("Warning: Eigendecomposition did not fully converge")
        
        # Sort by eigenvalue magnitude and remove first eigenvector
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Return as torch tensor
        return torch.from_numpy(eigenvectors[:, 1:dim+1]).float().to(self.device)

    def visualize(self, title="Graph Visualization", with_labels=True):
        """Visualize the graph using networkx.
        Handles both directed and undirected graphs based on symmetry.
        
        Args:
            title: Title for the plot
            with_labels: Whether to show node labels
        """
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
        except ImportError:
            print("networkx and matplotlib are required for visualization")
            return
            
        # Create directed or undirected graph based on symmetry
        if self.is_symmetric():
            G = nx.from_edgelist(self.edge_index.cpu().numpy().T)
        else:
            G = nx.from_edgelist(
                self.edge_index.cpu().numpy().T,
                create_using=nx.DiGraph
            )
            
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(
            G, pos, 
            with_labels=with_labels, 
            node_color='lightblue',
            node_size=500, 
            arrowsize=10 if not self.is_symmetric() else 5,
            font_size=10
        )
        plt.title(title)
        plt.show()

    def visualize_spectral(self, dim=2, normalized=True, color_by=None, 
                          sample_size=None):
        """Visualize graph using spectral embedding.
        
        Args:
            dim (int): Dimension of embedding (2 or 3)
            normalized (bool): Whether to use normalized adjacency matrix
            color_by (torch.Tensor, optional): Node values to use for coloring
            sample_size (int, optional): Number of nodes to sample for visualization
        """
        import matplotlib.pyplot as plt
        
        if dim not in [2, 3]:
            raise ValueError("Embedding dimension must be 2 or 3")
            
        # Compute embeddings
        embeddings = self.spectral_embedding(dim, normalized)
        
        # Sample nodes if requested
        if sample_size is not None and sample_size < self.num_nodes:
            idx = torch.randperm(self.num_nodes)[:sample_size]
            embeddings = embeddings[idx]
            if color_by is not None:
                color_by = color_by[idx]
        
        # Convert to numpy for plotting
        embeddings = embeddings.cpu().numpy()
        
        # Set up colors
        if color_by is not None:
            colors = color_by.cpu().numpy()
        else:
            # Default coloring by local density
            from sklearn.neighbors import KernelDensity
            kde = KernelDensity(kernel='gaussian', bandwidth=0.1)
            kde.fit(embeddings)
            colors = kde.score_samples(embeddings)
        
        # Create figure
        if dim == 2:
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(
                embeddings[:, 0], 
                embeddings[:, 1],
                c=colors, 
                cmap='viridis', 
                alpha=0.6,
                s=50
            )
            plt.colorbar(scatter)
            plt.title('2D Spectral Embedding' + 
                     (' (Normalized)' if normalized else ''))
            plt.xlabel('First eigenvector')
            plt.ylabel('Second eigenvector')
            
        else:  # dim == 3
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(
                embeddings[:, 0], 
                embeddings[:, 1], 
                embeddings[:, 2],
                c=colors, 
                cmap='viridis', 
                alpha=0.6,
                s=50
            )
            plt.colorbar(scatter)
            ax.set_title('3D Spectral Embedding' + 
                        (' (Normalized)' if normalized else ''))
            ax.set_xlabel('First eigenvector')
            ax.set_ylabel('Second eigenvector')
            ax.set_zlabel('Third eigenvector')
        
        plt.tight_layout()
        return plt.gcf()
    
    def __repr__(self):
        nnz = self.edge_index.size(1)  # number of non-zero elements
        density = nnz / (self.num_nodes ** 2)
        return (
            f"SparseAdjacencyMatrix(num_nodes={self.num_nodes}, "
            f"num_edges={nnz}, density={density:.4f}, "
            f"device={self.device})"
        ) 