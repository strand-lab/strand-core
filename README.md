# Strand Core

A Python library for efficient graph computations and machine learning.

## Features

- Efficient sparse matrix operations optimized for graph computations
- Helper utilities for graph algorithms
- Visualization tools (coming soon)
- Machine learning utilities (coming soon)

## Installation

### From Source (Development)
```bash
git clone https://github.com/strand-lab/strand-core.git
cd strand-core
pip install -e .
```

### From PyPI (Coming Soon)
```bash
pip install strand  # Changed from strand-core
```

## Quick Start

```python
import numpy as np
from strand.graphs import SparseAdjacencyMatrix

# Create an adjacency matrix
adj_data = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
])
adj = SparseAdjacencyMatrix(adj_data)

# Perform operations
result = adj.multiply(adj)
print(result.to_dense())
```

## Development

1. Clone the repository:
```bash
git clone https://github.com/strand-lab/strand-core.git
cd strand-core
```

2. Install development dependencies:
```bash
pip install -e ".[test]"
```

3. Run tests:
```bash
pytest
```

## License

MIT License