import numpy as np
from .manifold import *


class ScalarField:
    """
    Represents a scalar field defined on the vertices of a manifold.

    This class implements discrete approximations of geometric differential 
    operators using a graph structure based on point neighborhoods and 
    Radial Basis Function (RBF) Gaussian kernels.

    Attributes:
        manifold (Manifold): Object containing the geometry, points, and 
            search structure (KDTree).
        values (np.ndarray): The scalar values of the field at each point.
        
        _gradient (list[np.ndarray]): Local field variations across edges.
        _gradient_norm (np.ndarray): Estimated magnitude of the gradient at each point.
        _laplacian (np.ndarray): Random Walk Graph Laplacian approximation.
        _laplace_beltrami (np.ndarray): Symmetric Normalized Laplacian approximation.
    """
    _gradient: list[np.ndarray]
    _gradient_norm: np.ndarray
    _laplacian: np.ndarray
    _laplace_beltrami: np.ndarray
    
    def __init__(self, manifold: Manifold, values: np.ndarray) -> None:
        """
        Initializes the scalar field and triggers the computation of operators.

        Args:
            manifold (Manifold): The manifold instance defining the domain.
            values (np.ndarray): A (N,) shaped array containing scalar values.
        """
        self.manifold = manifold
        self.values = values
        self.compute_differential_operators()
        
    def compute_differential_operators(self) -> None:
        """
        Computes discrete approximations of differential operators using graph weights.

        The computation follows three main stages:
        1. Construction of local weights using a Gaussian kernel: $w_{ij} = \exp(-d_{ij}^2 / \sigma_i^2)$.
        2. Calculation of local degree (density): $d_i = \sum_j w_{ij}$.
        3. Estimation of operators via weighted finite differences.

        Note:
            Uses an adaptive local $\sigma$ based on the mean distance to the $k$ 
            nearest neighbors for each point.
        """
        self._gradient = [] # Each node may have a different number of edges
        self._gradient_norm = np.zeros((self.manifold.points.shape[0]))
        self._laplacian = np.zeros((self.manifold.points.shape[0]))
        self._laplace_beltrami = np.zeros((self.manifold.points.shape[0]))
        
        degrees = np.zeros(self.manifold.points.shape[0])
        neighbor_data = []

        # Build weights
        for i, p_i in enumerate(self.manifold.points):

            distances, indices = self.manifold.tree.query(p_i, k=self.manifold.k + 1)

            distances = distances[1:]
            indices = indices[1:]

            sigma = np.mean(distances) + 1e-12
            weights = np.exp(-(distances**2) / (sigma**2))

            d_i = np.sum(weights) + 1e-12
            degrees[i] = d_i

            neighbor_data.append((indices, weights, d_i))
        
        # Compute operators
        for i, (indices, weights, d_i) in enumerate(neighbor_data):
            
            gradient_edges = np.zeros((len(indices)))
            sum_laplacian, sum_gradient_norm, sum_lb = 0.0, 0.0, 0.0
            
            for n, j in enumerate(indices):

                w_ij = weights[n]
                d_j = degrees[j]
                
                diff = self.values[j] - self.values[i]

                gradient_edges[n] = np.sqrt(w_ij) * diff
                sum_gradient_norm += w_ij * diff**2
                sum_laplacian += w_ij * diff
                sum_lb += (w_ij * self.values[j]) / np.sqrt(d_j)

            self._gradient.append(gradient_edges)
            self._gradient_norm[i] = np.sqrt(0.5 * sum_gradient_norm)
            self._laplacian[i] = sum_laplacian / d_i
            self._laplace_beltrami[i] = -1 * (self.values[i] - (sum_lb / np.sqrt(d_i)))
            
    @property
    def gradient(self) -> list[np.ndarray]:
        """
        Computes the discrete edge-based gradient.

        Unlike a standard vector gradient in $\mathbb{R}^n$, this operator 
        returns the scalar variation projected onto each edge connected to 
        the point, weighted by the square root of the edge weight.

        Returns:
            list[np.ndarray]: A list where each element `i` is an array 
                containing the values $\sqrt{w_{ij}}(f_j - f_i)$ for neighbors $j$.
        """
        return self._gradient
    
    @property
    def gradient_norm(self) -> np.ndarray:
        """
        Estimates the surface gradient norm via local Dirichlet Energy.

        It is calculated as:
        $$\|\nabla f\|_i \approx \sqrt{\frac{1}{2} \sum_{j \in \mathcal{N}(i)} w_{ij} (f_j - f_i)^2}$$

        Returns:
            np.ndarray: An (N,) array representing the gradient magnitude at each point.
        """
        return self._gradient_norm
    
    @property
    def dirichlet_energy(self) -> np.ndarray:
        """
        Computes the local Dirichlet Energy of the scalar field at each point.

        The Dirichlet Energy measures how much the field varies across the 
        manifold. In its discrete form, it is defined as the square of the 
        gradient norm:
        
        $$E_D(f)_i = \|\nabla f\|_i^2 = \frac{1}{2} \sum_{j \in \mathcal{N}(i)} w_{ij} (f_j - f_i)^2$$

        This property provides a point-wise measure of the "roughness" of the 
        field. High values indicate sharp transitions or high-frequency 
        components, while low values indicate a smooth, nearly constant field.

        Returns:
            np.ndarray: An (N,) array containing the local Dirichlet Energy 
                for each point on the manifold.
        """
        return np.power(self._gradient_norm, 2)
            
    @property
    def laplacian(self) -> np.ndarray:
        """
        Computes the Random Walk Graph Laplacian.

        This operator is normalized by the node degree, making it 
        independent of local point density. It approximates the operator 
        $\Delta = I - D^{-1}W$.

        Returns:
            np.ndarray: The Laplacian value at each point. Positive values 
                indicate the point is a local minimum relative to its neighbors.
        """
        return self._laplacian
    
    @property
    def laplace_beltrami(self) -> np.ndarray:
        """
        Computes the Symmetric Normalized Laplacian.

        Implements the form $L_{sym} = I - D^{-1/2}WD^{-1/2}$. This version 
        is standard for spectral shape analysis and is robust against 
        sampling irregularities.

        Mathematically defined as:
        $$(L_{sym} f)_i = f_i - \frac{1}{\sqrt{d_i}} \sum_{j \in \mathcal{N}(i)} \frac{w_{ij} f_j}{\sqrt{d_j}}$$

        Returns:
            np.ndarray: The Symmetric Laplace-Beltrami operator value at each point.
        """
        return self._laplace_beltrami