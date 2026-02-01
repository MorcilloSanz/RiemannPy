import math
import numpy as np

from .manifold import *

class ScalarField:
    """
    Represents a scalar field defined on a manifold.
    
    It computes:

        - gradient: approximation of the surface gradient at each point.
        - gradient_norm: approximation of the norm of the surface gradient by computing the Dirichlet energy at each point.
        - laplacian: approximation of the Laplace-Beltrami operator at each point.
    """
    def __init__(self, manifold: Manifold, values: np.ndarray) -> None:
        self.manifold = manifold
        self.values = values
        
        if self.values.shape[0] != self.manifold.points.shape[0]:
            assert 'The shape of the values does not match the domain shape'
        
    def _init_scalar_field(self) -> None:
        """
        Computes the gradient, gradient norm and Laplace-Beltrami operators
        """
        self._gradient: list[np.ndarray] = [] # Each node may have a different number of edges
        self._gradient_norm = np.zeros((self.manifold.points.shape[0]))
        self._laplacian = np.zeros((self.manifold.points.shape[0]))
        
        for i, p in enumerate(self.manifold.points):
            
            distances, indices = self.manifold.tree.query(p, k=self.manifold.k + 1)
            neighborhood = self.manifold.points[indices]
            
            gradient_edges = np.zeros((neighborhood.shape[0]))
            sum_laplacian, sum_gradient_norm = 0.0, 0.0
            
            for n, p_j in enumerate(neighborhood):

                j = indices[n]
                w_ij = distances[n]

                gradient_edges[n] = math.sqrt(w_ij) * (self.values[j] - self.values[i])
                sum_gradient_norm += w_ij * np.power(self.values[j] - self.values[i], 2)
                sum_laplacian += w_ij * (self.values[i] - self.values[j])

            self._gradient.append(gradient_edges)
            self._gradient_norm[i] = np.sqrt(0.5 * sum_gradient_norm)
            self._laplacian[i] = sum_laplacian
            
    @property
    def gradient(self) -> list[np.ndarray]:
        """
        Computes the gradient at each point and returns it. Since the gradient is defined
        in the edges and each node may have a different number of edges, the shapes may 
        not coincide so the gradient is given by a list that contains the variation of 
        the scalar field at each edge.
        
        Returns:
            list[np.ndarray]: the gradient at each point of the manifold.
        """
        self._init_scalar_field()
        return self._gradient
    
    @property
    def gradient_norm(self) -> np.ndarray:
        """
        Computes the gradient norm at each point and returns it.

        Returns:
            np.ndarray: the gradient norm at each point.
        """
        self._init_scalar_field()
        return self._gradient_norm
            
    @property
    def laplacian(self) -> np.ndarray:
        """
        Computes the Laplace-Beltrami operator at each point and returns it.

        Returns:
            np.ndarray: the Laplace-Beltrami at each point.
        """
        self._init_scalar_field()
        return self._laplacian