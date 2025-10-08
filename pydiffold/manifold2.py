import numpy as np
from numpy import linalg as LA
from scipy.spatial import KDTree
import networkx as nx


class Manifold:
    """
    Represents a 2D differentiable manifold embedded in 3D space,
    discretized by a set of sample points (typically originating from a mesh).
    
    Estimates local differential geometry (tangent and normal spaces),
    builds a connectivity graph between neighboring points,
    and enables geodesic path computation along the manifold.
    """
    
    __MIN_NEIGHBORHOOD: int = 3
    
    def __init__(self, points: np.array, k=8) -> None:
        """
        Initializes the manifold from a discrete set of points on the surface.

        Args:
            points (np.array): An (N x 3) array of 3D coordinates sampling the manifold.
            k(int): knn search for PCA.
        """
        self.points = points
        self.k = k
        
        self.tree = KDTree(points)
        self.graph: nx.Graph = nx.Graph()
        
        self.__compute_graph()
        
    def __compute_graph(self) -> None:
        """
        Builds a k-nearest-neighbor (k-NN) graph representing the local
        connectivity structure of the manifold.

        Each point `i` is connected to its `k` nearest neighbors `j`
        (found using a KD-tree search). The edge weight corresponds to
        the Euclidean distance between the two points.

        This graph serves as a discrete approximation of the manifold's
        intrinsic topology and is used for operations such as:
            - Geodesic distance computation (e.g., Dijkstra, Floyd–Warshall)
            - Diffusion and propagation along the surface
            - Graph-based Laplace–Beltrami discretization

        Notes:
            - Points with fewer than `__MIN_NEIGHBORHOOD` neighbors are skipped.
            - The graph is undirected and uses symmetric weights.
            - The resulting adjacency encodes local Euclidean geometry, not
              intrinsic geodesic distances (which must be computed separately).
        """
        for i, p in enumerate(self.points):
            
            distances, indices = self.tree.query(p, k=self.k + 1)
            neighborhood: np.array = self.points[indices]
            
            if len(neighborhood) < self.__MIN_NEIGHBORHOOD:
                continue
            
            for idx, j in enumerate(indices):
                if j != i:
                    self.graph.add_edge(i, j, weight=distances[idx])
        
        
    def __eigen(self, data: np.array, bias=False) -> tuple[np.array, np.array]:
        """
        Computes and sorts the eigenvalues and eigenvectors of the covariance matrix
        derived from the local neighborhood data.

        The smallest eigenvector corresponds to the surface normal; the remaining two
        form an orthonormal tangent basis.

        Args:
            data (np.array): Transposed neighborhood data (3 x N).
            bias (bool): If True, uses biased covariance estimation.

        Returns:
            tuple:
                - np.array: Sorted eigenvalues (descending).
                - np.array: Corresponding eigenvectors (columns represent directions).
        """
        covariance_matrix = np.cov(data, bias=bias)
        eigenvalues, eigenvectors = LA.eig(covariance_matrix)

        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        return eigenvalues, eigenvectors
    
    def compute_orthonormal_basis(self) -> np.array:
        """
        Computes an orthonormal local reference frame (basis) for each point in the point cloud
        using Principal Component Analysis (PCA) on its k-nearest neighbors.

        For each point, PCA is applied to its local neighborhood to obtain the three
        orthogonal eigenvectors of the covariance matrix:
            - The eigenvector associated with the smallest eigenvalue represents the
            estimated surface normal.
            - The other two eigenvectors span the local tangent plane.

        These three vectors form an orthonormal basis (3x3) that defines the local
        geometry around each point.

        Returns:
            np.array: 
                A tensor of shape (N, 3, 3), where N is the number of points in the
                point cloud. For each point `i`, `basis[i]` is a 3×3 matrix whose columns
                correspond to the eigenvectors `[v₁, v₂, v₃]`, forming an orthonormal basis:
                
                - `v₁`: Principal direction (largest variance)
                - `v₂`: Secondary tangent direction
                - `v₃`: Normal direction (smallest variance)

        Notes:
            - Points with fewer than `__MIN_NEIGHBORHOOD` neighbors are skipped (basis remains zero).
            - Neighborhoods are determined using a KD-tree (`self.tree.query`).
            - The basis orientation is not guaranteed to be globally consistent
            (normals may point in opposite directions across the surface).
        """
        basis: np.array = np.zeros((self.points.shape[0], 3, 3))
        
        for node in self.graph.nodes:
            
            indices = list(self.graph.neighbors(node))
            neighborhood: np.array = self.points[indices]

            if len(neighborhood) < self.__MIN_NEIGHBORHOOD:
                continue
            
            _, eigenvectors = self.__eigen(neighborhood.T)
            basis[node] = eigenvectors
        
        return basis
    
    def compute_tangent_space_basis(self, orthonormal_basis: np.array) -> np.array:
        """
        Computes a smooth tangent basis for each point by averaging the tangent
        directions (e1, e2) of neighboring points to capture local curvature.

        Args:
            orthonormal_basis (np.array): (N, 3, 3) array where columns are [v1, v2, v3].

        Returns:
            np.array: (N, 2, 3) array, where each entry contains two tangent vectors
                    forming the local tangent space basis.
        """
        basis = np.zeros((self.points.shape[0], 2, 3))

        for node in self.graph.nodes:
            
            indices = list(self.graph.neighbors(node))
            neighborhood: np.array = self.points[indices]

            if len(neighborhood) < self.__MIN_NEIGHBORHOOD:
                continue
            
            e1_mean = np.mean(orthonormal_basis[indices, :, 0], axis=0)
            e2_mean = np.mean(orthonormal_basis[indices, :, 1], axis=0)

            e1 = e1_mean / np.linalg.norm(e1_mean)
            e2 = e2_mean / np.linalg.norm(e2_mean)

            basis[node, 0] = e1
            basis[node, 1] = e2

        return basis
    
    def compute_metric_tensor(self, tangent_space_basis: np.array) -> tuple[np.array, np.array]:
        """
        Computes the local metric tensor and its inverse for each point in the point cloud
        based on the (non-orthonormal) tangent space basis vectors.

        For each point `i`, the local tangent space is defined by two averaged tangent
        vectors `[e₁, e₂]` obtained from neighboring PCA bases (see `compute_tangent_space_basis`).
        These vectors are not necessarily orthogonal, allowing the metric tensor to
        capture local curvature and anisotropy of the surface.

        The metric tensor encodes the inner products between the tangent basis vectors:
            g_ij = ⟨e_i, e_j⟩ = e_i · e_j

        Thus, for each point `i`, the metric tensor is:
            g = [[⟨e₁, e₁⟩, ⟨e₁, e₂⟩],
                [⟨e₂, e₁⟩, ⟨e₂, e₂⟩]]

        Its inverse `g⁻¹` is also computed, as it is often required for differential
        geometric operations (e.g., gradient, Laplace–Beltrami, curvature).

        Args:
            tangent_space_basis (np.array):
                Array of shape (N, 2, 3), where each entry `tangent_space_basis[i]`
                contains the two tangent vectors `[e₁, e₂]` spanning the local tangent
                plane at point `i`.

        Returns:
            tuple[np.array, np.array]:
                - `metric_tensor`: Array of shape (N, 2, 2) containing the metric tensor
                for each point.
                - `inv_metric_tensor`: Array of the same shape containing the inverse
                of the metric tensor for each point.

        Notes:
            - If the tangent vectors are orthonormal (e.g., from raw PCA), then
            `metric_tensor[i] ≈ I₂`.
            - When the tangent vectors are averaged across neighborhoods, the resulting
            metric deviates from the identity, encoding local curvature information.
            - Singular metric tensors (e.g., due to degenerate neighborhoods) will
            raise a `LinAlgError` during inversion.
        """
        metric_tensor: np.array = np.zeros((self.points.shape[0], 2, 2))
        inv_metric_tensor: np.array = np.zeros(metric_tensor.shape)
        
        for i in range(self.points.shape[0]):
            
            e1, e2 = tangent_space_basis[i]
            
            metric_tensor[i] = np.array([
                [np.dot(e1, e1), np.dot(e1, e2)],
                [np.dot(e2, e1), np.dot(e2, e2)]
            ])
        
            inv_metric_tensor[i] = np.linalg.inv(metric_tensor[i])
            
        return metric_tensor, inv_metric_tensor
    
    def compute_metric_tensor_derivatives(self) -> None:
        metric_tensor_derivatives: np.array = np.zeros((self.points.shape[0], 2, 2, 2))
        return metric_tensor_derivatives
            
    
    def compute_christoffel_symbols(self, ) -> np.array:
        christoffel_symbols: np.array = np.zeros((self.points.shape[0], 2, 2, 2))
        return christoffel_symbols
    
    
if __name__ == "__main__":
    
    points: np.array = np.loadtxt('bunny.txt')
    manifold = Manifold(points)
    
    orthonormal_basis = manifold.compute_orthonormal_basis()
    normals = orthonormal_basis[:, :, 2]
    
    tangent_space_basis = manifold.compute_tangent_space_basis(orthonormal_basis)
    
    metric_tensor, inv_metric_tensor = manifold.compute_metric_tensor(tangent_space_basis)
    print(metric_tensor)