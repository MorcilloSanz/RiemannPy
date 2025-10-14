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
        self.graph = nx.Graph()
        
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
            neighborhood = self.points[indices]
            
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
        basis = np.zeros((self.points.shape[0], 3, 3))
        
        for node in self.graph.nodes:
            
            indices = list(self.graph.neighbors(node))
            neighborhood = self.points[indices]

            if len(neighborhood) < self.__MIN_NEIGHBORHOOD:
                continue
            
            _, eigenvectors = self.__eigen(neighborhood.T)
            basis[node] = eigenvectors
        
        return basis
    
    def compute_metric_tensor(self) -> list[np.array]:
        """
        Compute the local metric tensor for a graph embedded in Euclidean space.

        Each node in the graph is associated with a set of edges connecting it to neighboring nodes.
        For each node, this function computes the *local metric tensor* `g_{μν}` defined as:

            g_{μν} = <e_μ, e_ν> = e_μ · e_ν

        where each vector `e_μ` represents the tangent direction of an edge emanating from the node,
        computed as the difference between the coordinates of the neighboring point and the current node.

        In discrete geometry, this metric tensor provides a measure of the local geometric structure
        (edge lengths and angles) around each node, analogous to the continuous metric tensor on a manifold.

        ---
        **Mathematical definition**

        For each node `p_i`, with neighboring nodes `{p_j}`, let:

            e_μ = p_j - p_i

        Then the local metric tensor is given by:

            G_i = [ [ <e_1, e_1>, <e_1, e_2>, ..., <e_1, e_n> ],
                    [ <e_2, e_1>, <e_2, e_2>, ..., <e_2, e_n> ],
                    ...
                    [ <e_n, e_1>, <e_n, e_2>, ..., <e_n, e_n> ] ]

        where `<·,·>` denotes the standard Euclidean inner product.

        ---
        **Returns**
        -------
        list[np.ndarray]
            A list where each entry corresponds to the metric tensor of a node in the graph.
            Each tensor is a square matrix of shape `(k_i, k_i)`, where `k_i` is the number of edges
            connected to the i-th node (i.e., its degree).

        ---
        **Notes**
        -----
        - The resulting list may contain matrices of varying sizes, since each node can have a different degree.
        - If the graph is undirected, the orientation of the edge vectors is chosen such that all vectors
        point outward from the current node.
        - The metric tensor can be used to compute local geometric quantities such as edge lengths,
        angles, or local curvature approximations.

        **Example**
        -------
        >>> field = mesh.compute_metric_tensor_field()
        >>> field[0].shape
        (3, 3)
        >>> field[0]
        array([[1.00, 0.50, 0.20],
            [0.50, 1.00, 0.45],
            [0.20, 0.45, 1.00]])
        """
        tensor_field: list[np.array] = []
        
        for node in self.graph.nodes:
            
            edges = list(self.graph.edges(node))
            metric_tensor = np.zeros((len(edges), len(edges)))
            
            e = []
            for edge in edges:
                e_i = self.points[edge[1]] - self.points[edge[0]]
                e.append(e_i)
            
            for i in range(len(e)):
                for j in range(len(e)):
                    metric_tensor[i,j] = np.dot(e[i], e[j])
                    
            tensor_field.append(metric_tensor)
            
        return tensor_field
                
    def compute_metric_tensor_derivatives(self, metric_tensor: np.array) -> None:
        """
        Defined in each edge (i,j) of the graph
        
        $\partial_{\alpha} g_{\mu \nu}(p_i)  = \sqrt{w_{ij}}(g_{\mu \nu}(p_j) - g_{\mu \nu}(p_i))$

        Args:
            metric_tensor (np.array): _description_

        Returns:
            _type_: _description_
        """
        tensor_field: list[np.array] = []
        
        # No se pueden restar los tensores metricos de distintos nodos
        # puesto que cada uno puede tener un numero distinto de aristas
        # y por tanto las dimensiones de las matrices serán diferentes.
        # Hay que considerar los tensores métricos de las mismas dimensiones
        # 2x2 o 3x3 por ejemplo.
        
        edges = list(self.graph.edges(data="weight")) #(i, j, w_ij)
        
        for ij, edge in enumerate(edges):
            pass
        
        return tensor_field

    def compute_christoffel_symbols(self, ) -> np.array:
        christoffel_symbols = np.zeros((self.points.shape[0], 2, 2, 2))
        return christoffel_symbols
    
    
if __name__ == "__main__":
    
    # Create manifold
    points = np.loadtxt('bunny.txt')
    manifold = Manifold(points)
    
    # Tangent space
    orthonormal_basis = manifold.compute_orthonormal_basis()

    tangent_u = orthonormal_basis[:, :, 0]
    tangent_v = orthonormal_basis[:, :, 1]
    normals = orthonormal_basis[:, :, 2]
    
    # Metric tensor
    metric_tensor_field = manifold.compute_metric_tensor()
    
    #metric_tensor_derivatives = manifold.compute_metric_tensor_derivatives(metric_tensor)