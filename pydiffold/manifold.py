import numpy as np
import networkx as nx
from scipy.spatial import KDTree

from .transform import *
from .fitting import *
from .plotting import *

class Manifold:
    """
    Represents a 2D differentiable manifold embedded in 3D space, discretized 
    by a set of sample points (typically originating from a mesh).
    
    Estimates local differential geometry (tangent and normal spaces), builds
    a connectivity graph between neighboring points, and enables geodesic
    path computation along the manifold.
    """
    __MIN_NEIGHBORHOOD: int = 3
    
    def __init__(self, points: np.array, k: int=10) -> None:
        self.points = points
        self.k = k
        
        self.tree = KDTree(points)
        self.graph: nx.Graph = nx.Graph()
        
        self.__init_tensors()

    def __find_surface(self, points_local: np.array) -> np.array:
        """
        Fits a plane to a set of local points and transforms them to a local coordinate system.

        This function first fits a plane to the input points to define a local coordinate system. 
        It then transforms the points into this local system and fits a paraboloid to the transformed points.

        Args:
            points_local (np.array): Array of shape (n_points, n_dimensions) representing the points 
                                    in the local coordinate frame.

        Returns:
            tuple:
                - X_local (np.array): The transformed points in the local coordinate system.
                - params (np.array): Parameters of the fitted paraboloid in the local coordinate system.
        """
        normal, p0 = fit_plane(points_local)
        X_local, R, (u, v, w) = transform_to_local_coordinates(points_local, p0, normal)

        params = fit_paraboloid(X_local)

        return X_local, params, R

    def __compute_tangent_vectors(self, params: np.array, u: float, v: float) -> tuple[np.array, np.array]:
        """
        Computes the tangent vectors of a quadratic surface at a given point.

        The surface is defined by the function:
            f(x, y) = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f
        where `params = [a, b, c, d, e, f]`.  
        The tangent vectors are calculated at the point (u, v, f(u, v)).

        Args:
            params (np.array): Coefficients of the quadratic surface in the order [a, b, c, d, e, f].
            u (float): x-coordinate of the point where tangent vectors are computed.
            v (float): y-coordinate of the point where tangent vectors are computed.

        Returns:
            tuple[np.array, np.array]: Two tangent vectors at the point (u, v, f(u, v)):
                - r_u: Tangent vector in the direction of increasing x.
                - r_v: Tangent vector in the direction of increasing y.
        """
        a, b, c, d, e, f = params

        r_u = np.array([1, 0, 2*a*u + c*v + d])
        r_v = np.array([0, 1, 2*b*v + c*u + e])

        return r_u, r_v

    def __compute_metric_tensor(self, r_u: np.array, r_v: np.array) -> tuple[np.array, np.array]:
        """
        Computes the metric tensor g_{μν} and its inverse g^{μν} at a given point p = (u, v)

        Args:
            r_u (np.array): _description_
            r_v (np.array): _description_

        Returns:
            tuple[np.array, np.array]: The metric tensor and its inverse.
                - g: the metric tensor at a given point p = (u, v)
                - g_inv: the inverse metric tensor at a given point p = (u, v)
        """
        g = np.array([
            [np.dot(r_u, r_u), np.dot(r_u, r_v)],
            [np.dot(r_v, r_u), np.dot(r_v, r_v)]
        ])
        
        g_inv: np.array = np.linalg.inv(g)
        
        return g, g_inv

    def __init_tensors(self) -> None:
        
        self.metric_tensor = np.zeros((self.points.shape[0], 2, 2))
        self.metric_tensor_inv = np.zeros((self.points.shape[0], 2, 2))
        
        for i, p in enumerate(self.points):

            # Find surface
            distances, indices = self.tree.query(p, k=self.k + 1)
            
            points_local: np.array = self.points[indices]
            points_local_transformed, params, R = self.__find_surface(points_local)
            
            a, b, c, d, e, f = params

            # plot_paraboloid(points_local_transformed, params)
            
            # Fill graph
            if len(points_local) < self.__MIN_NEIGHBORHOOD:
                continue
            
            # Compute graph
            for idx, j in enumerate(indices):
                if j != i:
                    self.graph.add_edge(i, j, weight=distances[idx])

            # Find the transformed p in the transformed surface
            idx_local = np.where(np.all(np.isclose(points_local, p), axis=1))[0][0]
            p_transformed = points_local_transformed[idx_local]

            # p_transformed is p rotated so we can evaulate it in the surface (wich is also rotated)
            # p_transformed = (x_0, y_0, z_0) where z_0 = f(x_0, y_0) and f is the paraboloid
            u_transformed = p_transformed[0]  # As p_transformed = (x_0, y_0, z_0) where z_0 = f(x_0, y_0), u = x_0
            v_transformed = p_transformed[1]  # As p_transformed = (x_0, y_0, z_0) where z_0 = f(x_0, y_0), v = y_0

            # Transformed tangent vectors at p
            r_u_transformed, r_v_transformed = self.__compute_tangent_vectors(params, u_transformed, v_transformed)

            # Tangent vectors at p
            r_u = R.T @ r_u_transformed
            r_v = R.T @ r_v_transformed

            # metric tensor in p
            g, g_inv = self.__compute_metric_tensor(r_u, r_v)
            
            self.metric_tensor[i] = g
            self.metric_tensor_inv[i] = g_inv

    def geodesic(self, start_index: int, end_index: int) -> tuple[np.array, float]:
        """
        Computes a discrete geodesic path between two sample points on the manifold,
        using Dijkstra's algorithm over the connectivity graph.

        Args:
            start_index (int): Index of the source point.
            end_index (int): Index of the target point.

        Returns:
            tuple:
                - np.array: Sequence of vertex indices along the geodesic path.
                - float: Total length of the path.
        """
        shortest_path = nx.shortest_path(self.graph, start_index, end_index, weight="weight")
        total_cost: float = nx.path_weight(self.graph, shortest_path, weight="weight")
        return np.array(shortest_path), total_cost