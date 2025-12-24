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
        Computes the tangent vectors of a quadratic surface at a given
        p = (u, v) in local coordiantes.

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

    def __compute_metric_tensor(self, params: np.array, u: float, v: float) -> tuple[np.array, np.array]:
        """
        Computes the metric tensor g_{μν} and its inverse g^{μν} at a given 
        point p = (u, v) in local coordiantes.

        Args:
            params (np.array): Coefficients of the quadratic surface in the order [a, b, c, d, e, f].
            u (float): x-coordinate of the point in local coordinates.
            v (float): y-coordinate of the point in local coordinates.

        Returns:
            tuple[np.array, np.array]: The metric tensor and its inverse.
                - g: the metric tensor at a given point p = (u, v)
                - g_inv: the inverse metric tensor at a given point p = (u, v)
        """
        a, b, c, d, e, f = params
        
        f_u = 2*a*u + c*v + d
        f_v = 2*b*v + c*u + e
        
        E = 1 + f_u**2
        F = f_u * f_v
        G = 1 + f_v**2
        
        g = np.array([
            [E, F],
            [F, G]
        ])
        
        g_inv: np.array = np.linalg.inv(g)
        
        return g, g_inv
    
    def __compute_metric_tensor_derivatives(self, params: np.array, u: float, v: float) -> np.array:
        """
        Computes the metric tensor derivatives ∂_α g_{μν} at a given 
        point p = (u, v) in local coordinates.

        Args:
            params (np.array): Coefficients of the quadratic surface in the order [a, b, c, d, e, f].
            u (float): x-coordinate of the point in local coordinates.
            v (float): y-coordinate of the point in local coordinates.

        Returns:
            np.array: The partial derivatives ∂_α g_{μν}:
                ∂_1 g_{μν} = ∂_u g_{μν}
                ∂_2 g_{μν} = ∂_v g_{μν}
        """
        a, b, c, d, e, f = params
        metric_tensor_derivatives = np.zeros((2, 2, 2))
        
        f_u = 2*a*u + c*v + d
        f_uu = 2*a
        f_uv = c
        
        f_v = 2*b*v + c*u + e
        f_vv = 2*b
        f_vu = c
        
        dE_du = 2 * f_u * f_uu
        dF_du = f_uu * f_v + f_u * f_vu
        dG_du = 2 * f_v * f_vu
        
        metric_tensor_derivatives[0] = np.array([
            [dE_du, dF_du],
            [dF_du, dG_du]
        ])
        
        dE_dv = 2 * f_u * f_uv
        dF_dv = f_uv * f_v + f_u * f_vv
        dG_dv = 2 * f_v * f_vv
        
        metric_tensor_derivatives[1] = np.array([
            [dE_dv, dF_dv],
            [dF_dv, dG_dv]
        ])
        
        return metric_tensor_derivatives

    def __init_tensors(self) -> None:
        
        self.local_coordinates = np.zeros((self.points.shape[0], 2)) # Given by a chart φ
        
        self.normal_vectors = np.zeros((self.points.shape[0], 3))
        self.tangent_vectors = np.zeros((self.points.shape[0], 2, 3))
        
        self.metric_tensor = np.zeros((self.points.shape[0], 2, 2))
        self.metric_tensor_inv = np.zeros((self.points.shape[0], 2, 2))
        self.metric_tensor_derivatives = np.zeros((self.points.shape[0], 2, 2, 2))
        
        for i, p in enumerate(self.points):

            # Find the surface
            distances, indices = self.tree.query(p, k=self.k + 1)
            
            points_local: np.array = self.points[indices]
            points_local_transformed, params, R = self.__find_surface(points_local)
            
            a, b, c, d, e, f = params
            # plot_paraboloid(points_local_transformed, params)
            
            # Fill graph
            if len(points_local) < self.__MIN_NEIGHBORHOOD:
                continue

            for idx, j in enumerate(indices):
                if j != i:
                    self.graph.add_edge(i, j, weight=distances[idx])

            # Find the transformed p in the transformed surface
            idx_local = np.where(np.all(np.isclose(points_local, p), axis=1))[0][0]
            p_transformed = points_local_transformed[idx_local]

            # Since r(u, v) = <u, v, f(u, v)> --> r(p_transformed[0], p_transformed[1]) = <p_transformed[0], p_transformed[1], p_transformed[3]>
            # (u,v) are the local coordinates that would be given by a chart (rotated points projected into XY plane).
            u = p_transformed[0]
            v = p_transformed[1]
            
            self.local_coordinates[i] = [u, v]

            # Transformed tangent vectors at p
            r_u_transformed, r_v_transformed = self.__compute_tangent_vectors(params, u, v)

            # Tangent vectors at p. This is in ambient space coordinates, not local coordinates.
            r_u = R.T @ r_u_transformed
            r_v = R.T @ r_v_transformed
            self.tangent_vectors[i] = np.array([r_u, r_v])
            
            # Normal vector at p
            normal = np.cross(r_u, r_v)
            self.normal_vectors[i] = normal

            # Metric tensor in p
            g, g_inv = self.__compute_metric_tensor(params, u, v)
            self.metric_tensor[i] = g
            self.metric_tensor_inv[i] = g_inv
            
            # Metric tensor derivatives
            self.metric_tensor_derivatives[i] = self.__compute_metric_tensor_derivatives(params, u, v)

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