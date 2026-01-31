import numpy as np
import networkx as nx
from scipy.spatial import KDTree

from .fitting.transform import *
from .fitting.fitting import *
from .fitting.plotting import *

class Manifold:
    """
    Represents a 2D differentiable manifold embedded in 3D space, discretized 
    by a set of sample points (typically originating from a mesh).
    
    Estimates local differential geometry (tangent and normal spaces), builds
    a connectivity graph between neighboring points, and enables geodesic
    path computation along the manifold.
    """
    __MIN_NEIGHBORHOOD: int = 3
    
    def __init__(self, points: np.ndarray, k: int=10) -> None:
        self.points = points
        self.k = k
        
        self.tree = KDTree(points)
        self.graph: nx.Graph = nx.Graph()
        
        self._init_tensors()

    def _find_surface(self, points_local: np.ndarray) -> np.ndarray:
        """
        Fits a plane to a set of local points and transforms them to a local coordinate system.

        This function first fits a plane to the input points to define a local coordinate system. 
        It then transforms the points into this local system and fits a paraboloid to the transformed points.

        Args:
            points_local (np.ndarray): Array of shape (n_points, n_dimensions) representing the points 
                                    in the local coordinate frame.

        Returns:
            tuple:
                - X_local (np.ndarray): The transformed points in the local coordinate system.
                - params (np.ndarray): Parameters of the fitted paraboloid in the local coordinate system.
        """
        normal, p0 = fit_plane(points_local)
        X_local, R, (u, v, w) = transform_to_local_coordinates(points_local, p0, normal)

        params = fit_paraboloid(X_local)

        return X_local, params, R
    
    def _compute_surface_derivatives(self, params: np.ndarray, u: float, v: float) -> tuple[float, float]:
        """
        Computes the partial derivatives of the paraboloid f_u and f_v at a point p = (u, v).
        
        Args:
            params (np.ndarray): Coefficients of the quadratic surface in the order [a, b, c, d, e, f].
            u (float): x-coordinate of the point where tangent vectors are computed.
            v (float): y-coordinate of the point where tangent vectors are computed.
            
        Returns:
            tuple[float, float]: partial derivative with respect u, partial derivative with respect v.
        """
        a, b, c, d, e, f = params
        
        f_u = 2*a*u + c*v + d
        f_v = 2*b*v + c*u + e
        
        return f_u, f_v
    
    def _compute_surface_second_derivatives(self, params: np.ndarray, u: float, v: float) -> tuple[float, float, float]:
        """
        Computes the partial second derivatives of the paraboloid f_uu, f_uv and f_vv at a point p = (u, v).
        
        Args:
            params (np.ndarray): Coefficients of the quadratic surface in the order [a, b, c, d, e, f].
            u (float): x-coordinate of the point where tangent vectors are computed.
            v (float): y-coordinate of the point where tangent vectors are computed.
            
        Returns:
            tuple[float, float, float]: f_uu, f_uv, f_vv.
        """
        a, b, c, d, e, f = params
        
        f_uu = 2*a
        f_uv = c
        f_vv = 2*b
        
        return f_uu, f_uv, f_vv
        
    def _compute_tangent_vectors(self, params: np.ndarray, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the tangent vectors of a quadratic surface at a given
        p = (u, v) in local coordiantes.

        The surface is defined by the function:
            f(x, y) = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f
        where `params = [a, b, c, d, e, f]`.  
        The tangent vectors are calculated at the point (u, v, f(u, v)).

        Args:
            params (np.ndarray): Coefficients of the quadratic surface in the order [a, b, c, d, e, f].
            u (float): x-coordinate of the point where tangent vectors are computed.
            v (float): y-coordinate of the point where tangent vectors are computed.

        Returns:
            tuple[np.ndarray, np.ndarray]: Two tangent vectors at the point (u, v, f(u, v)):
                - r_u: Tangent vector in the direction of increasing x.
                - r_v: Tangent vector in the direction of increasing y.
        """
        f_u, f_v = self._compute_surface_derivatives(params, u, v)

        r_u = np.array([1, 0, f_u])
        r_v = np.array([0, 1, f_v])

        return r_u, r_v
    
    def _compute_first_fundamental_form(self, params: np.ndarray, u: float, v: float) -> tuple[float, float, float]:
        """
        Computes the first fundamental form at a given point p = (u, v) in local coordiantes.

        Args:
            params (np.ndarray): Coefficients of the quadratic surface in the order [a, b, c, d, e, f].
            u (float): x-coordinate of the point in local coordinates.
            v (float): y-coordinate of the point in local coordinates.

        Returns:
            tuple[float, float, float]: E, F, G terms of the first fundamental form.
        """
        f_u, f_v = self._compute_surface_derivatives(params, u, v)
        
        E = 1 + f_u**2
        F = f_u * f_v
        G = 1 + f_v**2
        
        return E, F, G
    
    def _compute_second_fundamental_form(self, params: np.ndarray, normal: np.ndarray, u: float, v: float) -> tuple[float, float, float]:
        """
        Computes the second fundamental form at a given point p = (u, v) in local coordiantes.

        Args:
            params (np.ndarray): Coefficients of the quadratic surface in the order [a, b, c, d, e, f].
            normal (np.ndarray): Normalised normal vector at (u, v).
            u (float): x-coordinate of the point in local coordinates.
            v (float): y-coordinate of the point in local coordinates.

        Returns:
            tuple[float, float, float]: L, M, N terms of the second fundamental form.
        """
        f_uu, f_uv, f_vv = self._compute_surface_second_derivatives(params, u, v)
        
        r_uu = np.array([0, 0, f_uu])
        r_uv = np.array([0, 0, f_uv])
        r_vv = np.array([0, 0, f_vv])
        
        L = np.dot(r_uu, normal)
        M = np.dot(r_uv, normal)
        N = np.dot(r_vv, normal)
        
        return L, M, N
        
    def _compute_metric_tensor(self, params: np.ndarray, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the metric tensor g_{μν} and its inverse g^{μν} at a given 
        point p = (u, v) in local coordiantes.

        Args:
            params (np.ndarray): Coefficients of the quadratic surface in the order [a, b, c, d, e, f].
            u (float): x-coordinate of the point in local coordinates.
            v (float): y-coordinate of the point in local coordinates.

        Returns:
            tuple[np.ndarray, np.ndarray]: The metric tensor and its inverse.
                - g: the metric tensor at a given point p = (u, v)
                - g_inv: the inverse metric tensor at a given point p = (u, v)
        """
        E, F, G = self._compute_first_fundamental_form(params, u, v)
        
        g = np.array([
            [E, F],
            [F, G]
        ])
        
        g_inv = np.array([
            [G, -F],
            [-F, E]
        ]) / (E * G - F**2)
        
        return g, g_inv
    
    def _compute_metric_tensor_derivatives(self, params: np.ndarray, u: float, v: float) -> np.ndarray:
        """
        Computes the metric tensor derivatives ∂_α g_{μν} at a given 
        point p = (u, v) in local coordinates.

        Args:
            params (np.ndarray): Coefficients of the quadratic surface in the order [a, b, c, d, e, f].
            u (float): x-coordinate of the point in local coordinates.
            v (float): y-coordinate of the point in local coordinates.

        Returns:
            np.ndarray: The partial derivatives ∂_α g_{μν}:
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
    
    def _compute_christoffel_symbols(self, metric_tensor_inv: np.ndarray, metric_tensor_derivatives: np.ndarray) -> np.ndarray:
        """
        Computes the Christoffel symbols of the second kind at each sample point
        using the metric tensor and its partial derivatives.

        The Christoffel symbols Γ^σ_{μν} define the Levi-Civita connection and describe
        how tangent vectors are differentiated along coordinate directions on a manifold.

        The computation uses the standard coordinate formula:
            Γ^σ_{μν} = (1/2) * g^σλ ( ∂_μ g_νλ + ∂_ν g_μλ − ∂_λ g_μν )

        Index meanings:
            - μ, ν: Indices of the coordinate directions along which the derivative acts.
            - λ: Dummy index used for summation (Einstein convention).
            - σ: Index of the output coordinate direction of the resulting connection.

        Data structure:
            - self.metric_tensor[i] is the 2×2 metric tensor g_μν at sample point i.
            - self.metric_tensor_inv[i] is the inverse metric tensor g^μν at point i.
            - self.metric_tensor_derivatives[i, α] is ∂_α g_μν at point i,
            for α ∈ {0, 1} representing the coordinate direction.

        Storage format:
            The computed Christoffel symbols are stored in `self.christoffel_symbols`,
            an array of shape (N, 2, 2, 2), where:
                - N is the number of sample points.
                - The last three indices correspond to Γ^σ_{μν}, ordered as:
                    [i, μ, ν, σ] → value of Γ^σ_{μν} at point i.

        Notes:
            - The Christoffel symbols are symmetric in the lower indices: Γ^σ_{μν} = Γ^σ_{νμ}.
            - Assumes a 2-dimensional Riemannian manifold (μ, ν, σ ∈ {0, 1}).
            - Derivatives of the metric tensor are assumed to be precomputed and provided.
            
        Args:
            metric_tensor_inv (np.ndarray): the inverse metric tensor at a given point p = (u, v).
            metric_tensor_derivatives (np.ndarray): 
        """
        christoffel_symbols = np.zeros((2, 2, 2))
        
        for mu in range(0, 2):
            for nu in range(0, 2):
                for sigma in range(0, 2):
                    
                    partial_mu = metric_tensor_derivatives[mu]
                    partial_nu = metric_tensor_derivatives[nu]

                    sum: float = 0
                    for l in range(0, 2):
                        partial_lambda: np.ndarray = metric_tensor_derivatives[l]
                        sum += metric_tensor_inv[sigma, l] * (partial_mu[nu, l] + partial_nu[mu, l] - partial_lambda[mu, nu])  
                    
                    christoffel_symbols[mu, nu, sigma] = 0.5 * sum
        
        return christoffel_symbols
    
    def _compute_gaussian_curvature(self, params: np.ndarray, normal: np.ndarray, u: float, v: float) -> float:
        """
        Computes the gaussian curvature K at a given point p = (u, v).

        Args:
            params (np.ndarray): Coefficients of the quadratic surface in the order [a, b, c, d, e, f].
            normal (np.ndarray): Normalised normal vector at (u, v).
            u (float): x-coordinate of the point in local coordinates.
            v (float): y-coordinate of the point in local coordinates.

        Returns:
            float: the gaussian curvature K.
        """
        E, F, G = self._compute_first_fundamental_form(params, u, v)
        L, M, N = self._compute_second_fundamental_form(params, normal, u, v)
        
        return (L*N - M**2) / (E*G - F**2)

    def _init_tensors(self) -> None:
        """
        It computes the following tensors:
    
            - points: the points in ambient space coordinates (x,y,z).
            - local_coordinats: the points in local coordinates (u,v) that would be given by a chart (rotated points projected into XY plane).
            - normal_vectors: the normal vector (normalised) at each point.
            - tangent_vectors: the tangent vectors (tangent space) at each point.
            - metric_tensor: the metric tensor at each point.
            - metric_tensor_inv: the inverse of the metric tensor at each point.
            - metric_tensor_derivatives: the derivatives of the metric tensor at each point.
            - christoffel_symbols: the christoffel symbols at each point.
            - gaussian_curvature: the gaussian curvature at each point.
            - scalar_curvature: the scalar curvature at each point.
            - ricci_curvature_tensor: the ricci curvature tensor at each point.
            - riemann_curvature_tensor: the riemann curvature tensor at each point.
        """
        self.local_coordinates = np.zeros((self.points.shape[0], 2)) # Given by a chart φ
        
        self.normal_vectors = np.zeros((self.points.shape[0], 3))
        self.tangent_vectors = np.zeros((self.points.shape[0], 2, 3))
        
        self.metric_tensor = np.zeros((self.points.shape[0], 2, 2))
        self.metric_tensor_inv = np.zeros((self.points.shape[0], 2, 2))
        self.metric_tensor_derivatives = np.zeros((self.points.shape[0], 2, 2, 2))
        
        self.christoffel_symbols = np.zeros((self.points.shape[0], 2, 2, 2))
        
        self.gaussian_curvature = np.zeros((self.points.shape[0]),)
        self.scalar_curvature = np.zeros((self.points.shape[0]),)
        self.ricci_curvature_tensor = np.zeros((self.points.shape[0], 2, 2))
        self.riemann_curvature_tensor = np.zeros((self.points.shape[0], 2, 2, 2, 2))
        
        for i, p in enumerate(self.points):

            # Find the surface
            distances, indices = self.tree.query(p, k=self.k + 1)
            
            points_local = self.points[indices]
            points_local_transformed, params, R = self._find_surface(points_local)
            
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
            r_u_transformed, r_v_transformed = self._compute_tangent_vectors(params, u, v)

            # Tangent vectors at p. This is in ambient space coordinates, not local coordinates.
            r_u = R.T @ r_u_transformed
            r_v = R.T @ r_v_transformed
            self.tangent_vectors[i] = np.array([r_u, r_v])
            
            # Normal vector at p
            normal = np.cross(r_u, r_v)
            normal /= np.linalg.norm(normal)
            self.normal_vectors[i] = normal

            # Metric tensor in p
            metric_tensor, metric_tensor_inv = self._compute_metric_tensor(params, u, v)
            self.metric_tensor[i] = metric_tensor
            self.metric_tensor_inv[i] = metric_tensor_inv
            
            # Metric tensor derivatives
            metric_tensor_derivatives = self._compute_metric_tensor_derivatives(params, u, v)
            self.metric_tensor_derivatives[i] = metric_tensor_derivatives
            
            # Christoffel symbols
            christoffel_symbols = self._compute_christoffel_symbols(metric_tensor_inv, metric_tensor_derivatives)
            self.christoffel_symbols[i] = christoffel_symbols
            
            # Gaussian curvature
            K = self._compute_gaussian_curvature(params, normal, u, v)
            self.gaussian_curvature[i] = K
            
            # Scalar curvature
            self.scalar_curvature[i] = 2 * K
            
            # Ricci curvature tensor
            self.ricci_curvature_tensor[i] = metric_tensor * K
            
            # Riemann curvature tensor
            det_g = metric_tensor[0, 0] * metric_tensor[1, 1] - metric_tensor[0, 1] * metric_tensor[1, 0]
            
            self.riemann_curvature_tensor[i][0, 1, 0, 1] =  K * det_g
            self.riemann_curvature_tensor[i][0, 1, 1, 0] = -K * det_g
            self.riemann_curvature_tensor[i][1, 0, 0, 1] = -K * det_g
            self.riemann_curvature_tensor[i][1, 0, 1, 0] =  K * det_g
            
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