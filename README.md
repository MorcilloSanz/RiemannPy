<p align="center">
    <img src="img/logo.png" width="400"/>
</p>

`RiemannPy` is a Python library for **Differential Geometry**, with a particular focus on **Riemannian Geometry on discrete data**. It is designed to bridge the gap between smooth geometric theory and practical numerical computation by providing tools to approximate **local differential structure** and **intrinsic differential operators** directly from point-sampled surfaces.

Unlike traditional geometry processing frameworks that rely on explicit mesh connectivity, `RiemannPy` operates purely on **point clouds**, making it especially useful in scenarios where connectivity is unknown, unreliable, or expensive to compute. By leveraging local neighborhoods and geometric fitting techniques, the library reconstructs key geometric quantities such as tangent spaces, metric tensors, and curvature in a fully discrete setting.

The main goal of `RiemannPy` is to enable **numerical experimentation in geometric analysis and partial differential equations (PDEs)** on manifolds. It provides a consistent framework to study intrinsic properties of surfaces and to simulate physical processes such as heat diffusion or wave propagation directly on sampled geometries.

The library is particularly well-suited for:
- Geometry processing and surface analysis.
- Discrete differential geometry research.
- Simulation of PDEs on manifolds.
- Prototyping algorithms that do not depend on mesh structures.

By abstracting complex geometric concepts into accessible data structures and operators, `RiemannPy` allows researchers and developers to focus on experimentation and algorithm design, rather than low-level geometric implementation details.

### RiemannPy for computing the curvature of a sampled surface
The `scalar curvature` of a sampled surface can be directly computed, providing a practical way to analyze the intrinsic geometry of complex and even irregular surfaces.
![](img/curvature.png)

### RiemannPy for Partial Differential Equations
The `Laplace–Beltrami operator` can be directly computed, enabling the efficient solution of Partial Differential Equations on manifolds defined by sampled surfaces.

![](img/heat_equation.png)

![](img/wave_equation.png)

```python
# Load manifold
points = np.array([...])
manifold = Manifold(points)
    
# Temperature scalar field at t=0
init_values = np.sin(manifold.points[:, 0] * 1.5) + np.cos(manifold.points[:, 1] * 1.5)
phi = ScalarField(manifold, init_values)

# Solve heat equation
alpha = 0.25
delta_t = 1.0

phi.compute_differential_operators()
phi.values = phi.values + delta_t * alpha * phi.laplace_beltrami
```

## Local differential structure

The `Manifold` class computes and stores the local differential geometry at each sample point, providing direct access to the intrinsic structure of the surface.

### Geometry & Local Coordinates
- **`points`**: Coordinates in ambient 3D space $(x, y, z)$.
- **`local_coordinates`**: 2D coordinates $(u, v)$ (in the tangent plane, given by a chart $ \varphi : U \subset \mathcal{M} \rightarrow \mathbb{R}^2$).
- **`normal_vectors`**: Unit normal vector $\vec{n}$.
- **`tangent_vectors`**: Basis $\{r_u, r_v\}$ spanning the tangent space $T_p\mathcal{M}$.

### Metric Structure
- **`metric_tensor`**: Metric tensor $g_{\mu\nu}$ (First Fundamental Form).
- **`metric_tensor_inv`**: Inverse metric $g^{\mu\nu}$.
- **`metric_tensor_derivatives`**: Partial derivatives $\partial_\alpha g_{\mu\nu}$.

### Connection & Curvature
- **`christoffel_symbols`**: Christoffel symbols $\Gamma^\sigma_{\mu\nu}$.
- **`gaussian_curvature`**: Gaussian curvature $K$.
- **`scalar_curvature`**: Scalar curvature $R = 2K$.
- **`ricci_curvature_tensor`**: Ricci tensor $R_{\mu\nu} = K g_{\mu\nu}$.
- **`riemann_curvature_tensor`**: Riemann tensor $R_{\mu\nu\sigma\rho}$.

### Geodesics
- **`geodesic`**: Computes the geodesic and its arc length between two points.

### Usage Example: Accessing Manifold Properties

Once the `Manifold` is initialized, all geometric tensors are precomputed and stored as NumPy arrays. Here is how to access them:

```python
import numpy as np
from riemannpy.manifold import Manifold

# 1. Initialize the manifold with a point cloud (N, 3)
points    = np.random.rand(100, 3)
manifold  = Manifold(points, k=12)

# 2. Access Geometric Basis & Coordinates
p_ambient = manifold.points[i]
p_local   = manifold.local_coordinates[i]
normal    = manifold.normal_vectors[i]
tangents  = manifold.tangent_vectors[i]

# 3. Access Metric Tensors
g         = manifold.metric_tensor[i]
g_inv     = manifold.metric_tensor_inv[i]
dg        = manifold.metric_tensor_derivatives[i]

# 4. Access Connection & Curvature
gamma     = manifold.christoffel_symbols[i]
K         = manifold.gaussian_curvature[i]
R_scalar  = manifold.scalar_curvature[i]
ricci     = manifold.ricci_curvature_tensor[i]
riemann   = manifold.riemann_curvature_tensor[i]

# 5. Geodesic Path Computation
geodesic, arc_length = manifold.geodesic(0, 2000)
geodesic_coords: np.array = manifold.points[geodesic]
```

## Differential operators

The `ScalarField` class represents a scalar function $f: M \to \mathbb{R}$ defined over the manifold. It provides discrete approximations of differential operators based on the local neighborhood of each point.

### Gradient
- **`gradient`**: Discrete edge-based representation of the surface gradient $\nabla_{\mathcal{M}} f$. Instead of a single vector in $\mathbb{R}^n$, it is represented as a list of arrays where each entry $i$ contains the weighted differences $\sqrt{w_{ij}}(f_j - f_i)$ along edges connected to point $i$.
- **`gradient_norm`**: Approximation of the gradient magnitude $\|\nabla_{\mathcal{M}} f\|$ at each point, computed from local variation and measuring the steepness of the field.

### Energy
- **`dirichlet_energy`**: Local Dirichlet energy $E_D(f)_i = \|\nabla_{\mathcal{M}} f\|_i^2$, measuring local roughness of the scalar field. High values indicate sharp transitions, while low values indicate smooth behavior.  

### Laplacian Operators
- **`laplacian`**: Approximation of the Random Walk Graph Laplacian $L_{rw} = I - D^{-1}W$, capturing the difference between a value and the weighted average of its neighbors, suitable for diffusion and smoothing.
- **`laplace_beltrami`**: Approximation of the Symmetric Normalized Laplacian $L_{sym} = I - D^{-1/2} W D^{-1/2}$, a numerically stable formulation widely used in spectral manifold analysis.

### Usage Example: Scalar Field Operations

The `ScalarField` class allows you to define a field (like temperature or pressure) over the manifold and compute its differential operators:

```python
import numpy as np

from riemannpy.manifold import Manifold
from riemannpy.field import ScalarField

# 1. Define Manifold
points             = np.random.rand(100, 3)
manifold           = Manifold(points, k=12)

# 2. Define scalar field
values             = np.sum(manifold.points, axis=1)
field              = ScalarField(manifold, values)

# 3. Differential operators
gradient           = field.gradient 
gradient_norm      = field.gradient_norm
dirichlet_energy   = field.dirichlet_energy
laplacian          = field.laplacian
laplace_beltrami   = filed.laplace_beltrami
```

## TODO
* Vector and Tensor fields.
* Covariant Derivative.
* Higher dimensions manifolds.

## Contributing
Please feel free to submit issues or pull requests.

## Dependencies
* [NumPy](https://github.com/numpy/numpy)
* [SciPy](https://github.com/scipy/scipy)
* [NetworkX](https://github.com/networkx/networkx)
* [Matplotlib](https://github.com/matplotlib/matplotlib)