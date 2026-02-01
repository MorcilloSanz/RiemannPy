# PyDiffOld: Riemannian Geometry & Discrete Scalar Fields

**PyDiffOld** is a Python library built for performing intrinsic differential geometry analysis and solving Partial Differential Equations (PDEs) on discrete manifolds. 

By constructing local charts from point clouds, the library enables the computation of the full Riemannian metric apparatus and the application of high-order differential operators—such as the Laplace-Beltrami operator—on arbitrary surfaces.

---

## 🚀 Key Features

### 1. Riemannian Geometry Engine
The library provides an exhaustive suite of tools to analyze the intrinsic curvature and metric structure of a manifold $\mathcal{M}$:

* **Metric Properties**: Computation of the Metric Tensor $g_{ij}$, its inverse $g^{ij}$, and its partial derivatives $\partial_k g_{ij}$.
* **Coordinate Charts**: Transformation between ambient space coordinates $(x, y, z)$ and local tangent coordinates $(u, v)$.
* **Connection & Curvature**:
    * **Christoffel Symbols**: $\Gamma^k_{ij}$ derived from the metric tensor.
    * **Tensors**: Full Ricci and Riemann Curvature Tensors.
    * **Scalar Metrics**: Gaussian and Scalar curvature.
* **Vector Fields**: Normal and tangent vector estimation at every point.

### 2. Scalar Field Calculus
PyDiffOld allows you to define a scalar field $\phi$ over the manifold and compute operators that respect the underlying geometry:

* **Manifold Gradient**: $\nabla_{\mathcal{M}} \phi$ (The intrinsic surface gradient).
* **Gradient Norm**: Computation of the local Dirichlet energy.
* **Laplace-Beltrami Operator**: $\Delta_{\mathcal{M}} \phi = \frac{1}{\sqrt{|g|}} \partial_i \left( \sqrt{|g|} g^{ij} \partial_j \phi \right)$.



---

## 🛠 Applications: Solving PDEs on Manifolds

Because PyDiffOld provides a discrete approximation of the Laplace-Beltrami operator, it can be used to solve complex physical equations on non-Euclidean domains.

### Heat Equation
Simulate thermal diffusion over complex geometries:
$$\frac{\partial \phi}{\partial t} = \alpha \Delta_{\mathcal{M}} \phi$$

### Wave Equation
Model wave propagation and oscillations on surfaces:
$$\frac{\partial^2 \phi}{\partial t^2} = c^2 \Delta_{\mathcal{M}} \phi$$



---

## 💻 Quick Start

```python
import numpy as np
from pydiffold.manifold import Manifold
from pydiffold.field import ScalarField

# 1. Initialize Manifold from points
points = np.loadtxt('bunny.txt')
manifold = Manifold(points)

# 2. Define a Scalar Field (e.g., initial temperature)
values = np.sin(points[:, 0]) + np.cos(points[:, 1])
phi = ScalarField(manifold, values)

# 3. Access Geometric Operators
laplacian = phi.laplacian
gradient = phi.gradient
curvature = manifold.gaussian_curvature