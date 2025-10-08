import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from pydiffold.manifold import Manifold
from pydiffold.function import ScalarField


if __name__ == "__main__":
    
    test_path: str = str(Path(__file__).resolve().parent)
    points: np.array = np.loadtxt(test_path + '/assets/bunny.txt')
    manifold = Manifold(points)
    
    print(f'\033[1;95mManifold normal bundle:\033[0m\n {manifold.normal_bundle}')
    print(f'\033[1;95mManifold tangent bundle:\033[0m\n {manifold.tangent_bundle}')
    print(f'\033[1;95mMetric tensor:\033[0m\n {manifold.metric_tensor}')
    print(f'\033[1;95mInverse metric tensor:\033[0m\n {manifold.metric_tensor_inv}')
    print(f'\033[1;95mMetric tensor derivatives ∂_μ g, ∂_ν g:\033[0m\n {manifold.metric_tensor_derivatives}')
    print(f'\033[1;95mChristoffel symbols Γ^σ_μν:\033[0m\n {manifold.christoffel_symbols}')
    print(f'\033[1;95mChristoffel symbols derivatives ∂_μ Γ, ∂_ν Γ:\033[0m\n {manifold.christoffel_symbols_derivatives}')
    print(f'\033[1;95mRiemann curvature tensor:\033[0m\n {manifold.riemann_tensor}')
    print(f'\033[1;95mRicci tensor:\033[0m\n {manifold.ricci_tensor}')
    print(f'\033[1;95mRicci scalar:\033[0m\n {manifold.ricci_scalar}')
    print(f'\033[1;95mGauss curvature:\033[0m\n {manifold.gauss_curvature}')
    print(f'\033[1;95mSurface variation:\033[0m\n {manifold.surface_variation}')
    print('\n')

    geodesic, arc_length = manifold.geodesic(0, 2000)
    print(f'\033[1;95mGeodesic of arc length {arc_length}:\033[0m\n {geodesic}')
    print(f'\033[1;95mGeodesic vertex coordinates:\033[0m\n {manifold.points[geodesic]}')
    print('\n')
    
    function: ScalarField = ScalarField(manifold)
    for i in range(points.shape[0]):
        function.set_value(np.random.uniform(0, 10), i)
        
    print(f'\033[1;95mFunction values shape:\033[0m\n {function.values.shape}')
    print(f'\033[1;95mPartial derivatives ∂_μ f, ∂_ν f:\033[0m\n {function.compute_partial_derivatives()}')
    print(f'\033[1;95mSurface gradient:\033[0m\n {function.compute_surface_gradient()}')
    print(f'\033[1;95mLaplace Beltrami:\033[0m\n {function.compute_laplace_beltrami(t=1)}')