import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

sys.path.append(str(Path(__file__).resolve().parent.parent))
from riemannpy.manifold import Manifold

def sample_torus(R=3, r=1.2, n=50):
    theta = np.linspace(0, 2 * np.pi, n)
    phi = np.linspace(0, 2 * np.pi, n)
    theta, phi = np.meshgrid(theta, phi)
    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    return np.vstack([x.flatten(), y.flatten(), z.flatten()]).T

def sample_helicoid(n=50):
    u = np.linspace(-2, 2, n)
    v = np.linspace(0, 2 * np.pi, n)
    u, v = np.meshgrid(u, v)
    x = u * np.cos(v)
    y = u * np.sin(v)
    z = v
    return np.vstack([x.flatten(), y.flatten(), z.flatten()]).T

def sample_catenoid(n=50):
    # Catenoid: x = cosh(v)cos(u), y = cosh(v)sin(u), z = v
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(-1.5, 1.5, n)
    u, v = np.meshgrid(u, v)
    x = np.cosh(v) * np.cos(u)
    y = np.cosh(v) * np.sin(u)
    z = v
    return np.vstack([x.flatten(), y.flatten(), z.flatten()]).T

if __name__ == "__main__":
    
    surfaces = {
        "Torus": sample_torus(),
        "Helicoid": sample_helicoid(),
        "Catenoid": sample_catenoid()
    }

    plt.rcParams.update({'font.family': 'serif', 'font.size': 8})
    fig = plt.figure(figsize=(18, 6))

    for i, (name, pts) in enumerate(surfaces.items(), 1):
        manifold = Manifold(pts, k=30)
        curv = manifold.scalar_curvature

        ax = fig.add_subplot(1, 3, i, projection='3d')
        
        vmin, vmax = np.percentile(curv, [5, 95])
        norm = Normalize(vmin=vmin, vmax=vmax)

        sc = ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=curv, cmap='coolwarm', s=8, norm=norm, antialiased=True
        )

        ax.set_title(f"{name}\nScalar Curvature", fontsize=12)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.4, aspect=10)
        
        ax.set_box_aspect([
            pts[:, 0].max() - pts[:, 0].min(),
            pts[:, 1].max() - pts[:, 1].min(),
            pts[:, 2].max() - pts[:, 2].min()
        ])
        ax.set_axis_off()
        
        # Specific views for better geometry appreciation
        if name == "Catenoid":
            ax.view_init(elev=20, azim=30)
        elif name == "Helicoid":
            ax.view_init(elev=20, azim=-45)
        else:
            ax.view_init(elev=35, azim=45)

    plt.tight_layout()
    plt.show()