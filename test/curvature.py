import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm

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
    
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(-1.5, 1.5, n)
    u, v = np.meshgrid(u, v)
    
    x = np.cosh(v) * np.cos(u)
    y = np.cosh(v) * np.sin(u)
    z = v
    
    return np.vstack([x.flatten(), y.flatten(), z.flatten()]).T

def sample_dupin_cyclide(a=1.5, b=1.2, d=1.0, n=70):
    
    c = np.sqrt(a**2 - b**2)
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, 2 * np.pi, n)
    u, v = np.meshgrid(u, v)
    denom = a - c * np.cos(u) * np.cos(v)
    
    x = (d * (c - a * np.cos(u) * np.cos(v)) + b**2 * np.cos(u)) / denom
    y = (b * np.sin(u) * (a - d * np.cos(v))) / denom
    z = (b * np.sin(v) * (c * np.cos(u) - d)) / denom
    
    return np.vstack([x.flatten(), y.flatten(), z.flatten()]).T

if __name__ == "__main__":
    
    surfaces = {
        "Torus": sample_torus(),
        "Helicoid": sample_helicoid(),
        "Catenoid": sample_catenoid(),
        "Dupin Cyclide": sample_dupin_cyclide()
    }

    plt.rcParams.update({'font.family': 'serif', 'font.size': 9})
    fig = plt.figure(figsize=(20, 6))

    for i, (name, pts) in enumerate(surfaces.items(), 1):
        
        # Manifold and curvature
        manifold = Manifold(pts, k=30)
        curv = manifold.scalar_curvature

        # Plots
        ax = fig.add_subplot(1, 4, i, projection='3d')
        
        # Custom normalization
        if name == "Dupin Cyclide":
            vmin, vmax = np.percentile(curv, [15, 85])
            norm = PowerNorm(gamma=0.7, vmin=vmin, vmax=vmax)
            s_size = 3
        else:
            vmin, vmax = np.percentile(curv, [5, 95])
            norm = Normalize(vmin=vmin, vmax=vmax)
            s_size = 5

        sc = ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=curv, 
            cmap='coolwarm', 
            s=s_size, 
            norm=norm, 
            antialiased=True,
            alpha=0.7
        )

        ax.set_title(f"{name}\nScalar Curvature", fontsize=13, pad=10)
        
        extents = np.array([
            pts[:, 0].max() - pts[:, 0].min(),
            pts[:, 1].max() - pts[:, 1].min(),
            pts[:, 2].max() - pts[:, 2].min()
        ])
        ax.set_box_aspect(extents)
        ax.set_axis_off()
        
        if name == "Catenoid": ax.view_init(elev=25, azim=40)
        elif name == "Helicoid": ax.view_init(elev=30, azim=-60)
        elif name == "Dupin Cyclide": ax.view_init(elev=55, azim=58)
        else: ax.view_init(elev=35, azim=45)

    plt.tight_layout()
    plt.show()