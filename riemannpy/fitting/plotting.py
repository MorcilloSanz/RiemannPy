import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_sphere(points, color='b', size=20):
    """
    Plot 3D points on a sphere.

    Args:  
        points (np.array): Array of shape (n, 3) containing 3D points on the sphere.  
        color (str, optional): Color of the points. Defaults to 'b'.  
        size (int, optional): Size of the points. Defaults to 20.  
    """  
    fig = plt.figure(figsize=(8,6))  
    ax = fig.add_subplot(111, projection='3d')  

    ax.scatter(points[:,0], points[:,1], points[:,2], color=color, s=size)  

    ax.set_xlabel('X')  
    ax.set_ylabel('Y')  
    ax.set_zlabel('Z')  
    ax.set_box_aspect([1,1,1])
    plt.show()  


def plot_paraboloid(points_local, params, resolution=30):
    """
    Plot the local points and the fitted paraboloid.

    Args:  
        points_local (np.array): Local 3D points used for fitting, shape (n, 3).  
        params (tuple): Paraboloid parameters (a, b, c, d, e, f).  
        resolution (int, optional): Grid resolution for the paraboloid surface. Default is 30.  
    """  
    a, b, c, d, e, f = params  

    # Extract x and y coordinates  
    x = points_local[:, 0]  
    y = points_local[:, 1]  

    # Create meshgrid  
    x_grid = np.linspace(x.min(), x.max(), resolution)  
    y_grid = np.linspace(y.min(), y.max(), resolution)  
    X, Y = np.meshgrid(x_grid, y_grid)  

    # Compute Z values from the paraboloid equation: z = a*x^2 + b*y^2 + c*xy + d*x + e*y + f  
    Z = a*X**2 + b*Y**2 + c*X*Y + d*X + e*Y + f  

    # Plot  
    fig = plt.figure(figsize=(8,6))  
    ax = fig.add_subplot(111, projection='3d')  
    ax.scatter(points_local[:,0], points_local[:,1], points_local[:,2], color='r', label='Local points')  
    ax.plot_surface(X, Y, Z, color='b', alpha=0.5, rstride=1, cstride=1)  

    ax.set_xlabel('X')  
    ax.set_ylabel('Y')  
    ax.set_zlabel('Z')  
    ax.legend()  
    plt.show()  
