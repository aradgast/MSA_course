import numpy as np
import matplotlib.pyplot as plt


def ellips_plot_2d(m, C, binnum, c):
    """
    This function plots a 2-dimensional ellipsoid corresponding to a 2-dimensional Gaussian
    random vector.

    Usage:
    ellips_plot_2d(m, C, binnum, c)

    Inputs:
    m - 2x1 mean vector.
    C - 2x2 covariance matrix.
    binnum - number of plotting points.
    c - color of plot (string).
    """
    # Calculate the eigenvectors and eigenvalues of the covariance matrix.
    evals, evecs = np.linalg.eigh(C)
    # Sort the eigenvalues in decreasing
    idx = evals.argsort()[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    # Calculate the angle of the eigenvectors.
    angle = np.arctan2(evecs[1, 0], evecs[0, 0])
    # Plot the ellipse.
    theta = np.linspace(0, 2 * np.pi, binnum)
    x = np.sqrt(evals[0]) * np.cos(theta)
    y = np.sqrt(evals[1]) * np.sin(theta)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    x, y = np.dot(R, np.array([x, y]))
    plt.plot(x + m[0], y + m[1], c)
