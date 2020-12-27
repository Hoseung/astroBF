"""visualization.py: Plotting utilities for GMM data."""

__author__      = "Ba-Hien TRAN"
__email__       = "bahientranvn@gmail.com"


import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns


def plot_gmm_data(x, mu, sigma, num_points=2000, point_color="red"):
    """Plot the gaussian mixture model data.
    
    Args:
        x (2D numpy array): The data set.
        mu (2D numpy array): The means of the data.
        sigma (3D numpy array): The covariance of the data.
        num_points (int): Number of data points used to estimate kde.
        point_color (string): The color of datapoints.
        
    Returns:
        ax (AxesSubplot object): The plotting object.
    """
    K = mu.shape[0]
    cmaps = ["Reds", "Greens", "Blues"]
    
    # Generate data points for each cluster
    random_data = []
    for k in range(K):
        random_data.append(np.random.multivariate_normal(mu[k, :], 
			   sigma[:, :, k], num_points))

    # Plot the clusters
    for k in range(K):
        ax = sns.kdeplot(random_data[k][:, 0], random_data[k][:, 1], 
                         cmap=cmaps[k], shade=True, shade_lowest=False)
        
    # Plot the data points
    num_plt_points = min(20000, x.shape[0])
    plt.scatter(x[:num_plt_points, 0], x[:num_plt_points, 1], 
                c=point_color, marker="+", alpha=0.5, s=4)
    
    return ax


def plot_gmm_data_1D(x, mu, sigma, pi, num_points=2000, point_color="red"):
    """Plot the gaussian mixture model data.
    
    Args:
        x (1D numpy array): The data set.
        mu (1D numpy array): The means of the data.
        sigma (1D numpy array): The covariance of the data.
        num_points (int): Number of data points used to estimate kde.
        point_color (string): The color of datapoints.
        
    Returns:
        ax (AxesSubplot object): The plotting object.
    """
    K = mu.shape[0]
    cmaps = ["Reds", "Greens", "Blues"]
    
    # Generate data points for each cluster
    models = []
    for k in range(K):
        models.append(sp.stats.norm(mu[k], sigma[k]))

    # Plot the clusters
    for k in range(K):
        xp = np.arange(mu[k]-5*sigma[k], mu[k]+5*sigma[k], 0.1*sigma[k])
        ax = plt.plot(xp, pi[k]*models[k].pdf(xp))
        
    # Plot the data points
    plt.hist(x, fc=point_color, alpha=0.5, density=True, bins=200)
    
    return ax
