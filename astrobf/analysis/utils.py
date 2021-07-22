import matplotlib.pyplot as plt 
import numpy as np

def get_matched_result(results, targets):
    if isinstance(results, list):
        return [gal for gal in results if gal['id'] in targets]
    elif isinstance(results, np.ndarray):
        return results[np.searchsorted(results['id'], targets)]

def get_typical_ind(group, n_samples=5):
    centroids = (group['gini'].mean(), group['m20'].mean()) # Simply take mean.
    from scipy.spatial import cKDTree

    kdt = cKDTree(np.vstack((group['gini'], group['m20'])).T)
    dist, indnn = kdt.query(centroids, n_samples)
    return indnn

def get_typical(all_gals, group, n_samples=5, do_plot=True):
    indnn = get_typical_ind(group, n_samples)
    if do_plot:
        plt.scatter(group['gini'], group['m20'], alpha=0.2)
        plt.scatter(group['gini'][indnn], group['m20'][indnn], s=40)
        plt.show()

    return [gal for gal in all_gals if gal['img_name'] in group['id'][indnn]]

