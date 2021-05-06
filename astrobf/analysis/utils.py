import matplotlib.pyplot as plt 
import numpy as np

def get_matched_result(results, idlist):
    return [gal for gal in results if gal['id'] in idlist]

def get_typical_ind(group, n_samples=5):
    centroids = (group['gini'].mean(), group['m20'].mean()) # Simply take mean.
    from scipy.spatial import cKDTree

    kdt = cKDTree(np.vstack((group['gini'], group['m20'])).T)
    dist, indnn = kdt.query(centroids, n_samples)
    return indnn

def get_typical(group, n_samples=5, do_plot=True):
    indnn = get_typical_ind(group, n_samples)
    if do_plot:
        plt.scatter(group['gini'], group['m20'], alpha=0.2)
        plt.scatter(group['gini'][indnn], group['m20'][indnn], s=40)
        plt.show()

    select_id = group['id'][indnn]
    return [gal for gal in all_gals if gal['img_name'] in select_id]

