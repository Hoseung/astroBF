import numpy as np
from glob import glob
import pickle
import time

import astrobf
from astrobf.morph import custom_morph
from astrobf.utils.misc import load_Nair
from astrobf.utils.misc import struct_to_ndarray, select_columns

# SKlearn
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering, AffinityPropagation, Birch, DBSCAN, MeanShift, SpectralClustering
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from astrobf.utils import metrics as mymetrics

def add_ttype(result_arr, cat):
    """
    utility function
    """
    inds = cat['ID'].searchsorted(result_arr['id'])
    print("Is every element matched?: ", np.all(cat[inds]['ID'] == result_arr['id']))
    result_arr['ttype'] = cat[inds]['TT'] # Is it the right t-type? 

def bench_clustering(clu, data, gt_labels):
    """Benchmark to evaluate the clu initialization methods.

    Parameters
    ----------
    clu : clu instance
        A :class:`~sklearn.cluster.A_clustering_method` instance with the initialization
        already set.
    name : str
        Name given to the strategy. It will be used to show the results in a
        table.
    data : ndarray of shape (n_samples, n_features)
        The data to cluster.
    labels : ndarray of shape (n_samples,)
        The labels used to compute the clustering metrics which requires some
        supervision.
    """

    estimator = make_pipeline(StandardScaler(), clu).fit(data)
    
    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics_w_Gt = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score, # similarity b/w the two, ignoring permutation.
        metrics.adjusted_mutual_info_score, #
        metrics.fowlkes_mallows_score, 
    ] # So far, Ground Truth labels are required. 
    results = [(m.__name__, m(gt_labels, estimator[-1].labels_)) for m in clustering_metrics_w_Gt]

    # The silhouette score requires the full dataset,
    # unless I provide pairwise distances b/w samples (metric=="precomputed")
    clustering_metrics_wo_Gt = [
        metrics.silhouette_score,
        metrics.calinski_harabasz_score,
        metrics.cluster.davies_bouldin_score,
    ]
    results += [(m.__name__, m(data, estimator[-1].labels_)) for m in clustering_metrics_wo_Gt]

    return results

def do_ML(result_arr, labeler, catalog, n_clusters=2, fields=['gini', 'm20', 'concentration'],
          return_cluster=False, cluster_method="ward", eval_weight=None):
    """
    Perform clustering/classification and return a metric to optimize.
    Uses multiple sets of TMO parameters.

    parameters
    ----------
    return_clustering : only for post analysis purpose
    cluster_method: ['kmeans', 'ward', 'agglomerate', 'spectral']
    """
    compact = struct_to_ndarray(select_columns(result_arr, fields))
    
    # Binary classification 
    labels = labeler(result_arr)
    #print("Label 1 samples {}/{}".format(np.sum(labels), len(result_arr)))
    #print("cluter_method", cluster_method)
    if cluster_method == "kmeans":
        clustering = KMeans(init="k-means++", n_clusters=n_clusters, n_init=4,
                        random_state=0)
    elif cluster_method == "ward":
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    elif cluster_method == "agglomerate":
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
    elif cluster_method == 'spectral':
        clustering = SpectralClustering(n_clusters=n_clusters, assign_labels='discretize')
    eval_metrics = bench_clustering(clustering, compact, labels)
    # Add sample-weightd fowlkes_mallows_score
    if not eval_weight == None:
        eval_weight = catalog[eval_weight]
    eval_metrics.append(['sample-weighted FMS', 
                    mymetrics.fowlkes_mallows_score_w(labels, clustering.labels_, weights=eval_weight)])
    if not return_cluster:
        return eval_metrics
    else:
        return eval_metrics, clustering


def run_morph_in_parts(galaxies, catalog, plist, ngroups):
    """
    measure morphology parameters of each class and return merged array of results.
    
    parameters
    ----------
    galaxies:
        list of Galaxy data set (image, name, slice)
    catalog:
        ndarray containing ID, Label (and t-type)
    plist:
        list of tmo parameters
    ngroups:
        number of groups
    """
    assert len(plist) == ngroups, "ngroups and number of TMO parameters don't match"
    
    result_list = []
    for i in range(ngroups):
        result_list.append(custom_morph.step_simple_morph(galaxies, 
                                                          plist[i], 
                                                          np.where(catalog['label'] == i)[0]))
        if "bad" in result_list[-1]:
            return "bad"

    # sort
    result_arr = np.concatenate(result_list)
    result_arr = result_arr[np.argsort(result_arr['id'])] # Sort first to apply 'searchsorted'
    inds = result_arr['id'].searchsorted(catalog["ID"])
    result_arr = result_arr[inds]
    return result_arr


