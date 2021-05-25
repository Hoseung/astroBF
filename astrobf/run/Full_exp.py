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

def do_ML_uni(result_arr, labeler, catalog, n_clusters=2, fields=['gini', 'm20', 'concentration'],
          return_cluster=False, cluster_method="ward", eval_weight=None):
    """
    Perform clustering/classification and return a metric to optimize.
    Uses only one set of TMO parameters.

    parameters
    ----------
    return_clustering : only for post analysis purpose
    """
    compact = struct_to_ndarray(select_columns(result_arr, fields))
    
    # Binary classification 
    labels = labeler(result_arr)
    print("Label 1 samples {}/{}".format(np.sum(labels), len(result_arr)))
    
    if cluster_method == "kmeans":
        clustering = KMeans(init="k-means++", n_clusters=n_clusters, n_init=4,
                        random_state=0)
    elif cluster_method == "ward":
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
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
    print("Label 1 samples {}/{}".format(np.sum(labels), len(result_arr)))
    print("cluter_method", cluster_method)
    if cluster_method == "kmeans":
        clustering = KMeans(init="k-means++", n_clusters=n_clusters, n_init=4,
                        random_state=0)
        print("km")
    elif cluster_method == "ward":
        print("ward")
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    elif cluster_method == "agglomerate":
        print("agg")
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
    elif cluster_method == 'spectral':
        print("spec")
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


def evaluate(param_list, cluster_method="agglomerate", eval_method='sample-weighted FMS'):
    """
    Incomplete. Refer to Jupyter version.
    """
    # Based on the original label, 
    for params in param_list:
        # label -> 0 / 1 or 
        # labeler(cat) 
        ind = np.where(cat['label'] == 1)
        result_arr = custom_morph.step_simple_morph(all_gals, params)


    if result_arr[0] == "bad":
        #print(result_arr)
        return {"mymetric": (-1, 0), "total_flux":(result_arr[1],0)}
    add_ttype(result_arr, cat)
    eval_metrics = do_ML(result_arr, fields=['gini', 'm20'], 
                         cluster_method=cluster_method,
                         eval_weight='area')
    clustering_score = [val for (name, val) in eval_metrics if name == eval_method][0]
    stderr = 0.0
    return {"mymetric": (clustering_score, stderr), "total_flux":(1,0)}


## Run
if __name__ == "__main__":
    from ax.service.ax_client import AxClient

    # raw images, masks, and weights of all 'good' galaxies.
    all_gals = pickle.load(open("../../bf_data/Nair_and_Abraham_2010/all_gals.pickle", "rb"))
    cat = load_Nair('../../bf_data/Nair_and_Abraham_2010/catalog/table2.dat')
    # Note that all_gals and cat are not visible in the following code,
    # but they are referenced as a global variable in *evaluate* and *do_ML* functions.

    # Initial evaluation result is unnecessary. 
    # result_arr = pickle.load(open("morph_init_result.pickle", "rb"))

    fields = ['gini', 'm20']#, 'concentration', 'asymmetry', 'smoothness']

    ax_client = AxClient()
    ax_client.create_experiment(
        parameters=[
            {"name": "b",
            "type": "range",
            "bounds": [1.0, 4.0],
            "value_type": "float",  # Optional, defaults to inference from type of "bounds".
            "log_scale": False},
            {"name": "c",
            "type": "range",
            "bounds": [0.5, 5.0]},
            {"name": "dl",
            "type": "range",
            "bounds": [0.5, 8.0]},
            {"name": "dh",
            "type": "range",
            "bounds": [0.5, 8.0]},
        ],
        objective_name="mymetric",
        #minimize=True,  # Optional, defaults to False. Maximize Shiloutte score
        parameter_constraints=["b - dl <= 100"], # all images are stretched to 100
        overwrite_existing_experiment =True,
        outcome_constraints=["total_flux >= 1e-5"],  # Optional.
    )

    # BO loop
    for i in range(100):
        parameters, trial_index = ax_client.get_next_trial()    
        ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))

    # Save experiment result
    ax_client.save_to_json_file("Ward_3to7_rescale100_test.json")

    # Do visualization in notebook


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
            return {"mymetric": (-1, 0), "total_flux":(0,0)}

    # sort
    result_arr = np.concatenate(result_list)
    result_arr = result_arr[np.argsort(result_arr['id'])] # Sort first to apply 'searchsorted'
    inds = result_arr['id'].searchsorted(catalog["ID"])
    result_arr = result_arr[inds]
    return result_arr