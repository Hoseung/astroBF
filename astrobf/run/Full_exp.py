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


def add_ttype(result_arr, cat):
    """
    utility function
    """
    inds = cat['ID'].searchsorted(result_arr['id'])
    print("Is every element matched?: ", np.all(cat[inds]['ID'] == result_arr['id']))
    result_arr['ttype'] = cat[inds]['TT']

def bench_clustering(clu, data, labels):
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
    #results = [estimator[-1].inertia_] # Kmeans only.

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results = [m(labels, estimator[-1].labels_) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    results += [
        metrics.silhouette_score(data, estimator[-1].labels_,
                                 metric="euclidean", sample_size=300,)
    ]

    return results

def do_ML(result_arr, n_clusters=2, fields=['gini', 'm20', 'concentration'],
          return_cluster=False, cluster_method="ward"):
    """
    Perform clustering/classification and return a metric to optimize.

    parameters
    ----------
    return_clustering : only for post analysis purpose
    """
    compact = struct_to_ndarray(select_columns(result_arr, fields))
    
    # Binary classification 
    labels = np.array((result_arr['ttype'] >= 3) * (result_arr['ttype'] <= 6), dtype=int)
    print("Label 1 samples {}/{}".format(np.sum(labels), len(result_arr)))
    
    if cluster_method == "kmeans":
        clustering = KMeans(init="k-means++", n_clusters=n_clusters, n_init=4,
                        random_state=0)
    elif cluster_method == "ward":
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    eval_metrics = bench_clustering(clustering, compact, labels)
    if not return_cluster:
        return eval_metrics
    else:
        return eval_metrics, clustering


def evaluate(params, cluster_method="agglo"):
    result_arr = custom_morph.step_simple_morph(all_gals, params)
    if result_arr[0] == "bad":
        #print(result_arr)
        return {"mymetric": (-1, 0), "total_flux":(result_arr[1],0)}
    add_ttype(result_arr, cat)
    eval_metrics = do_ML(result_arr, fields=['gini', 'm20'], cluster_method='ward')
    silhouette_score = eval_metrics[-1]
    stderr = 0.0
    return {"mymetric": (silhouette_score, stderr), "total_flux":(1,0)}



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
