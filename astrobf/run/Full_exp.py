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
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn import svm

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

def bench_classification(pred, labels, weight=None, avg='macro'):
    """
    All metrics should scale 0(worst) to 1(best).
    The first entry will be used as the BO optimization target metric.
    Others are only for reference.
    """
    clf_metrics_hardmax=[
        (metrics.f1_score, {'sample_weight':weight,'average':'macro'}),
        (metrics.accuracy_score, {'sample_weight':weight, 'normalize':True}),
        (metrics.balanced_accuracy_score, {'sample_weight':weight, 'adjusted':True}),
        #(metrics.average_precision_score, {'sample_weight':weight, 'average':avg}),
        #(metrics.f1_score, {'sample_weight':weight,'average':'micro'}),
        (metrics.precision_score, {'sample_weight':weight, 'average':avg}),
        (metrics.recall_score, {'sample_weight':weight, 'average':avg}),
        (metrics.jaccard_score, {'sample_weight':weight, 'average':avg})
        ]
    clf_metrics_prob = [metrics.brier_score_loss, 
                        metrics.log_loss, 
                        ]
    results = [(m[0].__name__, m[0](pred, labels, **m[1])) for m in clf_metrics_hardmax]
    
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

    if not eval_weight == None:
        eval_weight = catalog[eval_weight]
    
    _classification = False
    if cluster_method == "kmeans":
        clustering = KMeans(init="k-means++", n_clusters=n_clusters, n_init=4,
                        random_state=0)
    elif cluster_method == "ward":
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    elif cluster_method == "agglomerate":
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
    elif cluster_method == 'spectral':
        clustering = SpectralClustering(n_clusters=n_clusters, assign_labels='discretize')
    elif cluster_method == "SVM":
        _classification = True
        clf = svm.SVC(gamma=1, decision_function_shape='ovr')
    
    if _classification:    
        clf.fit(compact, labels, sample_weight=eval_weight)

        preds = clf.predict(compact)
        """
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        pipeline = Pipeline([
            ('scaler',StandardScaler()),
            ('clf', svm.SVC(gamma=1, decision_function_shape='ovr'))
            ])

        # use the pipeline object as you would
        # a regular classifier
        pipeline.fit(compact,labels)


        preds = pipeline.predict(X_test)

        """
        eval_metrics=bench_classification(preds, labels)
        #clustering = None
        return eval_metrics, preds
    else:
        eval_metrics = bench_clustering(clustering, compact, labels)
        eval_metrics.append(['sample-weighted FMS', 
                mymetrics.fowlkes_mallows_score_w(labels, clustering.labels_, weights=eval_weight)])
        
        if not return_cluster:
            return eval_metrics
        else:
            return eval_metrics, clustering




