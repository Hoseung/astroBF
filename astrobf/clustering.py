from time import time
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, Birch, DBSCAN, MeanShift, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from scipy import spatial

def dunn_index(pts, labels, centroids):
    """
    https://stackoverflow.com/a/60666838/4294919
    
    Scales poorly with large number of clusters!
    """
    # O(k n log(n)) with k clusters and n points; better performance with more even clusters
    max_intracluster_dist = max(diameter(pts[labels==i]) for i in np.unique(labels))
    # O(k^2) with k clusters; can be reduced to O(k log(k))
    # get pairwise distances between centroids
    cluster_dmat = spatial.distance_matrix(centroids, centroids)
    # fill diagonal with +inf: ignore zero distance to self in "min" computation
    np.fill_diagonal(cluster_dmat, np.inf)
    min_intercluster_dist = cluster_sizes.min()
    return min_intercluster_dist / max_intracluster_dist



def bench_clustering(clu, name, data, labels):
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

    t0 = time()
    estimator = make_pipeline(StandardScaler(), clu).fit(data)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].inertia_]

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    results += [
        metrics.silhouette_score(data, estimator[-1].labels_,
                                 metric="euclidean", sample_size=300,)
    ]

    return results




def run_bench(data, labels, n_clusters):
    """
    labels needed to evaluate some metrics requiring supervision.
    """
    print(82 * '_')
    print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')
    # Show the results
    formatter_result = ("{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}"
                        "\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}")
    
    
    kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=4,
                    random_state=0)
    results = bench_clustering(kmeans, name="k-means++", data=data, labels=labels)
    print(formatter_result.format(*results))
    
    kmeans = KMeans(init="random", n_clusters=n_clusters, n_init=4, random_state=0)
    results = bench_clustering(kmeans, name="random", data=data, labels=labels)
    print(formatter_result.format(*results))

    pca = PCA(n_components=n_clusters).fit(data)
    kmeans = KMeans(init=pca.components_, n_clusters=n_clusters, n_init=1)
    results = bench_clustering(kmeans, name="PCA-based", data=data, labels=labels)
    print(formatter_result.format(*results))

    print(82 * '_')