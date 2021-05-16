import numpy as np
import scipy as sp
from math import log
from sklearn.metrics.cluster._supervised import check_clusterings, entropy, contingency_matrix
from sklearn.utils.validation import check_array
from sklearn.utils.fixes import _astype_copy_false

def mutual_info_score(labels_true, labels_pred, *, contingency=None, weights=None):
    """Mutual Information between two clusterings.
    The Mutual Information is a measure of the similarity between two labels of
    the same data. Where :math:`|U_i|` is the number of the samples
    in cluster :math:`U_i` and :math:`|V_j|` is the number of the
    samples in cluster :math:`V_j`, the Mutual Information
    between clusterings :math:`U` and :math:`V` is given as:
    .. math::
        MI(U,V)=\\sum_{i=1}^{|U|} \\sum_{j=1}^{|V|} \\frac{|U_i\\cap V_j|}{N}
        \\log\\frac{N|U_i \\cap V_j|}{|U_i||V_j|}
    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.
    This metric is furthermore symmetric: switching ``label_true`` with
    ``label_pred`` will return the same score value. This can be useful to
    measure the agreement of two independent label assignments strategies
    on the same dataset when the real ground truth is not known.
    Read more in the :ref:`User Guide <mutual_info_score>`.
    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        A clustering of the data into disjoint subsets.
    labels_pred : int array-like of shape (n_samples,)
        A clustering of the data into disjoint subsets.
    contingency : {ndarray, sparse matrix} of shape \
            (n_classes_true, n_classes_pred), default=None
        A contingency matrix given by the :func:`contingency_matrix` function.
        If value is ``None``, it will be computed, otherwise the given value is
        used, with ``labels_true`` and ``labels_pred`` ignored.
    Returns
    -------
    mi : float
       Mutual information, a non-negative value
    Notes
    -----
    The logarithm used is the natural logarithm (base-e).
    See Also
    --------
    adjusted_mutual_info_score : Adjusted against chance Mutual Information.
    normalized_mutual_info_score : Normalized Mutual Information.
    """
    if contingency is None:
        labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
        contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
    else:
        contingency = check_array(contingency,
                                  accept_sparse=['csr', 'csc', 'coo'],
                                  dtype=[int, np.int32, np.int64])

    if isinstance(contingency, np.ndarray):
        # For an array
        nzx, nzy = np.nonzero(contingency)
        nz_val = contingency[nzx, nzy]
    elif sp.issparse(contingency):
        # For a sparse matrix
        nzx, nzy, nz_val = sp.find(contingency)
    else:
        raise ValueError("Unsupported type for 'contingency': %s" %
                         type(contingency))

    contingency_sum = contingency.sum()
    pi = np.ravel(contingency.sum(axis=1))
    pj = np.ravel(contingency.sum(axis=0))
    log_contingency_nm = np.log(nz_val)
    contingency_nm = nz_val / contingency_sum
    # Don't need to calculate the full outer product, just for non-zeroes
    outer = (pi.take(nzx).astype(np.int64, copy=False)
             * pj.take(nzy).astype(np.int64, copy=False))
    log_outer = -np.log(outer) + log(pi.sum()) + log(pj.sum())
    mi = (contingency_nm * (log_contingency_nm - log(contingency_sum)) +
          contingency_nm * log_outer)
    mi = np.where(np.abs(mi) < np.finfo(mi.dtype).eps, 0.0, mi)
    return np.clip(mi.sum(), 0.0, None)


def homogeneity_completeness_v_measure(labels_true, labels_pred, *, beta=1.0, weights=None):
    """Compute the homogeneity and completeness and V-Measure scores at once.
    Those metrics are based on normalized conditional entropy measures of
    the clustering labeling to evaluate given the knowledge of a Ground
    Truth class labels of the same samples.
    A clustering result satisfies homogeneity if all of its clusters
    contain only data points which are members of a single class.
    A clustering result satisfies completeness if all the data points
    that are members of a given class are elements of the same cluster.
    Both scores have positive values between 0.0 and 1.0, larger values
    being desirable.
    Those 3 metrics are independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score values in any way.
    V-Measure is furthermore symmetric: swapping ``labels_true`` and
    ``label_pred`` will give the same score. This does not hold for
    homogeneity and completeness. V-Measure is identical to
    :func:`normalized_mutual_info_score` with the arithmetic averaging
    method.
    Read more in the :ref:`User Guide <homogeneity_completeness>`.
    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        ground truth class labels to be used as a reference
    labels_pred : array-like of shape (n_samples,)
        cluster labels to evaluate
    beta : float, default=1.0
        Ratio of weight attributed to ``homogeneity`` vs ``completeness``.
        If ``beta`` is greater than 1, ``completeness`` is weighted more
        strongly in the calculation. If ``beta`` is less than 1,
        ``homogeneity`` is weighted more strongly.
    Returns
    -------
    homogeneity : float
       score between 0.0 and 1.0. 1.0 stands for perfectly homogeneous labeling
    completeness : float
       score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling
    v_measure : float
        harmonic mean of the first two
    See Also
    --------
    homogeneity_score
    completeness_score
    v_measure_score
    """
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)

    if len(labels_true) == 0:
        return 1.0, 1.0, 1.0

    entropy_C = entropy(labels_true)
    entropy_K = entropy(labels_pred)

    contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
    MI = mutual_info_score(None, None, contingency=contingency)

    homogeneity = MI / (entropy_C) if entropy_C else 1.0
    completeness = MI / (entropy_K) if entropy_K else 1.0

    if homogeneity + completeness == 0.0:
        v_measure_score = 0.0
    else:
        v_measure_score = ((1 + beta) * homogeneity * completeness
                           / (beta * homogeneity + completeness))

    return homogeneity, completeness, v_measure_score


def homogeneity_score(labels_true, labels_pred, **kwargs):
    """Homogeneity metric of a cluster labeling given a ground truth.
    A clustering result satisfies homogeneity if all of its clusters
    contain only data points which are members of a single class.
    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.
    This metric is not symmetric: switching ``label_true`` with ``label_pred``
    will return the :func:`completeness_score` which will be different in
    general.
    Read more in the :ref:`User Guide <homogeneity_completeness>`.
    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        ground truth class labels to be used as a reference
    labels_pred : array-like of shape (n_samples,)
        cluster labels to evaluate
    Returns
    -------
    homogeneity : float
       score between 0.0 and 1.0. 1.0 stands for perfectly homogeneous labeling
    References
    ----------
    .. [1] `Andrew Rosenberg and Julia Hirschberg, 2007. V-Measure: A
       conditional entropy-based external cluster evaluation measure
       <https://aclweb.org/anthology/D/D07/D07-1043.pdf>`_
    See Also
    --------
    completeness_score
    v_measure_score
    Examples
    --------
    Perfect labelings are homogeneous::
      >>> from sklearn.metrics.cluster import homogeneity_score
      >>> homogeneity_score([0, 0, 1, 1], [1, 1, 0, 0])
      1.0
    Non-perfect labelings that further split classes into more clusters can be
    perfectly homogeneous::
      >>> print("%.6f" % homogeneity_score([0, 0, 1, 1], [0, 0, 1, 2]))
      1.000000
      >>> print("%.6f" % homogeneity_score([0, 0, 1, 1], [0, 1, 2, 3]))
      1.000000
    Clusters that include samples from different classes do not make for an
    homogeneous labeling::
      >>> print("%.6f" % homogeneity_score([0, 0, 1, 1], [0, 1, 0, 1]))
      0.0...
      >>> print("%.6f" % homogeneity_score([0, 0, 1, 1], [0, 0, 0, 0]))
      0.0...
    """
    return homogeneity_completeness_v_measure(labels_true, labels_pred, **kwargs)[0]


def completeness_score(labels_true, labels_pred, **kwargs):
    """Completeness metric of a cluster labeling given a ground truth.
    A clustering result satisfies completeness if all the data points
    that are members of a given class are elements of the same cluster.
    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.
    This metric is not symmetric: switching ``label_true`` with ``label_pred``
    will return the :func:`homogeneity_score` which will be different in
    general.
    Read more in the :ref:`User Guide <homogeneity_completeness>`.
    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        ground truth class labels to be used as a reference
    labels_pred : array-like of shape (n_samples,)
        cluster labels to evaluate
    Returns
    -------
    completeness : float
       score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling
    References
    ----------
    .. [1] `Andrew Rosenberg and Julia Hirschberg, 2007. V-Measure: A
       conditional entropy-based external cluster evaluation measure
       <https://aclweb.org/anthology/D/D07/D07-1043.pdf>`_
    See Also
    --------
    homogeneity_score
    v_measure_score
    Examples
    --------
    Perfect labelings are complete::
      >>> from sklearn.metrics.cluster import completeness_score
      >>> completeness_score([0, 0, 1, 1], [1, 1, 0, 0])
      1.0
    Non-perfect labelings that assign all classes members to the same clusters
    are still complete::
      >>> print(completeness_score([0, 0, 1, 1], [0, 0, 0, 0]))
      1.0
      >>> print(completeness_score([0, 1, 2, 3], [0, 0, 1, 1]))
      0.999...
    If classes members are split across different clusters, the
    assignment cannot be complete::
      >>> print(completeness_score([0, 0, 1, 1], [0, 1, 0, 1]))
      0.0
      >>> print(completeness_score([0, 0, 0, 0], [0, 1, 2, 3]))
      0.0
    """
    return homogeneity_completeness_v_measure(labels_true, labels_pred, **kwargs)[1]


def v_measure_score(labels_true, labels_pred, *, beta=1.0, **kwargs):
    """V-measure cluster labeling given a ground truth.
    This score is identical to :func:`normalized_mutual_info_score` with
    the ``'arithmetic'`` option for averaging.
    The V-measure is the harmonic mean between homogeneity and completeness::
        v = (1 + beta) * homogeneity * completeness
             / (beta * homogeneity + completeness)
    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.
    This metric is furthermore symmetric: switching ``label_true`` with
    ``label_pred`` will return the same score value. This can be useful to
    measure the agreement of two independent label assignments strategies
    on the same dataset when the real ground truth is not known.
    Read more in the :ref:`User Guide <homogeneity_completeness>`.
    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        ground truth class labels to be used as a reference
    labels_pred : array-like of shape (n_samples,)
        cluster labels to evaluate
    beta : float, default=1.0
        Ratio of weight attributed to ``homogeneity`` vs ``completeness``.
        If ``beta`` is greater than 1, ``completeness`` is weighted more
        strongly in the calculation. If ``beta`` is less than 1,
        ``homogeneity`` is weighted more strongly.
    Returns
    -------
    v_measure : float
       score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling
    References
    ----------
    .. [1] `Andrew Rosenberg and Julia Hirschberg, 2007. V-Measure: A
       conditional entropy-based external cluster evaluation measure
       <https://aclweb.org/anthology/D/D07/D07-1043.pdf>`_
    See Also
    --------
    homogeneity_score
    completeness_score
    normalized_mutual_info_score
    Examples
    --------
    Perfect labelings are both homogeneous and complete, hence have score 1.0::
      >>> from sklearn.metrics.cluster import v_measure_score
      >>> v_measure_score([0, 0, 1, 1], [0, 0, 1, 1])
      1.0
      >>> v_measure_score([0, 0, 1, 1], [1, 1, 0, 0])
      1.0
    Labelings that assign all classes members to the same clusters
    are complete be not homogeneous, hence penalized::
      >>> print("%.6f" % v_measure_score([0, 0, 1, 2], [0, 0, 1, 1]))
      0.8...
      >>> print("%.6f" % v_measure_score([0, 1, 2, 3], [0, 0, 1, 1]))
      0.66...
    Labelings that have pure clusters with members coming from the same
    classes are homogeneous but un-necessary splits harms completeness
    and thus penalize V-measure as well::
      >>> print("%.6f" % v_measure_score([0, 0, 1, 1], [0, 0, 1, 2]))
      0.8...
      >>> print("%.6f" % v_measure_score([0, 0, 1, 1], [0, 1, 2, 3]))
      0.66...
    If classes members are completely split across different clusters,
    the assignment is totally incomplete, hence the V-Measure is null::
      >>> print("%.6f" % v_measure_score([0, 0, 0, 0], [0, 1, 2, 3]))
      0.0...
    Clusters that include samples from totally different classes totally
    destroy the homogeneity of the labeling, hence::
      >>> print("%.6f" % v_measure_score([0, 0, 1, 1], [0, 0, 0, 0]))
      0.0...
    """
    return homogeneity_completeness_v_measure(labels_true, labels_pred, **kwargs)[2]


def fowlkes_mallows_score_w(labels_true, labels_pred, *, sparse=False, weights=None):
    """Measure the similarity of two clusterings of a set of points.
    .. versionadded:: 0.18
    The Fowlkes-Mallows index (FMI) is defined as the geometric mean between of
    the precision and recall::
        FMI = TP / sqrt((TP + FP) * (TP + FN))
    Where ``TP`` is the number of **True Positive** (i.e. the number of pair of
    points that belongs in the same clusters in both ``labels_true`` and
    ``labels_pred``), ``FP`` is the number of **False Positive** (i.e. the
    number of pair of points that belongs in the same clusters in
    ``labels_true`` and not in ``labels_pred``) and ``FN`` is the number of
    **False Negative** (i.e the number of pair of points that belongs in the
    same clusters in ``labels_pred`` and not in ``labels_True``).
    The score ranges from 0 to 1. A high value indicates a good similarity
    between two clusters.
    Read more in the :ref:`User Guide <fowlkes_mallows_scores>`.
    Parameters
    ----------
    labels_true : int array, shape = (``n_samples``,)
        A clustering of the data into disjoint subsets.
    labels_pred : array, shape = (``n_samples``, )
        A clustering of the data into disjoint subsets.
    sparse : bool, default=False
        Compute contingency matrix internally with sparse matrix.
    Returns
    -------
    score : float
       The resulting Fowlkes-Mallows score.
    Examples
    --------
    Perfect labelings are both homogeneous and complete, hence have
    score 1.0::
      >>> from sklearn.metrics.cluster import fowlkes_mallows_score
      >>> fowlkes_mallows_score([0, 0, 1, 1], [0, 0, 1, 1])
      1.0
      >>> fowlkes_mallows_score([0, 0, 1, 1], [1, 1, 0, 0])
      1.0
    If classes members are completely split across different clusters,
    the assignment is totally random, hence the FMI is null::
      >>> fowlkes_mallows_score([0, 0, 0, 0], [0, 1, 2, 3])
      0.0
    References
    ----------
    .. [1] `E. B. Fowkles and C. L. Mallows, 1983. "A method for comparing two
       hierarchical clusterings". Journal of the American Statistical
       Association
       <http://wildfire.stat.ucla.edu/pdflibrary/fowlkes.pdf>`_
    .. [2] `Wikipedia entry for the Fowlkes-Mallows Index
           <https://en.wikipedia.org/wiki/Fowlkes-Mallows_index>`_
    """
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
 
    n_samples, = labels_true.shape
    if weights is None:
        weights = np.ones(n_samples)

    u_labels_t = np.unique(labels_true)
    u_labels_p = np.unique(labels_pred)

    c = np.zeros((len(u_labels_t), len(u_labels_t)))

    # from smaller label to larger lable (0 -> N)
    for i, l1 in enumerate(u_labels_t):
        for j, l2 in enumerate(u_labels_t):
            c[i,j] = np.sum(weights[(labels_true == l1) * (labels_pred == l2)])

    tk = np.dot(c.ravel(), c.ravel()) - n_samples # c.ravel instead of c.data
    pk = np.sum(np.asarray(c.sum(axis=0)).ravel() ** 2) - n_samples 
    qk = np.sum(np.asarray(c.sum(axis=1)).ravel() ** 2) - n_samples
    if np.sum(tk) == 0:
        return 0
    else:
        return np.sqrt(tk / pk) * np.sqrt(tk / qk) 