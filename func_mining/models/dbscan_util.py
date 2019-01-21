import numpy as np
import sklearn.cluster
import sklearn.metrics


def cluster_representatives(dbscan_instance, features, component_data,
                            use_core_samples=True):
    """Extracts cluster representatives for the dbscan method using different
    strategies.

    Parameters
    ----------
    dbscan_instance : scikit-learn instance
        trained DBSCAN model
    features: np.array     
        containing all features for the clustering
    component_data: pandas dataframe
        dataframe containing all available information on components 
    use_core_samples: bool, optional
        select if metric for representative selection is only applied to 
        core samples
        
    Returns
    -------
    cluster_ids_repr : np.array 
        cluster labels for which a representative was found
    cluster_representatives : dict 
        containing the config-id of each cluster representative
    is_representative: bool array (size: dscan_instance.labels_)
        flag for each representative component
    """

    cluster_labels = dbscan_instance.labels_
    cluster_representatives = {'config_id': {}}
    cluster_ids_repr = []
    is_representative = np.zeros_like(cluster_labels, dtype=bool)

    cluster_ids = np.unique(cluster_labels)

    if use_core_samples:
        is_core_sample = np.zeros_like(cluster_labels, dtype=bool)
        is_core_sample[dbscan_instance.core_sample_indices_] = True
    else:
        is_core_sample = np.ones_like(cluster_labels, dtype=bool)

    cluster_representatives['min_intra_euclidean_dist'] = {}

    paired_distances = sklearn.metrics.pairwise_distances(features, metric='euclidean')

    for cluster_id in cluster_ids:
        intra_indices = (cluster_labels == cluster_id) & is_core_sample

        cluster_distances = paired_distances[intra_indices, :]

        intra_cluster_mean_dist = np.mean(cluster_distances[:, intra_indices], axis=1)
        assert len(component_data['features'].index) == paired_distances.shape[0]

        best_score = np.amin(intra_cluster_mean_dist)
        cluster_repr = component_data['features'].index[intra_indices][np.argmin(intra_cluster_mean_dist)]
        intra_indices_list = np.where(intra_indices)[0]
        repr_indices = intra_indices_list[np.argmin(intra_cluster_mean_dist)]

        is_representative[repr_indices] = True
        cluster_representatives['min_intra_euclidean_dist'][cluster_id] = best_score
        cluster_representatives['config_id'][cluster_id] = cluster_repr
        cluster_ids_repr.append(cluster_id)

    assert np.count_nonzero(is_representative, axis=0) == len(cluster_ids_repr)

    is_representative = is_representative * 1

    return cluster_ids_repr, cluster_representatives, is_representative


def optimize_dbscan_hyperparameter(features, epsilon_list, min_samples=20,
                                   clustering_distance_metric='euclidean',
                                   silhouette_threshold=0.2):
    """Optimizes the epsilon hyperparameter of DBSCAN clustering from 
    a given list and returns the optimal hyperparameter as well as the 
    trained model. As optimization criterion, the number of clusters with 
    a silhouette value above a given threshold is utilized.

    Parameters
    ----------

    features: np.array     
        containing all features for the clustering
    epsilon_list: list
        all epsilon values to evaluate
    min_samples: int, optional
        DBSCAN hyperparameter
    clustering_distance_metric: string, optional
        distance metric for DBSCAN clustering
    silhouette_threshold: float, optional
        threshold silhouette score to compute number of homogeneous 
        clusters for each parameter configuration
        
    Returns
    -------
    best_epsiolon: float
        selected hyperparameter
    cluster_instance: scikit-learn instance 
        selected clustering model trained with selected hyperparameter
    sweep_results: dict
        containing various metrics that characterize the parameter sweep
    """
    cluster_counts = np.zeros_like(epsilon_list)
    outlier_counts = np.zeros_like(epsilon_list)
    silhouette_coefficients = np.zeros_like(epsilon_list, dtype=float)
    cluster_counts_meaningful = np.zeros_like(epsilon_list)

    max_expected_clusters = 200
    cluster_member_sizes = np.zeros((len(epsilon_list), max_expected_clusters))

    # (1) loop over different epsilon values
    for index, epsilon in enumerate(epsilon_list):

        cluster_instance = sklearn.cluster.DBSCAN(eps=epsilon, min_samples=min_samples,
                                                  metric=clustering_distance_metric)

        cluster_instance.fit(features)
        cluster_count = len(set(cluster_instance.labels_)) - (1 if -1 in cluster_instance.labels_ else 0)

        outlier_count = np.sum(cluster_instance.labels_ == -1)

        cluster_counts[index] = cluster_count
        outlier_counts[index] = outlier_count
        cluster_memberships = np.bincount(cluster_instance.labels_[cluster_instance.labels_ != -1])
        cluster_member_sizes[index, 0:cluster_memberships.size] = cluster_memberships
        if cluster_count > 1:

            sample_silhouette_values = sklearn.metrics.silhouette_samples(features, cluster_instance.labels_)
            silhouette_coefficients[index] = np.mean(sample_silhouette_values[cluster_instance.labels_ != -1])

            cluster_labels_unique = np.unique(cluster_instance.labels_[cluster_instance.labels_ != -1])
            counter = 0
            for cl_no in np.nditer(cluster_labels_unique):
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_instance.labels_ == cl_no]
                if np.amin(ith_cluster_silhouette_values) > silhouette_threshold:
                    counter += 1
            cluster_counts_meaningful[index] = counter

            # (2) select hyperparameter and train model
    best_epsilon = epsilon_list[np.argmax(cluster_counts_meaningful)]
    cluster_instance = sklearn.cluster.DBSCAN(eps=best_epsilon, min_samples=min_samples,
                                              metric=clustering_distance_metric)

    sweep_results = {'cluster_member_sizes': cluster_member_sizes,
                     'cluster_counts': cluster_counts,
                     'outlier_counts': outlier_counts,
                     'cluster_counts_meaningful': cluster_counts_meaningful}

    print("selected value: {}".format(best_epsilon))

    return best_epsilon, cluster_instance, sweep_results
