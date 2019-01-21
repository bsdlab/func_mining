import numpy as np
import os
import sklearn.metrics
import scipy.io as sio


def calculate_per_cluster_MSE(valid_component_data, cluster_labels, events_for_plotting,
                              data_base_dir):
    """calculates the average mean-squared error within each cluster 
    (including the noise cluster) based on the envelope of the original 
    time series.

    Parameters
    ----------

    valid_component_data: dict    
        dict containing all valid components
    cluster labels: np.array
        labels determined by the clustering
    events_for_plotting: list
        list of strings with event names for plotting
    data_base_dir: str
        directory to the component data base
        
    Returns
    -------
    per_cluster_MSE: np.array
        cluster-wise mean squared error on the original envelope traces 
        (also for the outlier cluster)
    envelope_data: dict
        contains the envelope traces of all configurations, the mean 
        envelope per cluster as well as the corresponding time points    

    """

    cluster_ids = np.unique(cluster_labels)
    events = ['Pause', 'GetReady', 'GoCue', '1stHit', '2ndHit', '3rdHit', '4thHit', 'TrialEnd']
    per_cluster_MSE = np.ones(len(cluster_ids))
    per_cluster_MSE.fill(np.nan)
    per_cluster_traces = []
    per_cluster_mean_trace = []

    time_points = np.empty((0, 1), dtype=np.float)
    for idx, cluster_id in enumerate(cluster_ids):
        intra_indices = (cluster_labels == cluster_id)
        per_cluster_config_rid = valid_component_data['metrics'].index[intra_indices]
        per_cluster_envelopes_list = []
        for config_id in per_cluster_config_rid:

            config_dir = os.path.join(data_base_dir, 'records', config_id, 'envelope.mat')
            envelope_cell = sio.loadmat(config_dir)
            event_indices = [i for i, item in enumerate(events) if item in events_for_plotting]
            envelope_per_event = envelope_cell['mean_envelope']
            time_points = envelope_cell['time_points']
            envelope_concat = np.array([])
            for event_index in event_indices:
                tmp = envelope_per_event[event_index, 0]
                envelope_concat = np.append(envelope_concat, [tmp])

            per_cluster_envelopes_list.append(envelope_concat)

        per_cluster_envelopes = np.asarray(per_cluster_envelopes_list)

        per_cluster_traces.append(per_cluster_envelopes)

        # calculate per cluster MSE 
        mean_per_cluster_envelope = np.mean(per_cluster_envelopes, axis=0)
        per_cluster_mean_trace.append(mean_per_cluster_envelope)
        MSE_samples = np.zeros(per_cluster_envelopes.shape[0])
        for count, single_cluster_envelopes in enumerate(per_cluster_envelopes):
            MSE_samples[count] = sklearn.metrics.mean_squared_error(mean_per_cluster_envelope, single_cluster_envelopes)
        per_cluster_MSE[idx] = np.mean(MSE_samples)

    envelope_data = {'per_cluster_traces': per_cluster_traces,
                     'per_cluster_mean_trace': per_cluster_mean_trace,
                     'time_points': time_points}

    return per_cluster_MSE, envelope_data


def calculate_per_cluster_pattern_heterogeneity(cluster_ids, pattern_angle_to_representatives,
                                                cluster_labels):
    '''
    caculates the cluster-average angle across all angles from each filter/
    pattern of a cluster to its representative
    
    Parameters
    ----------

    cluster_ids: list
        cluster ids (including outlier cluster)
    pattern_angle_to_representatives: np.array (Nconfigs x Ncluster)
        angle of each sample to the corresponding cluster representative
    cluster_labels: np.array (Nconfigs x 1)
        cluster label for each configuration
        
    Returns
    -------
    per_cluster_pattern_heterogeneity: np.array
        cluster-wise averaged angle of all cluster samples to their 
        representative 
    
    """

    '''

    per_cluster_pattern_heterogeneity = np.empty(len(cluster_ids))

    for cluster_idx, cluster_id in enumerate(cluster_ids):
        tmp = pattern_angle_to_representatives[:, cluster_idx]
        intra_cluster_mean_dist = (tmp[cluster_labels == cluster_id])
        per_cluster_pattern_heterogeneity[cluster_idx] = np.mean(intra_cluster_mean_dist)

    return per_cluster_pattern_heterogeneity
