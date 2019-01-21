import pandas as pd
import matplotlib.patches
import matplotlib.pyplot as plt
import math
import mne
import numpy as np
import seaborn as sns
import sklearn.neighbors


def plot_cluster_sizes(cluster_labels, cluster_membership_counts,
                       sort_by_size=False, clustering_axis_str=None):
    """
    Plot cluster sizes for a parameter sweep (e.g. epsilon for DBSCAN).
    
    Parameters
    ----------
    
    cluster_labels: np.array
        cluster labels 
        
    cluster_membership_counts: np.array
        number of samples in each cluster for every parameter config
                    
    """
    cluster_membership_counts_mat = cluster_membership_counts.copy()
    if cluster_membership_counts_mat.ndim == 1:
        cluster_membership_counts_mat = np.reshape(cluster_membership_counts_mat, (1, -1))  # one row per clustering
    # drop clusters that never have any members
    completely_empty_columns = np.all(cluster_membership_counts_mat == 0, axis=0)
    cluster_membership_counts_mat = cluster_membership_counts_mat[:, np.logical_not(completely_empty_columns)]
    if sort_by_size:  # reverse sort of each row (clustering)
        cluster_membership_counts_mat = np.sort(cluster_membership_counts_mat, axis=1)[:, ::-1]

    plot_stacked_barchart(x=cluster_labels, y=cluster_membership_counts_mat, x_axis_str=clustering_axis_str,
                          plot_legend=True, legend_prefix="Cluster")


def plot_cluster_results_for_parameter_sweep(sweep_results, epsilon_list):
    """
    Plot cluster size, number of outliers and number of homogeneous 
    clusters for the parameter sweep.
    
    Parameters
    ----------
    
    sweep_results: dict
        containing different cluster metrics 
        
    epsilon_list: np.array
        evaluated epislon parameters
                    
    """

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.plot(epsilon_list, sweep_results['cluster_counts'], label="cluster count", marker="v")
    ax2.plot(epsilon_list, sweep_results['outlier_counts'], label="outlier count", marker="v")
    ax3.plot(epsilon_list, sweep_results['cluster_counts_meaningful'], label="# homogeneous clusters", marker="v")
    for ax in (ax1, ax2, ax3):
        ax.legend()
        ax.set_xlabel('epsilon')
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        ax.yaxis.set_major_locator(plt.MaxNLocator(10))


def plot_stacked_barchart(x, y, x_axis_str=None, palette_name="colorblind",
                          plot_legend=False, legend_prefix=""):
    """
    Stacked bar chart 
            
    """
    y_cum = np.cumsum(y, axis=1)  # for plotting stacked bar chart
    bar_colors = sns.color_palette(palette_name, y_cum.shape[1])
    for stack_idx in reversed(range(y_cum.shape[1])):
        sns.barplot(x=x, y=y_cum[:, stack_idx], color=bar_colors[stack_idx])
    if x_axis_str:
        plt.xlabel(x_axis_str)
    # plt.yticks
    plt.ylim((0, int(np.max(np.max(y_cum, axis=1)) * 1.1)))
    legend_patches = [matplotlib.patches.Patch(color=cl_col, label="%s %d" % (legend_prefix, cl_no))
                      for cl_no, cl_col in enumerate(bar_colors)]
    if plot_legend:
        plt.legend(handles=legend_patches, ncol=int(math.ceil(len(legend_patches) / 2)), loc="upper center")


def plot_clusterwise_param_distribution(cluster_labels, input_df, feature_name,
                                        log_scaling=False, yLim=None):
    """
    Plot per-cluster distribution / scatter plot of a certain parameter
    (e.g. the underlying frequency band where each spatial filter was 
    extracted from).
    
    Parameters
    ----------
    
    cluster_labels: np.array
        cluster labels 
        
    input_df: dataframe
        data frame containing the parameter of interest
            
    feature_name : str
        name of the feature contained as key in input_df
        
    """

    parameters_df = input_df.copy()
    parameters_df['cluster'] = cluster_labels
    parameters_df[feature_name] = parameters_df[feature_name].apply(pd.to_numeric)
    plt.figure(figsize=(7, 5))
    sns.stripplot(x="cluster", y=feature_name, data=parameters_df, jitter=True)
    if log_scaling:
        plt.gca().set_yscale('log')
    if yLim is not None:
        plt.ylim(yLim)


def plot_silhouette_coefficients(cluster_labels, sample_silhouette_values, silhouette_avg,
                                 palette_name="colorblind", plot_legend=False):
    """
    Plot silhouette coefficients for all samples of a clustering color 
    coded by their underlying cluster.
    
    Parameters
    ----------
    
    cluster_labels: np.array
        cluster labels 
        
    sample_silhouette_values: np.array
        silhouette value for each sample
    
    silhouette_avg : float
        silhouette average across all samples
        
    """
    cluster_labels_unique = np.unique(cluster_labels)

    plt.figure(figsize=(7, 7))
    ax = plt.gca()
    ax.set_xlim([-1, 1])
    ax.set_ylim([0, len(cluster_labels) + (len(cluster_labels_unique)) * 25])

    bar_colors = sns.color_palette(palette_name, len(cluster_labels_unique))
    y_lower = 10
    color_idx = 0

    for cl_no in np.nditer(cluster_labels_unique):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == cl_no]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=bar_colors[color_idx],
                         edgecolor=bar_colors[color_idx], alpha=0.7)
        color_idx += 1

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10

    ax.set_xlabel("silhouette coefficient values")
    ax.set_ylabel("data points")
    ax.set_yticks([])  # Clear the yaxis labels / ticks

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="black", linestyle="--")

    if plot_legend:
        legend_prefix = "cluster"
        if -1 in cluster_labels:
            legend_patches = [matplotlib.patches.Patch(color=cl_col, label="%s %d" % (legend_prefix, cl_no - 1))
                              for cl_no, cl_col in enumerate(bar_colors)]
        else:
            legend_patches = [matplotlib.patches.Patch(color=cl_col, label="%s %d" % (legend_prefix, cl_no))
                              for cl_no, cl_col in enumerate(bar_colors)]
        plt.legend(handles=legend_patches, ncol=int(math.ceil(len(legend_patches) / 2)), loc="lower center")


def plot_kNN_distance(features, N_neighbors, distance_metric='euclidean',
                      epsilon_lower_percentile=2, whiskerlength=3):
    """
    Plot kNN-distance (as proposed in the original DBSCAN paper by Ester et
    al.) to determine a suitable range for the epsilon hyperparameter.

    Parameters
    ----------
    
    features: np.array
        feature set E for DBSCAN clustering
        
    N_neighbours: int
        number of neighbours considered for k-NN distance plot
    
    distance_metric : str
        distance metric for the kNN-plot
        
    epsilon_lower_percentile : int
        percentile to determine lower limit for epsilon range
        
    whiskerlength : float
        parameter for the "knee"/slope increase detection of the kNN-plot
        
        
    Returns
    ----------
    
    kNN_distance: np.array
        sorted kNN distances of all samples
    
    epsilon_range: np.array
        lower and upper boundary of epsilon values
    
    """

    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=N_neighbors,
                                              algorithm='auto',
                                              metric=distance_metric).fit(features)
    distances, indices = nbrs.kneighbors(features)
    kNN_distance = np.sort(distances[:, N_neighbors - 1])

    epsilon_range = np.empty([2, 1])
    epsilon_range[0] = np.percentile(kNN_distance, epsilon_lower_percentile)

    # plot slope to identify "knee"
    slope = np.gradient(kNN_distance)
    # knee detection by variance criterion 
    low_lim_point = 2 * int(features.shape[0] * (epsilon_lower_percentile / 100))
    slope_threshold = np.mean(slope) + whiskerlength * np.std(slope)
    index_knee_onset = np.amin(np.argwhere(slope[low_lim_point:] > slope_threshold))
    epsilon_range[1] = kNN_distance[index_knee_onset]

    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
    ax1.plot(kNN_distance)
    ax1.set_title('upper epsilon limit: {}'.format(epsilon_range[1, 0]))
    ax1.axhline(epsilon_range[0], color='r')
    ax1.axhline(epsilon_range[1], color='r')
    ax1.set_ylabel('{}-NN distance'.format(N_neighbors))
    ax2.plot(slope)
    ax2.set_title('whiskerlength: {} '.format(whiskerlength))
    ax2.set_ylabel('slope of {}-NN distance'.format(N_neighbors))
    ax2.axhline(slope_threshold, color='r')
    ax2.set_xlabel('input data points')

    return kNN_distance, epsilon_range


def plot_envelopes_per_cluster(per_cluster_MSE, envelope_data, events_for_plotting):
    """
    Plot cluster-wise envelope traces for all specified events.

    Parameters
    ----------
    
    per_cluster_MSE : np.array (nClusters)
        pattern of cluster representative
    
    envelope_data : dict 
        dict with all envelope traces for the single clusters, the mean 
        envelope per cluster and the corresponding time lables
        
    events_for_plotting : List[str] (nChannels)
        event names for plotting
    
    """

    per_cluster_traces = envelope_data['per_cluster_traces']
    per_cluster_mean_trace = envelope_data['per_cluster_mean_trace']
    time_points = envelope_data['time_points']

    events = ['Pause', 'GetReady', 'GoCue', '1stHit', '2ndHit', '3rdHit', '4thHit', 'TrialEnd']
    event_indices = [i for i, item in enumerate(events) if item in events_for_plotting]

    fig, axs = plt.subplots(len(per_cluster_traces), len(event_indices),
                            figsize=(3 * len(event_indices), 2 * len(per_cluster_traces)),
                            facecolor='w', edgecolor='k', sharey=True, sharex=True)
    fig.subplots_adjust(hspace=.1, wspace=.05)

    for count, (single_cluster_envelopes, single_cluster_mean) in enumerate(
            zip(per_cluster_traces, per_cluster_mean_trace)):
        event_related_envelopes = np.split(single_cluster_envelopes, len(event_indices), axis=1)
        event_related_mean_envelope = np.split(single_cluster_mean, len(event_indices), axis=0)
        for idx, event_index in enumerate(event_indices):
            single_comps = event_related_envelopes[idx]
            mean = event_related_mean_envelope[idx]
            axs[count, idx].plot(time_points.T, single_comps.T, 'k', linewidth=0.5)
            axs[count, idx].plot(time_points.T, mean.T, 'r', linewidth=1.5)
            if count == 0:
                axs[count, idx].set_title(events[event_index])
            if idx == 0:
                # annotate cluster label
                axs[count, idx].set_ylabel('log-envelope')
                label = 'cluster ' + str(count - 1)
                axs[count, idx].annotate(label, (0, 0.5), xytext=(-45, 0), ha='right', va='center',
                                         size=14, rotation=90, xycoords='axes fraction',
                                         textcoords='offset points')

                # annotate log-MSE per cluster
                label_top = "log(IC-MSE) = %.3f " % np.log10(per_cluster_MSE[count])
                axs[count, idx].annotate(label_top, xy=(0.05, 0.95), xytext=(5, -5), fontsize=12,
                                         xycoords='axes fraction', textcoords='offset points',
                                         bbox=dict(facecolor='white', boxstyle='round', alpha=0.7),
                                         horizontalalignment='left', verticalalignment='top')

            if count == len(per_cluster_traces) - 1:
                axs[count, idx].set_xlabel('time [ms]')


def plot_representative_filter_pattern(representative_patterns, representative_filters,
                                       channel_names, cluster_ids):
    """
    Plot cluster-wise filter/pattern pairs for the cluster representatives.

    Parameters
    ----------
    
    representative_patterns : np.array (nClusters x nChannels)
        pattern of cluster representative
    
    representative_filters : np.array (nClusters x nChannels)
        spatial filters of cluster representative
        
    channel_names : List[str] (nChannels)
        channel names according to 10/20 system
    
    cluster_ids : list
        list of cluster labels 
    
    """

    channel_info = mne.create_info(channel_names, sfreq=1000,
                                   ch_types='eeg', montage='standard_1020')

    for cluster_idx, cluster_id in enumerate(cluster_ids):
        _, (ax1, ax2) = plt.subplots(1, 2, sharex='row', sharey='row', figsize=(3, 1.5))
        A_repr = representative_patterns.iloc[cluster_idx, :].as_matrix()
        W_repr = representative_filters.iloc[cluster_idx, :].as_matrix()
        mne.viz.plot_topomap(W_repr, channel_info, axes=ax1, show=False)
        mne.viz.plot_topomap(A_repr, channel_info, axes=ax2, show=False)

        label = 'cluster {}'.format(cluster_id)

        ax1.annotate(label, (0, 0.5), xytext=(0, 0), ha='right', va='center',
                     size=14, rotation=90, xycoords='axes fraction',
                     textcoords='offset points')

        if cluster_idx == 0:
            ax1.set_title('Filter', fontsize=14)
            ax2.set_title('Pattern', fontsize=14)
