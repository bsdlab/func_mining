import numpy as np
import sklearn


def index_lists_to_value_lists(index_list, base_list):
    return [[base_list[idx] for idx in idx_list] for idx_list in index_list]


def normalize_patterns(patterns):
    """Normalize patterns by squaring each entry and dividing by the norm

    Parameters
    ----------
    patterns : np.array (nPatterns x nChannels)
        matrix with one pattern per row

    Returns
    -------
    out : np.array (nPatterns x nChannels)
        matrix with normalized patterns (positive entries, norm of each row 1)
    """
    if patterns.ndim == 1:  # single pattern
        patterns_norm = np.reshape(patterns, newshape=(1, -1))
    elif patterns.ndim == 2:
        patterns_norm = patterns
    else:
        raise ValueError('pattern array must be 1D or 2D')
    patterns_norm = np.square(patterns_norm)
    patterns_norm = patterns_norm / np.reshape(np.linalg.norm(patterns_norm, axis=1),
                                               newshape=(-1, 1))
    if patterns.ndim == 1:
        patterns_norm = np.squeeze(patterns_norm)
    return patterns_norm


def calculate_filter_angle(filter1, filter2):
    """Calculate angle between filters according to Krauledat et al. (2007)
        
    Parameters
    ----------
    filter1 : np.array (nChannels)
        matrix containing a spatial filter

    Returns
    -------
    out : np.array (filter1 x filter2)
        matrix with filter angles
    """

    X, Y = sklearn.metrics.pairwise.check_pairwise_arrays(filter1, filter2)
    X_normalized = sklearn.preprocessing.normalize(X, copy=True)
    if X is Y:
        Y_normalized = X_normalized
    else:
        Y_normalized = sklearn.preprocessing.normalize(Y, copy=True)

    K = np.dot(X_normalized, Y_normalized.T)
    # rarely: > 1 due to rounding issues?
    if K > 1:
        K = 1

    angle = np.arccos(K)
    assert not np.isnan(angle).any(), "invalid angle - dot product {} ".format(K[np.argwhere(np.isnan(angle))])
    assert not np.isnan(K).any(), "illegal angle between filters"

    # find entries that are larger than 0.5*pi 
    for n, value in np.ndenumerate(angle):
        if value > (0.5 * np.pi):
            angle[n] = np.pi - angle[n]

    return angle


def spatial_filter_angle(filtersA, filtersB=None):
    """Calculates the filter angle between all spatial filters in filtersA and filtersB

    If filtersB is not supplied, pairwise distances between all rows in filtersA are computed.

    Parameters
    ----------
    filtersA : np.array
        filters (nFiltersA x nChannels)
    filtersB : np.array, optional
        filters (nFiltersB x nChannels); default is None

    Returns
    -------
    out : np.array (nPatternsA x nPatternsB)
        pairwise cosine distances for rows from A and B (B=A if filtersB=None)
    """
    if filtersB is None:
        filtersB = filtersA

    filtersA_normalized = normalize_patterns(filtersA)
    filtersB_normalized = normalize_patterns(filtersB)

    # calculate EMD between patterns, incorporating channel distances
    theta_distances = np.zeros((filtersA.shape[0], filtersB.shape[0]))
    theta_distances[:] = np.nan

    for i in range(filtersA_normalized.shape[0]):
        for j in range(filtersB_normalized.shape[0]):
            theta_dist = calculate_filter_angle(filtersA_normalized[i, :].reshape(1, -1),
                                                filtersB_normalized[j, :].reshape(1, -1))
            assert not np.isnan(theta_dist), "got illegal distance for pattern {}, {}".format(
                filtersA_normalized[i, :], filtersB_normalized[j, :])
            theta_distances[i, j] = theta_dist

    return theta_distances
