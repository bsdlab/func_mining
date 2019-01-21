import numpy as np
import pandas as pd
import sklearn.preprocessing
import sklearn.decomposition
import matplotlib.pyplot as plt
import io

def events_for_clustering(events_for_clustering,time_segments):
    """
    Select events and time samples for the clustering step.

    Parameters
    ----------
    events_for_clustering: list (str)
        contains all events that will be used for the clustering
    time_segments: np.array 
        should contains numbers from 1-18. Each time segment corresponds
        to an interval where the envelope was subsampled.
            
    Returns
    -------
    clustering_feature_names: tuple
        strings specifying features in component_data that
        will be selected for the clustering
        
    """
    feature_name = 'envelope_'
    tmp = [feature_name + event for event in events_for_clustering]
    clustering_feature_names = []

    for feature_name_stem in tmp:
        for i in time_segments:
            feature_name = feature_name_stem +'_'+ str(i)
            clustering_feature_names.append(feature_name)
    clustering_feature_names = tuple(clustering_feature_names)
    
    return clustering_feature_names

def calculate_feature_matrix_from_dataframe(dataframe, feature_column_names, scaler='mean-var', allow_nans=False):
    """
    Selects features for the clustering from the input dataframe.

    Parameters
    ----------
    dataframe: pandas dataframe
        contains all components and their parameters, features, etc.
    feature_column_names: list
        contains feature names that are used for the clustering
    scaler: string, optional
        scikit-learn specific scaler for data standardization 
    allows_nans: bool, optional
        flag if NaN entries are allowed
        
    Returns
    -------
    features_scaled: nd.array (N_data x D_input)
        scaled features
    feature_df: dataframe
        corresponding data frame with only features used for clustering
    feature_scaler: scikit-learn instance
        
    """
    if scaler == 'mean-var':
        feature_scaler = sklearn.preprocessing.StandardScaler()
    elif scaler == 'median-iqr':
        feature_scaler = sklearn.preprocessing.RobustScaler()
    else:
        raise ValueError("unknown scaling method")

    # convert to numeric arguments
    if allow_nans:
        error_strategy = 'coerce'
    else:
        error_strategy = 'raise'  # exception on unparseable value
    feature_df = dataframe.loc[:, feature_column_names].apply(pd.to_numeric, errors=error_strategy)

    feature_matrix = feature_df.values
    # check for first column that we have identical column order in matrix as in input colnames
    assert np.all(feature_matrix[:, 0] == feature_df[feature_column_names[0]])

    features_scaled = feature_scaler.fit_transform(feature_matrix)
    return features_scaled, feature_df, feature_scaler


def apply_dimensionality_reduction(features,method='PCA',N_components=None,
                                   explained_variance=0.9,whiten=False,kernel_function='rbf'):
    """
    Dimensionalty reduction of the given envelope features by either PCA or 
    kernel PCA.

    Parameters
    ----------
    features: nd.array
        input features         
    method: string  
        dimensionality reduction method, default: PCA 
    N_components: int
        N_components that are used to reduce the dimensionality of the input features
    explained_variance: float, optional
        only for PCA: if N_components=None, the explained variance can be used 
        to determine a number of components explained a percentage of the variance
    whiten: boolean, optional 
        flag to apply data whitening before dim. reduction 
    kernel_function: string, optional for 'kernelPCA'
        specify kernel function 
        
        
    Returns
    -------
    features_red: nd.array (N_datax N_components)
        features projected to subspace
    
    dr_instance: sklearn instance of specified method

    N_components: int
        number of components used

    """
    
    if method == 'PCA':
        dr_instance = sklearn.decomposition.PCA(whiten=whiten)
        dr_instance.fit(features)
    
        if N_components is None:
            #select no. of PCA components according to explained variance
            pca_expl_var_cumsum = np.cumsum(dr_instance.explained_variance_ratio_)

            _, number_pca_components = _find_nearest(pca_expl_var_cumsum,explained_variance)
            print("Selected", number_pca_components,"PCA components")   

    elif method == 'kernelPCA':
        dr_instance = sklearn.decomposition.KernelPCA(kernel=kernel_function)
        dr_instance.fit(features)
           
    else:
        print('method not correclty specified')
    
    # apply dim reduction to input features
    features_red = dr_instance.transform(features)[:, 0:N_components]
    
    features_all = dr_instance.transform(features)
    explained_variance = np.var(features_all, axis=0)
    explained_variance_ratio = explained_variance/np.sum(explained_variance)
    explained_var = np.cumsum(explained_variance_ratio)
    
    return features_red, dr_instance, N_components, explained_var
    
def _find_nearest(a, a0):
    """Element in nd array `a` closest to the scalar value `a0`"""
    idx = np.abs(a - a0).argmin()
    return a[idx], idx


