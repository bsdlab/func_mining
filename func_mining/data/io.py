import datetime
import functools
import numpy as np
import os
import os.path
import pathlib
import pandas as pd
import pickle
import scipy.io
import sklearn.feature_extraction
import csv

def load_csv_format_to_dataframes(filename):
    """Load component data from bsv file in a dictionary of dataframes.
    The dictionary consists of four keys: 'features', 'metrics', 'parameters'
    and 'meta'.

    Input
    ----------
    filename : str
        full path to bsv-file
    
    Returns
    -------
    dfs_by_category : dict of data frames
        for each information category that characterizes a component, 
        a dataframe is provided
    """
    
    col_names = ['resultId', 'category', 'key', 'value']
    components = pd.read_csv(filename, delimiter='|', header=None)
    if len(components.columns) == len(col_names):
        components = components.rename(columns=col_names, inplace=True)
    elif len(components.columns) == len(col_names) + 2:
        # empty columns due to separators at beginning and end of line
        temp_colnames = ['dummy1'] + col_names + ['dummy2']
        col_rename_map = dict(zip(components.columns, temp_colnames))
        components.rename(columns=col_rename_map,
                          inplace=True)
        components = components[col_names]
    else:
        raise ValueError('unknown data format: expected columns for {}, but got {}'.format(col_names, components.columns))
    
    categories = components['category'].unique().tolist()
    assert 'features' in categories

    dfs_by_category = {}
    for category in categories:
        df_cur_cat = components[components['category'] == category]
        df_cur_cat_wide = df_cur_cat.pivot(index='resultId', columns='key', values='value')
        dfs_by_category[category] = df_cur_cat_wide

    # remove prefixes that are invalid on system
    if 'meta' in categories:
        component_base_dir = os.path.dirname(filename)
        dfs_by_category['meta']['record_dir'] = dfs_by_category['meta']['record_dir'].apply(
            lambda record_dir: _extract_valid_subdir_path(path_with_valid_suffix=record_dir,
                                                          base_directory=component_base_dir))

    return dfs_by_category

def _extract_valid_subdir_path(base_directory, path_with_valid_suffix):
    base_path = pathlib.Path(base_directory)
    orig_subdir_parts = list(pathlib.PurePath(path_with_valid_suffix).parts)
    for first_part_idx in list(range(len(orig_subdir_parts)))[::-1]:
        subpath = pathlib.PurePath(*orig_subdir_parts[first_part_idx:])
        if (base_path / subpath).exists():
            return str(subpath)

    raise ValueError('could not find valid subdirectory under {} with suffix from {}'.format(
        base_directory, path_with_valid_suffix))
        
def load_filter_pattern(component_data):
    """Manually load spatial filters/patterns for the cluster representatives. 

    Input
    ----------
    component_data : dict
       dict of dataframes with all information about the components
    
    Returns
    -------
    patterns : data frame   
        spatial patterns of all configs
    filters : data frame
        spatial filters of all configs

    """
    pattern_colnames = [col for col in component_data['features'].columns 
                    if col.startswith('pattern_weight_')
                    and not component_data['features'][col].hasnans]
    pattern_channels = [channel_colname.split('_')[2] for channel_colname in pattern_colnames]
    patterns = component_data['features'].loc[:, pattern_colnames].astype(float)
    
    filter_colnames = [col for col in component_data['features'].columns 
                   if col.startswith('filter_weight_')
                   and not component_data['features'][col].hasnans]
    filter_channels = [channel_colname.split('_')[2] for channel_colname in filter_colnames]
    filters = component_data['features'].loc[:, filter_colnames].astype(float)
    
    return patterns, filters, filter_channels

