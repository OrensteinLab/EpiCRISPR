'''
This module is used to define parameters needed for features and model training.
'''
import os
def get_full_enconding_parameters_(if_bulges = None, if_downstream_in_sequence = False, down_stream_nuc = 9):
    '''
    Gets the length of the sequence and the number of nucleotides in the sequence for the full enconding.
    '''
    if if_bulges is None:
        raise ValueError("if_bulges  must be provided.")
    if if_bulges:
        seq_len = 24
        nuc_num = 5 #ATCG-
    else:
        seq_len = 23
        nuc_num = 4 #ATCG
    if if_downstream_in_sequence:
        seq_len += down_stream_nuc
    return (seq_len, nuc_num) 

def get_PiCRISPR_enconding_parameters(if_bulges = None):
    '''
    Gets the length of the sequence and the number of nucleotides in the sequence for the PiCRISPR enconding.
    '''
    if if_bulges is None:
        raise ValueError("if_bulges  must be provided.")
    if if_bulges:
        raise ValueError("PiCRISPR encoding does not support bulges")
    else:
        seq_len = 23
        nuc_num = 6 #ATCG
    return (seq_len, nuc_num)

def get_encoding_parameters(coding_type, if_bulges, if_downstream_in_sequence = False, down_stream_nuc = 9):
    '''
    Gets the length of the sequence and the number of nucleotides in the sequence for the given encoding type.
    '''
    if coding_type == 1:
        return get_PiCRISPR_enconding_parameters(if_bulges)
    elif coding_type == 2:
        return get_full_enconding_parameters_(if_bulges,if_downstream_in_sequence = if_downstream_in_sequence,
                                               down_stream_nuc = down_stream_nuc)
    else:
        raise ValueError("Invalid coding type")
    
#################
'''
Epigenetic feature utilities
'''
           

def parse_feature_column_dict(feature_column_dict, only_epigenetics = False):
    '''
    This function adds all the features to a list called All_features.
    If there is only one list of features it just return this dict.
    If there are subsets of features from the given group it skips as they inserted to the dict already.
    If only_epigenetics is True it only returns the epigenetic features.
    '''
    all_features = []
    if len(feature_column_dict) == 1:
        return feature_column_dict
    filtered_dict = feature_column_dict.copy()
    for keys,features in feature_column_dict.items():
        if only_epigenetics:
            if "epigenetic" not in keys.lower():
                filtered_dict.pop(keys)
                continue
        if "Subset" in keys:
            continue
        all_features.extend(features)
    if not only_epigenetics: # If only epigenetics we do not need to add all features
        filtered_dict["All_features"] = all_features
    return filtered_dict




def get_features_columns_args_ensembles(runner = None,file_manager = None, t_guides = None, 
                              model_base_path = None, ml_results_base_path = None,
                                n_models = None, n_ensmbels = None, features_dict = None, multi_process= False):
    """
    Creates a list of arguments for training/testing the models with features.

    ARGS:
        runner (Runner object): Runner object
        file_manager (FileManager object): FileManager object
        t_guides (list): List of guides for training/testing
        model_base_path (str): Path to save/get the models
        ml_results_base_path (str): Path to save the results
        n_models (int): Number of models to train
        n_ensmbels (int): Number of ensembles to train
        features_dict (dict): Dictionary of features - {group: [features]}
        multi_process (bool): If to run the models in parallel
    
    Returns:
        list (list) of tuples: [(group, [feature],runner, file_manager,t_guides,model_base_path,ml_results_base_path, n_models, n_ensmbels,multi_process)] 
        List of arguments for training/testing the models with features.
        
  
    """
    arg_list = []
    for group, features in features_dict.items():
        if len(features) > 1: # More then 1 feature in the group run all togther.
            arg_list.append((group, features,runner, file_manager,t_guides,model_base_path,ml_results_base_path, n_models, n_ensmbels, multi_process))
        if "Subset" in group or "All" in group:
            continue
        for feature in features:
            arg_list.append((group, [feature],runner, file_manager,t_guides,model_base_path,ml_results_base_path, n_models, n_ensmbels,multi_process))
    return arg_list    


    
def split_epigenetic_features_into_groups(features_columns):
    '''Give a list of features combine the features into groups based on
      their endings - binary, score, enrichment, etc.
      
      Returns a dictionary with the groups as keys and the features as values.'''

    # Create a dictionary to store groups based on endings
    groups = {}
    # Group strings based on their endings
    for feature in features_columns:
        ending = feature.split("_")[-1]  # last part after _ "can be score, enrichment, etc.."
        groups.setdefault(ending, []).append(feature)
    return groups  

def get_feature_name(feature_column):
    '''
    This function splits the feature column by _ and returns the name of the feature
    If the feature is more then one word it will return the first word.
    featurename_type_.... -> featurename'''
    feature_column_splited = feature_column.split("_")
    if len(feature_column_splited) > 1 :
        return feature_column_splited[0]
    else:
        return feature_column
def get_features_string(feature_list, subsets = False, group_ending = None):
    '''
    This function returns a string describing the features in the list.
    If there is more than one feature, it will return a string describing all the features.
    '''
    if len(feature_list) > 1:
        if subsets:
            return "_".join(get_feature_name(feature) for feature in feature_list)
        elif group_ending is None: # subsets False
            raise ValueError("More than one feature, no subset and no group ending given")
        else: # group ending given
            return f"All-{group_ending}"
    else:
        return get_feature_name(feature_list[0])

def get_feature_column_suffix(group,feature):
    '''
    This function returns a path with the corrected suffix for the given group and feature.
    '''
    subsets= False
    if "Subset" in group:
        subsets = True
        group = group.split("-")[1]
    feature_name = get_features_string(feature, subsets, group.split("_")[-1])
    return os.path.join(group,feature_name)

################## TASK UTILITIES ##################
def set_task_column(task, args):
    '''
    This function sets the task column for the given task.
    '''
    if task == "Binary":
        args["task_column"] = "binary"
    elif task == "Regression":
        args["task_column"] = "regression"
    elif task == "Multi":
        args["task_column"] = "multi"
    else:
        raise ValueError("Invalid task")
    return args

### Transformations ###
def transform_labels(y_vals, transform_type):
    '''Transform the y values based on the given transformation type.
    The y_vals can be 1d np array or list of 1d np arrays.
    Returns the transformed y values.'''
    transform_type = transform_type.lower()
    if transform_type == "log":
        return log_transformation(y_vals)
    elif transform_type == "minmax":
        return minmax_transformation(y_vals)
    else:
        raise ValueError("Invalid transformation type.")
def log_transformation(y_vals):
    import numpy as np
    '''Conduct log transformation on the y values.
    The y_vals can be 1d np array or list of 1d np arrays.
    Returns the transformed y values.'''
    if isinstance(y_vals, list):
        transformed = [np.log(y_val + 1) for y_val in y_vals]
        for index,array in enumerate(transformed):
            if array.size != y_vals[index].size:
                raise ValueError("The size of the log transformed array does not match the original array.")
        return transformed
    else:
        transformed = np.log(y_vals + 1)
        if len(transformed) != len(y_vals):
            raise ValueError("The size of the log transformed array does not match the original array.")
        return transformed
def minmax_transformation(y_vals):
    normalized_data = (y_vals - y_vals.min()) / (y_vals.max() - y_vals.min())
    return normalized_data


