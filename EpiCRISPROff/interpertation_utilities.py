'''
This module contain helper function and utilities for model and data interpertability.
'''
import numpy as np
from features_engineering import  generate_features_and_labels, synthesize_all_mismatch_off_targets
from features_and_model_utilities import get_feature_name

from file_utilities import create_paths, create_folder
import itertools
from scipy.stats import zscore
import pandas as pd
import os
import pickle
from datetime import datetime

# Get the current date and time
current_datetime = datetime.now()

# Extract date and time separately
current_date = current_datetime.date()
current_time = current_datetime.strftime("%H:%M:%S")  # Format time as HH:MM:SS

######## DATA ########
def nucleotides_for_heatmap():
    '''
    Create the row labels (nucleotides) per position for the heatmap.
    Returns:
        row_labels (list): List of row labels.
        x_ticks (list): List of x-ticks.'''
    nucleotides_product = list(itertools.product(*(["ACGT-"] * 2)))
    row_labels = [f"{a}:{b}" for a, b in nucleotides_product]
    x_ticks = list(range(1, 25))
    return row_labels, x_ticks
def get_model(model_path, model_type, sample = 0):
    '''
    Loads the model from the given path
    
    Args:
        model_path (str): path to model/folder of models
        model_type (str): type of the model - deep,ml
        sample (int, optional): number of models to sample.
    Returns:
        models (list): list of models    
    '''
    import tensorflow as tf
    from models import argmax_layer
    models = []
    models_path = create_paths(model_path)
    if sample > 0 and len(models_path) > sample:
        models_path = np.random.choice(models_path, sample, replace=False)
    for model_path in models_path:
        if model_type == "deep":
            model = tf.keras.models.load_model(model_path,custom_objects={'argmax_layer': argmax_layer})
            models.append(model)

        else:
            pass
    return models, models_path
 
def get_data(data_path, features = None):
    '''
    Loads the data from the given path
    Uses generate_features_and_labels function to get x,y,guides, off-targets.
    
    Args:
        data_path (str): path to the data
        only_seq (bool): If True, only the sequence features will be used otherwise split to sequence and epigenetics.
    Returns:
        x,y,guides, off targets
        x: list of arrays- each array is all (gRNA,OTS) pairs.
        y: list of arrays - each array is the labels for the pairs.
        guides: list of guides
        otss: list of off-targets
    '''
    Columns_dict = {
    "TARGET_COLUMN": "target",
    "REALIGNED_COLUMN": "realigned_target",
    "OFFTARGET_COLUMN": "offtarget_sequence",
    "CHROM_COLUMN": "chrom",
    "START_COLUMN": "chromStart",
    "END_COLUMN": "chromEnd",
    "BINARY_LABEL_COLUMN": "Label",
    "REGRESSION_LABEL_COLUMN": "Read_count",
    "MISMATCH_COLUMN": "missmatches",
    "BULGES_COLUMN": "bulges"
}
    Columns_dict['Y_LABEL_COLUMN'] = Columns_dict['BINARY_LABEL_COLUMN']
    only_seq = False if features else True
    x,y,guides,otss = generate_features_and_labels(data_path=data_path,manager=None,
                                              if_bp=False,if_only_seq=only_seq,if_seperate_epi=False,
                                              epigenetic_window_size=0,features_columns=features,
                                              if_data_reproducibility=False,columns_dict=Columns_dict,
                                              sequence_coding_type=2,if_bulges=True,return_otss=True, exclude_ontarget=True)
    return x,y,guides,otss

def filter_data_for_interpertation(x_background, y,  sgrna_otss, 
                        specific_indices = None, number_of_points = 200):
    '''
    Sample a subset of the data for interpertation.
    Args:
        x_background (array): all ENCODED gRNA-OT pairs of a sgRNA.
        y (array): labels of the pairs.
        sgrna_otss (array): all (gRNA,OT) seqeuences.
        specific_indices (list, optional): List of specific indices to sample.
        number_of_points (int, optional): Total number of samples.
            None: Balanced amount of positive and negatives will be returned.
            0: all positives will be returned.
            >0: number of points to sample.
        
    Returns:
        (tuple): x_selected, sgrna_otss, additional_features
        x_selected (array): selected encoded gRNA-OT pairs for a given sgRNA.
        sgrna_otss (array): selected sequences (gRNA,OT) pairs for a given sgRNA.
        additional_features (int): number of additional features.'''
    
    if specific_indices is not None:
        pass #NOTE: ADD SPESIFIC INDICES WITH FEATURE ENGINGERRING FUNCTION
    
    # NOTE: SAMPLE OUT NEGATIVES (NOT BY STARTIFYING - NEED TO COMPLETE)
    sampled_indices = get_sampled_indices(y, number_of_points = number_of_points)
    if isinstance(x_background,list):
        x_selected = [x[sampled_indices] for x in x_background]
    else:
        x_selected = x_background[sampled_indices]
    sgrna_otss = sgrna_otss[sampled_indices]
    return x_selected, sgrna_otss
    
    
def get_sampled_indices(y, number_of_points = None):
    '''
    Sample all positives and randomly sample negatives to complete the gap to number of points
    if number of points is None than balanced amount of positive and negatives will be returned.
    Args:
        y (np.array): labels
        number_of_points (int, optional): Total number of samples.
            None: Balanced amount of positive and negatives will be returned.
            0: all positives will be returned.
            >0: number of points to sample.
    Returns:
        list of indices
    '''
    positive_indexes = np.where(y == 1)[0]
    negative_indexes = np.where(y == 0)[0]
    positive_number = len(positive_indexes)
    if number_of_points is None: # balanced
        print("no number of points given, return balanced positives and negatives")
        negative_indexes = np.random.choice(negative_indexes, positive_number, replace=False)
        return np.concatenate((positive_indexes, negative_indexes))
    elif number_of_points <= positive_number:
        if number_of_points == 0:
            print("number of points is 0, return all positives")
            return positive_indexes
        else:
            positive_indexes = np.random.choice(positive_indexes, number_of_points, replace=False)
            return positive_indexes
        
    negative_number = number_of_points - positive_number # else sample negatives    
    random_negative_indices = np.random.choice(negative_indexes, negative_number, replace=False)
    return np.concatenate((positive_indexes, random_negative_indices))
    

######## SHAP ########
  
def convert_features_names(shap_values, agg_function, seqeunce_length, bits_per_base, additional_features_length = 0):
    '''
    Convert the shap values of all features into group of features
    by aggregating the values of each group of features
    Args:
        shap_values (list of numpy arrays): List of SHAP value arrays.
        agg_function (function): Aggregation function to use.
    Returns:
        list of numpy.ndarray: Aggregated SHAP values.
    '''
    original_values = shap_values.values  # Shape: (num_samples, N)
    groups = [list(range(i * bits_per_base, (i + 1) * bits_per_base)) for i in range(seqeunce_length)]
    feature_names = [f"Base_{i+1}" for i in range(seqeunce_length)]
    total_sequence_length = seqeunce_length * bits_per_base
    if additional_features_length > 0:
        groups.append(list(range(total_sequence_length , total_sequence_length + additional_features_length)))  
        feature_names.append("epigenetics")
    # Aggregate SHAP values by summing grouped features
    grouped_shap_values = np.zeros((original_values.shape[0], len(groups)))
    for i, indices in enumerate(groups):
        grouped_shap_values[:, i] = np.sum(original_values[:, indices], axis=1)
    # Update feature names
    shap_values.values = grouped_shap_values
    shap_values.feature_names = feature_names
    shap_values.data = shap_values.data[:, [g[0] for g in groups]]
    return shap_values

######## Epigenetics ########
def epigenetic_05_vector(features, sg_ot_pair):
    """
    Adds to the sg_ot_pair/s epigenetic vector with 0.5 values and 1/0 values for each feature.
    Given N is the number of features, each feature will get 2 vectors
    where the feature will be set to 1/0 and rest of the features to 0.5.
    In total 2*N vectors will be created for each sg_ot_pair.
    The sg_ot_pair/s will be repeated for each 2*N vectors.

    For Example: 
        sg_ot_pair = [1,0,0,1], features = ['H3K4me3','H3K27ac']
        (H3K4me3) -> [1,0,0,1] + [1,0.5], [0,0.5]. 
        (H3K27ac) -> [1,0,0,1] + [0.5,1], [0.5,0].
    ...
    
    Args:
        features (list): List of epigenetic features.
        sg_ot_pair (np.array): Pair of sgRNA and off-target.
    Returns:
        (nd.array): Input for the model.
        
    """
    number_of_features = len(features)
    one_pair = True if sg_ot_pair.ndim == 1 else False # If only one sgRNA-OT pair
    if one_pair: # If only one sgRNA-OT pair
        sg_ot_pair = sg_ot_pair.reshape(1, len(sg_ot_pair))
    constant_sg_ot = np.repeat(sg_ot_pair, 2*number_of_features, axis=0) # Repeat the same sgRNA-OT pair/pairs
    epi_vector = np.full((2*number_of_features, number_of_features), 0.5) # Create 2 vectors for each feature with 0.5 values
    rows = np.arange(number_of_features) * 2 # Feature rows jumps of 2
    epi_vector[rows, rows // 2] = 0 # even rows - 0
    epi_vector[rows + 1, rows // 2] = 1 # odd rows - 1
    if not one_pair: # reapeat the epignetic vector for All sgRNA-OT pairs
        epi_vector = np.tile(epi_vector, (sg_ot_pair.shape[0], 1)) 
    x_input = [constant_sg_ot, epi_vector] # Create input for model
    return x_input
    model_output = model.predict(x_input) # Get model output
def epigenetic_genome_disterbution_vector(features, sg_ot_pair, epigenetic_disterbution_file):
    """
    Adds to the sg_ot_pair/s epigenetic vector with disterbution values and 1/0 values for each feature.
    Given N is the number of features, each feature will get 2 vectors
    where the feature will be set to 1/0 and rest of the features to the disterbution values.
    In total 2*N vectors will be created for each sg_ot_pair.
    The sg_ot_pair/s will be repeated for each 2*N vectors.
    
    For example: 
        sg_ot_pair = [1,0,0,1], features = ['H3K4me3','H3K27ac'], disterbution: {H3K4me3: 0.12, H3K27ac: 0.3}
        (H3K4me3) -> [1,0,0,1] + [1,0.3], [0,0.3].
        (H3K27ac) -> [1,0,0,1] + [0.12,1], [0.12,0].
    
    Args:
        features (list): List of epigenetic features.
        sg_ot_pair (np.array): Pair of sgRNA and off-target.
        epigenetic_disterbution_file (pd.DataFrame): data frame where columns are 
        epigenetic features and each column has disterbution value
        COLUMNS MUST MATCH THE FEATURES.
    
    Returns:
        (nd.array): Input for the model.
    """
    column_features = epigenetic_disterbution_file.columns.tolist()
    if all([feature in column_features for feature in features]):
        raise ValueError("All features must be in the disterbution file")
    column_features = {col: get_feature_name(col) for col in column_features}
    epigenetic_disterbution_file = epigenetic_disterbution_file.rename(columns=column_features)
    number_of_features = len(features)
    one_pair = True if sg_ot_pair.ndim == 1 else False # If only one sgRNA-OT pair
    if one_pair: # If only one sgRNA-OT pair
        sg_ot_pair = sg_ot_pair.reshape(1, len(sg_ot_pair))
    constant_sg_ot = np.repeat(sg_ot_pair, 2*number_of_features, axis=0) # Repeat the same sgRNA-OT pair/pairs
    epi_vector = np.zeros((2*number_of_features, number_of_features)) # Create 2 vectors for each feature 
    for feature_index, feature in enumerate(features):
        disterbution_value = epigenetic_disterbution_file[feature].values[0] # Get disterbution value
        epi_vector[:, feature_index] = disterbution_value # Fill the feature with the disterbution value
    rows = np.arange(number_of_features) * 2 # Feature rows jumps of 2
    epi_vector[rows, rows // 2] = 0 # even rows - 0
    epi_vector[rows + 1, rows // 2] = 1 # odd rows - 1
    if not one_pair: # reapeat the epignetic vector for All sgRNA-OT pairs
        epi_vector = np.tile(epi_vector, (sg_ot_pair.shape[0], 1)) 
    x_input = np.concatenate([constant_sg_ot, epi_vector],axis=1) # Create input for model
    return x_input

def return_epigentic_disterbution_vector(disterbution_file, features):
    epi_vector = []
    for feature in features:
        if feature in disterbution_file.columns:
            pass
def add_epigenetic_vector_to_offtargets(guides_dict,features,epigenetic_disterbution_file):
    guide_dict_with_epi_genetics = {}
    for guide, mismatch_dict in guides_dict.items():
        mismatch_dict_with_epigenetics = {mismatch_number: epigenetic_genome_disterbution_vector(features,synthesized_off_targets,epigenetic_disterbution_file) 
                                        for mismatch_number,synthesized_off_targets in mismatch_dict.items()}
        guide_dict_with_epi_genetics[guide] =  mismatch_dict_with_epigenetics
    return guide_dict_with_epi_genetics
def create_off_targets_for_guides(guide_list,if_flatten = True, mismatch_limit = 6, sample_data=True):
    """
    Create synthetic off-target for sgRNAs in the guide_list.
    
    Args:
        guide_list (list): List of sgRNAs.
        if_flatten (bool): If True, flatten the off-targets.
        mismatch_limit (int): Maximum number of mismatches.
    Returns:
        dict: Dictionary of sgRNA and their synthetic off-targets.
        {guide: {mismatch_number: np.array[off-targets]}}
    """
    
    guides_dict = {}
    for guide in guide_list:
        guides_dict[guide] = synthesize_all_mismatch_off_targets(sgrna_sequence=guide,mismatch_limit=mismatch_limit,
                                                                 if_range=True,if_flatten=if_flatten,
                                                                 sample_data=sample_data)
    return guides_dict
    

def epi_feature_importance_from_model_output(features, model_output):
    """
    Calculate the epigenetic feature importance from the model outputs.
    The importance is calculated by substracting the prediction of the model where the epigenetic feature is on and off.
    If the Delta is positive the feature is important.
    Args:
        features (list): list of features ['H3K4me3','H3K27ac'...].
        model_output (np.array): model output - should match in number the sgRNA-OT pairs.
    Returns:
        dict: Dictionary of epigenetic feature importance {'feature name': [importance values]}."""
    number_of_features = 2*len(features)

    model_output = model_output.reshape(-1) # Reshape to 1D array
    if len(model_output) % (number_of_features) != 0:
        raise ValueError("Model output length should be a multiple of 2*number of features")
    model_output = model_output.reshape(int(len(model_output)/(number_of_features)),number_of_features) # Every row is all features pairs for one sgRNA-OT
    epi_feature_importance = {feature: model_output[:, 2*i+1] - model_output[:, 2*i] for i, feature in enumerate(features)} # Calculate difference
    return epi_feature_importance

def epigenetic_pertubation_importance(features, sg_ot_pair, model):
    '''
    Function calculate the epigentic feature importance by perturbarting over the epigenetic features.
    It substracts the prediction of the model where the epigenetic feature is on and off.
    If the Delta is positive the feature is important.

    Creates 2^(number_of_features - 1) epigenetic vectors pairs.
    Each pair has the same indexes on except for the given index.
    For example: number_of_features = 3, index = 0 -> create pairs of (0,1,1) - (1,1,1), (0,0,1) - (1,0,1) ...
    So the wanted epigenetic feature is off/on in the pairs.
    Args:
        features (list): List of epigenetic features.
        sg_ot_pair (np.array): Pair of sgRNA and off-target.
        model (tf.keras.Model): Model to interpret.
    Returns:
        dict: Dictionary of epigenetic feature importance.
    '''
    all_combination_epi_vectors = generate_all_bit_combinations(len(features))
    if sg_ot_pair.ndim == 1:
        sg_ot_pair = sg_ot_pair.reshape(1, len(sg_ot_pair))
    constant_x = np.repeat(sg_ot_pair, len(all_combination_epi_vectors), axis=0) # Repeat the same sgRNA-OT pair
    x_input = [constant_x, all_combination_epi_vectors] # Create input for model
    model_output = model.predict(x_input) # Get model output
    bit_values = {tuple(bits): prediction[0] for bits, prediction in zip(all_combination_epi_vectors, model_output)} # Create dict of bit values
    epi_feature_importance = {}
    for feature_index, feature in enumerate(features): # Iterate over features
        
        epi_feature_importance[feature] = subtract_pairs_by_masking(bit_values, feature_index,len(features))
    return epi_feature_importance

def generate_all_bit_combinations(N):
    '''
    Generate all possible bit combinations for N bits.
    Args:
        N (int): Number of bits.
    Returns:
        np.array: All possible bit combinations'''
    return np.array(list(itertools.product([0, 1], repeat=N)))

def subtract_pairs_by_masking(bit_values, i,N):
    '''
    Substract matching values of pairs of vectors where the i-th bit is on and off.
    Args:
        bit_values (dict): Dictionary of vectors and their values.
        i (int): Index of the bit to mask.
    Returns:
        np.array: Differences between matching pairs.
    '''
    bit_combinations = np.array(list(bit_values.keys()))  # Convert dict keys to array
    values = np.array(list(bit_values.values()))  # Convert dict values to array

    # Create masks
    mask_on = bit_combinations[:, i] == 1  # Select where bit i is 1
    mask_off = bit_combinations[:, i] == 0  # Select where bit i is 0

    # bits_on = bit_combinations[mask_on]  # Entries where bit i is 1
    # bits_off = bit_combinations[mask_off]  # Entries where bit i is 0
    
    values_on = values[mask_on]  # Values for bit i = 1
    values_off = values[mask_off]  # Values for bit i = 0
    
    # bits_on_reduced = np.delete(bits_on, i, axis=1)
    # bits_off_reduced = np.delete(bits_off, i, axis=1)

    # # Convert to structured arrays for row-wise matching
    # dtype = [('f{}'.format(j), bits_on_reduced.dtype) for j in range(N-1)]
    # bits_on_struct = bits_on_reduced.view(dtype)
    # bits_off_struct = bits_off_reduced.view(dtype)

    # Find matching pairs
    # _, idx_on, idx_off = np.intersect1d(bits_on_struct, bits_off_struct, return_indices=True)

    # # Compute differences for matched indices
    # differences_ = values_on[idx_on] - values_off[idx_off]
    differences = values_on - values_off  # Calculate differences
    # print(np.equal(differences, differences_).all())  # Check if differences are equal
    
    return differences  

def convert_importance_dicts_to_2d_arrays(epigenetic_importance_arrays, mean_pertubation_importance_list, importance_05_list):
    '''
    Converts the mean pertubation importance and 0.5 importance values into 2D arrays for each feature.
    i.e. the output will be a dictionary with features as keys and 2D arrays as values.
    the 2D arrays will have 2 rows: mean pertubation importance and 0.5 importance values.
    Args:
        epigenetic_importance_arrays (dict): Dictionary of features and their 2D arrays.
        mean_pertubation_importance_list (list): List of dictionaries of mean pertubation importance values.
        importance_05_list (list): List of dictionaries of 0.5 importance values.
    Returns:
        dict: Dictionary of features and their 2D arrays.
    '''
    # Convert the list of dictionaries into a 2D array for each feature
    for feature in epigenetic_importance_arrays.keys():
        # Stack the mean and 05 importance values along axis 0 for each feature
        mean_values = [mean_pertubation_importance[feature] for mean_pertubation_importance in mean_pertubation_importance_list]
        importance_values = [importance_05[feature] for importance_05 in importance_05_list]
        
        # Update the epigenetic correlation arrays
        epigenetic_importance_arrays[feature][0, :] = mean_values
        epigenetic_importance_arrays[feature][1, :] = importance_values
    return epigenetic_importance_arrays

def remove_outliers_z(data, z_threshold=3):
    '''
    Remove outliers from the data using z score threshold.
    Args:
        data (np.array/pd.DataFrame): Data to remove outliers from.
        z_threshold (float, optional): Z score threshold.
    Returns:
        np.array/pd.DataFrame: Data without outliers.
    '''
    amount = len(data)
    filtered =  data[(np.abs(zscore(data)) < z_threshold).all(axis=1)]
    print(f"Removed {amount - len(filtered)} outliers")
    return filtered
    

def remove_outliers_iqr(data, iqr_threshold=1.5):
    '''
    Remove outliers from the data using IQR threshold.
    Args:
        data (np.array/pd.DataFrame): Data to remove outliers from.
        iqr_threshold (float, optional): IQR threshold.
    Returns:
        np.array/pd.DataFrame: Data without outliers.
    '''
    amount = len(data)
    if isinstance(data, np.ndarray):
        q1 = np.quantile(data, 25)
        q3 = np.quantile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered = data[(data >= lower_bound) & (data <= upper_bound)]

    elif isinstance(data, pd.DataFrame):
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        filtered = data[~((data < (q1 - iqr_threshold * iqr)) | (data > (q3 + iqr_threshold * iqr))).any(axis=1)]
    else:
        raise ValueError("Data should be a numpy array or a pandas dataframe")
    print(f"Removed {amount - len(filtered)} outliers")
    return filtered

def keep_data_percentile(data, lower_bound = 0.05, upper_bound = 0.95):
    '''
    Keep data within the given percentile.
    Args:
        data (np.array/pd.DataFrame): Data to keep.
        lower_bound (float, optional): Lower percentile.
        upper_bound (float, optional): Upper percentile.
    Returns:
        np.array/pd.DataFrame: Data within the given percentile.
    '''
    amount = len(data)
    if isinstance(data, np.ndarray):
        lower = np.quantile(data, lower_bound)
        upper = np.quantile(data, upper_bound)
        filtered = data[(data >= lower) & (data <= upper)]
    elif isinstance(data, pd.DataFrame):
        lower = data.quantile(lower_bound)
        upper = data.quantile(upper_bound)
        filtered = data[(data >= lower) & (data <= upper)]
    else:
        raise ValueError("Data should be a numpy array or a pandas dataframe")
    print(f"Removed {amount - len(filtered)} outliers")
    return filtered

def get_model_scores_for_features(model_path, guide_dict_with_epi_genetics, save_model_scores = False, output_path = None,
                                  model_suffix=None):
    """
    Use the run_models class to get the model scores for each guide and mismatch number.
    
    Args:
        model_path (str): Path to the model/s.
        guide_dict_with_epi_genetics (dict): Dictionary of guides and their off-targets - 
            {guide: {mismatch_number: off-targets}}
        save_model_scores (bool): If True, save the model scores in a npy file for each guide and mismatch number - 
            {guide}_{mismatch_number}_scores.npy.
        output_path (str): Path to save the model scores.
        model_suffix (str): Suffix to add to the model name.
    Returns:
        dict: Dictionary of model scores for each guide and mismatch number.
            {guide: {mismatch_number: model_scores}}
        """
    # Run the ensemble on the data
    from run_models import run_models
    models = create_paths(model_path)
    runner = run_models()
    runner.setup_runner(ml_task='classification',cross_val=3,model_num=6,features_method=2,cw=2,encoding_type=2,if_bulges=True)
    model_outputs = {}
    if save_model_scores:
        save_folder_name = os.path.join("Model_scores","Raw_scores")
        model_temp_path = create_folder(output_path,save_folder_name)
        
    for guide, mismatch_dict in guide_dict_with_epi_genetics.items():
        guide_outputs = {}
        for mismatch_num, off_targets in mismatch_dict.items():
            y_scores,test,indexes = runner.test_ensmbel(models,x_features=off_targets,tested_guide_list=None)
            avg_scores = np.mean(y_scores,axis=0)
            if save_model_scores:
                guide_mis_path = f'{guide}_{mismatch_num}_{model_suffix}_scores.npy' if model_suffix else f'{guide}_{mismatch_num}_scores.npy'
                np.save(f'{model_temp_path}/{guide_mis_path}',avg_scores)
            guide_outputs[mismatch_num] = avg_scores
        model_outputs[guide] = guide_outputs 
    return model_outputs

def save_importance_values(model_outputs, output_path):
    save_folder_name = os.path.join("Model_scores","Importance_scores")
    output_path = create_folder(output_path,save_folder_name)
    for guide, mismatch_dict in model_outputs.items():
        
        for mismatch_num, importance_dict in mismatch_dict.items():
            temp_output = os.path.join(output_path,f'{guide}_{mismatch_num}.pkl')
            with open(temp_output, 'wb') as f:
                pickle.dump(importance_dict, f)
def load_importance_values(importance_path, guide_list, mismatch_limit):
    importance_dict = {}
    error_file = f'{importance_path}/error_file_{current_date}:{current_time}.txt'
    for guide in guide_list:
        importance_dict[guide] = {}
        for mismatch_num in range(1,mismatch_limit + 1):
            temp_output = os.path.join(importance_path,f'{guide}_{mismatch_num}.pkl')
            try:
                with open(temp_output, 'rb') as f:
                    importance_dict[guide][mismatch_num] = pickle.load(f)
            except Exception as e:
                print(f"Error: {e}")
                with open(error_file, 'a') as error_f:
                    error_f.write(f"Error: {e} - {temp_output}\n")
    return importance_dict





 