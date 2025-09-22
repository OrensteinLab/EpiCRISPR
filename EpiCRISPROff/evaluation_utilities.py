from collections import defaultdict
import numpy as np
import pandas as pd
import pickle
from Figures.ROC_PR_figs import plot_bar_plot, key_renaming
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve, mean_squared_error
from scipy.stats import pearsonr, spearmanr,wilcoxon
import os
from file_utilities import create_paths, create_folder

######## Generall evaluations ##########

def evaluate_model(y_test, y_scores, task = None):
    """
    Evaluates the model given its task.

    For classification: 
        auroc, auprc, n_rank, last_fn_index, last_fn_ratio.
    For regression:
        pearson_r, spearman_r, mse, pearson_p, spearman_p.
    Args:
        y_test (array-like): The actual labels.
        y_scores (array-like): The predicted scores.
        task (str): The task of the model. Options: "classification", "regression".
    Returns:
        metrics (dict): The evaluation metrics.
    """
    if task.lower() == "classification":
        return evaluate_classification(y_test, y_scores)
    elif task.lower() == "reg_classification":
        y_test = (y_test > 0).astype(int) # transform to binary labels
        return evaluate_classification(y_test, y_scores)
    elif task.lower() == "regression" or task.lower() == "t_regression":
        return evalaute_regression(y_test, y_scores)
    else:
        raise RuntimeError(f"Task {task} is not supported")
    

def evaluate_classification( y_test, y_pos_scores_probs, return_rates = False):
    '''
    This function evaluates classification task.
    Given, Args:
    1. y_test - (list/ndarray) the actual labels.
    2. y_pos_scores_probs - (list/ndarray) the predicted scores.
    3. return_rates - (bool) if to return the rates.
    Calculate the fpr,tpr,roc_tresholds, auroc, percesion, recall, auprc, n_rank, last_fn_index, last_fn_ratio.
    -----------
    Returns: by defualt tuple of auroc, auprc, n_rank, last_fn_index, last_fn_ratio.
    if return_rates: (defualt results, dict{fpr,tpr,percesion,recall})
    '''
    fpr, tpr, roc_tresholds = roc_curve(y_test, y_pos_scores_probs)
    auroc = auc(fpr, tpr)
    percesion, recall, tresholds = precision_recall_curve(y_test, y_pos_scores_probs)
    auprc = average_precision_score(y_test, y_pos_scores_probs)
    n_rank = get_auc_by_tpr(get_tpr_by_n_expriments(y_pos_scores_probs,y_test,1000,tpr))[0]
    last_fn_index = get_predictions_needed_to_1_tpr(tpr, roc_tresholds, y_pos_scores_probs)
    last_fn_ratio = get_last_fn_ratio(labels=y_test,predictions=None,tpr=None,last_index=last_fn_index)
    metrics_tuple = (auroc,auprc,n_rank,last_fn_index,last_fn_ratio)
    if return_rates:
        rate_dict = {"fpr":fpr, "tpr":tpr, "percesion":percesion, "recall":recall}
        return (metrics_tuple, rate_dict)
    return metrics_tuple

def evalaute_regression(y_test, y_scores):
    '''This function evaluate the regression model by calculating the pearson and spearman correlations, it also reports the MSE.
    The evaluation is between all data points, and between only the positive OTSs with label > 0.

    Args:
    1. y_test - the actual labels.
    2. y_scores - the predicted scores.
    ------------   
    Returns: tuple of pearson_r, spearman_r, mse, pearson_p, spearman_p.
    '''
    p_r, p_p = pearsonr(y_test, y_scores)
    s_r, s_p = spearmanr(y_test, y_scores)
    mse= mean_squared_error(y_test , y_scores)
    
    return p_r, s_r, mse, p_p, s_p 


### Metrics evaluations: AUC, AUPRC, N-rank, 

def get_tpr_by_n_expriments(predicted_vals,y_test,n, tpr = None):
    '''This function gets the true positive rate for n expriemnets by calculating:
for each 1 <= n' <= n prediction values, what the % of positive predcition out of the the whole TP amount.
for example: '''
    if not tpr is None:
        if len(tpr) >= n:
            return tpr[:n]
        else:
            print("there are more true positives than expriments return the whole tpr")
            return tpr

    # valid that test amount is more then n
    if n > len(y_test):
        print(f"n expriments: {n} is bigger then data points amount: {len(y_test)}, n set to data points")
        n = len(y_test)
    
    tp_amount = np.count_nonzero(y_test) # get tp amount
    if predicted_vals.ndim > 1:
        predicted_vals = predicted_vals.ravel()
    sorted_indices = np.argsort(predicted_vals)[::-1] # Get the indices that would sort the prediction values array in descending order    
    tp_amount_by_prediction = 0 # set tp amount by prediction
    
    tpr_array = np.zeros(n)
    for i in range(n):
        # Accumulate true positives
        tp_amount_by_prediction += y_test[sorted_indices[i]]
        # Calculate TPR
        tpr_array[i] = tp_amount_by_prediction / tp_amount
        # If TPR reaches 1, fill the remaining array with 1s and break
        if tp_amount_by_prediction == tp_amount:
            tpr_array[i:] = 1
            break

    return tpr_array
       


def get_predictions_needed_to_1_tpr(tpr_arr, tresholds = None, predictions = None, return_1_tpr = False):
    """
    Get the index of the last occurrence of the true positive.
    This is equal asking when the tpr =1.
    
    Parameters:
    - tpr_arr (array-like): Array of True Positive Rates (TPR).
    - tresholds (array-like): Array of tresholds.
    - predictions (array-like): Array of predictions.
    - return_1_tpr (bool): If True, return the index of the first TPR = 1.
    Returns:
    - The amount of predictions needed to get tpr = 1.
    if return_1_tpr is True, return the index of the first TPR = 1.
                             
    """
    if len(tpr_arr) == 0:
        raise ValueError("TPR array is empty.")
    if tresholds is None or predictions is None:
        raise ValueError("Predictions or tresholds are missing.")
    tpr_1_index = np.where(tpr_arr == 1)[0][0]
    if return_1_tpr:
        return len(predictions[predictions >= tresholds[tpr_1_index]]), tpr_1_index
    return len(predictions[predictions >= tresholds[tpr_1_index]])
def get_last_fn_ratio(predictions, labels=None, tpr = None, last_index = None, tresholds = None):
    """
    Calculate the ratio of the last false negative index to the total number of labels,
    adjusted by the number of positive labels.
    
    Parameters:
    - predictions (array-like): The predicted binary labels.
    - labels (array-like): The true binary labels (ground truth).
    
    Returns:
    - last_fn_ratio (float): Adjusted ratio of the last false negative index.
    """
    if labels is None:
        raise ValueError("Labels are missing.")
    active_labels = np.count_nonzero(labels)
    total_labels = len(labels)
    if last_index is not None:
        return (last_index - active_labels) / total_labels
    if tpr is None:
        fpr,tpr,_ = roc_curve(labels, predictions)
    last_fn_index = get_predictions_needed_to_1_tpr(tpr, tresholds, predictions)
    return (last_fn_index - active_labels) / total_labels


def get_auc_by_tpr(tpr_arr):
    """
    Calculate the Area Under the Curve (AUC) for the given TPR array.

    Parameters:
    - tpr_arr (array-like): Array of TPR values (y-axis of the curve).

    Returns:
    - calculated_auc (float): AUC value.
    - amount_of_points (int): Number of points in the TPR array.
    """
    amount_of_points = len(tpr_arr)
    x_values = np.arange(1, amount_of_points + 1) / amount_of_points
    calculated_auc = auc(x_values, tpr_arr)

    return calculated_auc, amount_of_points

######## K_cross #########
def convert_k_cross_dict(feature_dict, error_file):
    """
    Given a k_cross feature_dict from the shape:
        {Feature: {Partition: (y_scores, y_test, indexes)}}
    Convert it to the shape:
        {Partition: {Feature: (y_scores, y_test, indexes)}}
    Args:
        feature_dict (dict): Dictionary containing the features data.
        error_file (str): Path to the error file to log issues.
    Returns:
        dict: Converted dictionary with the structure {partition: {feature: (y_scores, y_test, indexes)}}
    """
    feature_partitions = {feature: list(partitions.keys()) for feature, partitions in feature_dict.items()}
    max_feature, max_partitions = max(
    ((feature, len(partitions)) for feature, partitions in feature_partitions.items()),
    key=lambda x: x[1]
    )
    max_partition_list = feature_partitions[max_feature]
    print(f"Feature: {max_feature} has maximum partitions of {max_partitions} with the partitions:\n{max_partition_list}")
    # Check if all features have the same partitions
    for feature, partitions in feature_partitions.items():
        difference = set(max_partition_list).difference(partitions)
        if len(difference) > 0:
            print(f"Feature {feature} missing the following partitions: {difference}.")
            with open(error_file, "a") as f:
                f.write(f"Feature {feature} missing the following partitions: {difference}.\n")
    # convert dict
    converted_dict = defaultdict(dict)
    for feature, partitions in feature_dict.items():
        for partition, (y_scores, y_test, indexes) in partitions.items():
            converted_dict[partition][feature] = (y_scores, y_test, indexes)
    return converted_dict


def get_k_groups_results(k_results_dictionary, task, k_group_columns):
    """
    This function calculates the evaluations metrics for each partition.
    If task is regression evalute model by regression metrics - pearson,spearman,mse
    If task is classification evalute model by classification metrics - auroc,auprc,n-rank,last-tp
    
    Args:
        k_results_dictionary (dict): dictionary with the results for each group. 
        {group: (score,test,indexes)}

    Returns:
        results_data_frame (pd.DataFrame): data frame with the evaluations for each group.        
    """
    results_data_frame = pd.DataFrame(columns = k_group_columns)
    for index,(group, results) in enumerate(k_results_dictionary.items()):
        predictions, labels, _ = results
        results = evaluate_model(labels, predictions,task)
        results_data_frame.loc[index] = [group, *results]
    return results_data_frame

def averaged_k_cross_results(feature_dict, k_group_columns, error_file):
    """
    Given a feature dict:{feature_name: results dataframe} summarize the results of all the features by
    calculating the mean and std for each column in the k_group_columns.

    Args:
        feature_dict (dict): Dictionary containing the features data.
        k_group_columns (list): List of columns to summarize.
        error_file (str): Path to the error file to log issues.
    Returns:
        pd.DataFrame: Data frame with the summarized results.
    """
    columns = [f'{col}_mean' for col in k_group_columns] + [f'{col}_std' for col in k_group_columns] 
    summerized_data_frame = pd.DataFrame(index=feature_dict.keys(), columns=columns)
    
    for feature, results in feature_dict.items():  # Open each DataFrame once
        for col in k_group_columns:
            values = results[col].values
            mean = np.mean(values)
            std = np.std(values)
            summerized_data_frame.loc[feature, f'{col}_mean'] = mean
            summerized_data_frame.loc[feature, f'{col}_std'] = std
    return summerized_data_frame

def compare_by_ratio(feature_dict, feature_to_compare_to, k_group_columns):
    """
    Given a feature dict:{feature_name: results dataframe} and a feature to compare to,
    calculate the ratio of each metric in the k_group_columns compared to the feature_to_compare_to.

    Args:
        feature_dict (dict): Dictionary containing the features data.
        feature_to_compare_to (str): Name of the feature to compare to.
        k_group_columns (list): List of columns to summarize.
    Returns:
        pd.DataFrame: Data frame with the ratios.
    """
    ratios = pd.DataFrame(columns = k_group_columns, index = feature_dict.keys())
    compare_to_data = feature_dict[feature_to_compare_to]
    number_of_partitions = len(compare_to_data)
    ones_array = np.ones(number_of_partitions,dtype=np.int8)
    ratios.loc[feature_to_compare_to] = {col: ones_array for col in k_group_columns}
    for feature, results in feature_dict.items():
        if feature == feature_to_compare_to:
            continue
        for col in k_group_columns:
            try:
                # Calculate the ratio of each entery and than the mean
                ratio = results[col].values / compare_to_data[col].values
                ratios.loc[feature, col] = ratio
            except ValueError as e:
                print(f"Error calculating ratio for {feature} and {feature_to_compare_to} on column {col}: {e}")
    return ratios

def get_p_val_k_cross(feature_dict, feature_to_compare_to, k_group_columns, alternative_dict = None):
    """
    Given a feature dict:{feature_name: results dataframe} and a feature to compare to,
    calculate the p-value using wilxoscon test for each column in the k_group_columns.

    Args:
        feature_dict (dict): Dictionary containing the features data.
        feature_to_compare_to (str): Name of the feature to compare to.
        k_group_columns (list): List of columns to summarize.
        alternative_dict (dict): Dictionary containing the alternative hypothesis for each column.
    Returns:
        pd.DataFrame: Data frame with the p-values.
    """

    if isinstance(feature_dict, pd.DataFrame):
        feature_dict = feature_dict.to_dict(orient='index')
    p_values = pd.DataFrame(columns = k_group_columns, index = feature_dict.keys())
    for feature, results in feature_dict.items():
        if feature == feature_to_compare_to:
            continue
        for col in k_group_columns:
            if alternative_dict is not None:
                alternative = alternative_dict[col]
            else:
                alternative = "two-sided"
            try:
                stat, p_value = wilcoxon(results[col], feature_dict[feature_to_compare_to][col],alternative=alternative)
                p_values.loc[feature, col] = p_value
            except ValueError as e:
                print(f"Error calculating p-value for {feature} and {feature_to_compare_to} on column {col}: {e}")
    return p_values  

def compute_average_k_cross_results(features_results, k_group_columns,
                                     k_groups_alternatives, all_partitions_path, error_file, save_results = True):
    """
    Given a dictionary of {feature_name: results dataframe} calculate the average results of each column in the data frame.
    
    Args:
        features_results (dict): Dictionary containing the features data.
        k_group_columns (list): List of columns to summarize.
        k_groups_alternatives (dict): Dictionary containing the alternative hypothesis for each column.
        all_partitions_path (str): Path to save the results.
        error_file (str): Path to the error file to log issues.
    """
    features = list(features_results.keys())
    averaged_results = averaged_k_cross_results(features_results, k_group_columns,error_file)
    p_vals = get_p_val_k_cross(features_results,'Only-sequence',k_group_columns,k_groups_alternatives)
    average_path = create_folder(all_partitions_path, "averaged_results")
    
    averaged_results.rename(index=key_renaming,inplace=True)
    p_vals.rename(index=key_renaming,inplace=True)
    for col in k_group_columns:

        mean = averaged_results[f'{col}_mean'].values
        std = averaged_results[f'{col}_std'].values
        p_val = p_vals[col]
        
        plot_bar_plot(metric_values=mean,std_values=std,p_values=p_val,
                      feature_names=features,metric_name=col.upper(),plot_name=f'{col.upper()}_k_cross_averages_refined', key_renaming_bool=True
                                            )
    if save_results:
        averaged_results.to_csv(os.path.join(average_path, "averaged_results.csv"))
        p_vals.to_csv(os.path.join(average_path, "p_vals.csv"))
        print(f"Results and p vals are saved in {average_path}")

def compute_average_k_cross_ratios(features_results, k_group_columns,
                                     k_groups_alternatives, all_partitions_path, error_file, save_results = True):
    """
    Given a dictionary of {feature_name: results dataframe} calculate the average ratios of each column in the data frame
    compared to the 'Only-sequence' feature."""
    features = list(features_results.keys())
    ratio_results = compare_by_ratio(features_results,'Only-sequence',k_group_columns)
    ratio_columns = ['auroc','auprc']
    ratio_results = ratio_results[ratio_columns]
    ratio_p_vals = get_p_val_k_cross(ratio_results,'Only-sequence',ratio_columns,k_groups_alternatives)
    ratios_path = create_folder(all_partitions_path, "ratios")
    for col in ratio_columns:
        all_features_values = np.stack(ratio_results[col].values)
        mean = np.mean(all_features_values,axis=1)
        std = np.std(all_features_values,axis=1)
        p_val = ratio_p_vals[col]
        plot_bar_plot(metric_values=mean,std_values=std,p_values=p_val,
                      feature_names=features,metric_name=col.upper(),plot_name=f'{col.upper()}_k_cross_ratio', key_renaming_bool=True
                                            )
        
    if save_results:
        ratio_results.to_csv(os.path.join(ratios_path, "ratios.csv"))
        ratio_p_vals.to_csv(os.path.join(ratios_path, "p_vals.csv"))
        print(f"Ratios and p vals are saved in {ratios_path}")
def merge_by_mismatches(guides_dict, error_file):
    """
    Given a dicionary of {mismatch: {guide: {feature: (y_scores, y_test, indexes)}}}
    merge the scores,tests, indexes of each guide for each mismatch.
    
    Args:
        guides_dict (dict): Dictionary containing the guides data.
        error_file (str): Path to the error file to log issues.
    
    Returns:
        dict: Merged dictionary with the structure {mismatch: {feature: (y_scores, y_test, indexes)}}
    """
    merged_guide_dict = {}
    for mismatch, guide_dict in guides_dict.items():
        merged_guide_dict[mismatch] = defaultdict(lambda: [[], [], []])
        for guide, features in guide_dict.items():
            for feature, (y_scores, y_test, indexes) in features.items():
                merged_guide_dict[mismatch][feature][0].append(y_scores)
                merged_guide_dict[mismatch][feature][1].append(y_test)
                merged_guide_dict[mismatch][feature][2].append(indexes)
        for feature, (y_scores, y_test, indexes) in merged_guide_dict[mismatch].items():
            merged_guide_dict[mismatch][feature][0] = np.concatenate(y_scores, axis=0)
            merged_guide_dict[mismatch][feature][1] = np.concatenate(y_test, axis=0)
            merged_guide_dict[mismatch][feature][2] = np.concatenate(indexes, axis=0)
        test_samples = next(iter(merged_guide_dict[mismatch].values()))[1]
        if sum(test_samples>0) == 0:
            with open(error_file, "a") as f:
                f.write(f"all guides have no positives for mismatch {mismatch}\n")
            merged_guide_dict.pop(mismatch)
            continue
    return merged_guide_dict


def get_guide_information(data_name, guide, statistics_file):
    """
    Returns the information of a single/multiple guide/s from the statistics file.

    Args:
        data_name (str): Name of the dataset.
        guide (str or list): Guide sequence or list of guide sequences.
        statistics_file (str): Path to the statistics file.

    Returns:
        dict: Dictionary containing guide information {guide: {feature: value}}
        for feature in data.

    """
    data = pd.read_csv(statistics_file)
    data = data[data['Data_set'] == data_name] # keep corresponding data
    # if multiple guides
    keys  = ["Data set", "Gene name", "Guide sequence", "Verified OTSs", "Cell-based OTSs", "Potential OTSs"]
    if isinstance(guide, list):
        guide_info = data[data['guide_sequence'].isin(guide)]
        guide_info = guide_info.to_dict(orient='list')
        guide_info["Data set"] = set(guide_info["Data_set"])
        guide_info["sgRNAs"] = len(guide)
        try:
            #guide_info["Verified OTSs"] = 0
            guide_info["Verified OTSs"] = int(sum(guide_info["amplified_otss"]))
        except Exception as e:
            print(f"Error calculating Verified OTSs: {e}")
            guide_info["Verified OTSs"] = 0
        try:
            guide_info["Cell-based OTSs"] = int(sum(guide_info["vivo_otss"]))
        except Exception as e:
            print(f"Error calculating Cell-based OTSs: {e}")
            guide_info["Cell-based OTSs"] = 0
        try:
            guide_info["Potential OTSs"] = int(sum(guide_info["potential_otss"]))
        except Exception as e:
            print(f"Error calculating Potential OTSs: {e}")
            guide_info["Potential OTSs"] = 0
        #guide_info = guide_info.to_dict(orient='list')
        keys.append("sgRNAs")
        keys.remove("Guide sequence")
        keys.remove("Gene name")
    else:
        guide_info = data[data['Guide sequence'] == guide]
        guide_info = guide_info.to_dict(orient='records')[0]
    guide_info = {key: guide_info[key] for key in keys}
    return guide_info

def keep_indexes_from_scores_labels_indexes(y_scores, y_test, indexes, spesific_indexes):
    """
    Get spesific indexes from the y_scores, y_test and indexes.

    Args:
        y_scores (np.array): The prediction scores.
        y_test (np.array): The true labels.
        indexes (np.array): The indexes of the samples.
        spesific_indexes (list): The spesific indexes to keep.
    Returns:
        selected_y_scores (np.array): The selected y_scores.
        selected_y_test (np.array): The selected y_test.
        spesific_indexes (list): The spesific indexes to keep.
    """
    positional_indexes = np.where(np.isin(indexes, spesific_indexes))[0] # get the postional indexes of the spesific indexes
    if y_scores.ndim == 1:
        selected_y_scores = y_scores[positional_indexes]
    elif y_scores.ndim == 2:
        selected_y_scores = y_scores[:, positional_indexes]
    
    selected_y_test = y_test[positional_indexes]
    return selected_y_scores, selected_y_test, spesific_indexes

def init_feature_dict_for_all_scores(ml_results_paths,n_ensebmles, y_test, indexes,  reg_classification = False,
                                     additional_data = None):
    """

    This function will return a dictionary with all the features and their scores, labels and indexes.
    
    Args:
        ml_results_paths (list): list of paths to the models results folders.
            Each folder is a different model.
        n_ensebmles (int): number of ensembles in the results.
        reg_classification (bool): if the task is classification by regression.
            If True, the function will transform the labels to binary labels.
        additional_data (tuple): (name, path) - tuple of additional data to add to the feature dict.
    
    Returns:
        features_dict (dict): dictionary with the features and their scores, labels and indexes.
            Dictionary structure: {feature: (y_scores, y_test, indexes)}
            where y_scores is a 2d array of prediction_scores if more than 1 ensemble.
    
    """
    
    fill_feature_dict_args = []
    for ml_results_path in ml_results_paths:
        if "Only_sequence" in ml_results_path:
            feature = "Only-seq"
        else : 
            feature = ml_results_path.split("/")[-1]
        fill_feature_dict_args.append(({},feature,ml_results_path,n_ensebmles,y_test,indexes,reg_classification))
    proccess = min(os.cpu_count(), len(fill_feature_dict_args))
    features_dict = {}
    if additional_data:
        ##NOTE: validate path and feature
        ml_path = additional_data[0]
        feature_name = additional_data[1]
        additional_data_ = fill_feature_dict_with_scores({},feature_name,ml_path,n_ensebmles,y_test,indexes,reg_classification)
        features_dict.update(additional_data_)
    # with Pool(proccess) as pool:
    #     results = pool.starmap(fill_feature_dict_with_scores, fill_feature_dict_args)
    results = []
    for arg in fill_feature_dict_args:
        results.append(fill_feature_dict_with_scores(*arg))
    
    for result in results:
        features_dict.update(result)
    return features_dict

def get_ots_indexes_till_last_tp(guides_results_dict,guides_dict):
    """
    Saves all the off-targets indexes before and including the last true-positive.

    Returns:
        Dictionary : {guide_seqeunce : np.array (2,number of points till last tp) - with the indexes and the scores}
    """
    guide_indexes_till_last_tp = {}
    for guide_seq, guide_data in guides_dict.items():
        guide_scores,guide_indexes= guide_data['Only-seq'][0],guide_data['Only-seq'][2]
        guide_only_seq_index = guides_results_dict[guide_seq][1].index('Only-seq')
        guide_last_tp = guides_results_dict[guide_seq][0]['last_fn_values'][guide_only_seq_index][0]
        array_to_save = np.zeros(shape=(2,guide_last_tp))
        sorted_predictions = np.argsort(guide_scores)[::-1]
        prediction_values = guide_scores[sorted_predictions][:guide_last_tp]
        all_indexes_till_last_tp = guide_indexes[sorted_predictions[:guide_last_tp]]
        array_to_save[0] = all_indexes_till_last_tp
        array_to_save[1] = prediction_values
        guide_indexes_till_last_tp[guide_seq] = array_to_save
    return guide_indexes_till_last_tp

def fill_feature_dict_with_scores(feature_dict, feature, scores_folder_path, n_ensembles, y_test, indexes,
                                   reg_classification = False):
    """
    For each models in the scores folder path, it will extract the scores, labels and indexes.
    Each ensemble is being averaged and the results are being saved in the feature dict.
    
    Args:
        feature_dict (dict): dictionary with the features and their scores, labels and indexes.
            Dictionary structure: {feature: (y_scores, y_test, indexes)}
            where y_scores is a 2d array of prediction_scores if more than 1 ensemble.
        feature (str): name of the feature.
        scores_folder_path (str): path to the scores folder.
        n_ensembles (int): number of ensembles in the results.
        reg_classification (bool): if the task is classification by regression.
            If True, the function will transform the labels to binary labels.
    Returns:
        feature_dict (dict): dictionary with the features and their scores, labels and indexes.
            Dictionary structure: {feature: (y_scores, y_test, indexes)}
            where y_scores is a 2d array of prediction_scores if more than 1 ensemble.
    """
    if n_ensembles > 1: # multiple ensembles in the results
        ensembles = create_paths(os.path.join(scores_folder_path, "Scores"))
        y_scores = []
        for ensemble in ensembles:
            with open(ensemble,'rb') as score_file:
                ensemble_scores = pickle.load(score_file)
            y_scores.append(ensemble_scores)
        y_scores = np.array(y_scores)
    else: # one ensemble
        with open(create_paths(os.path.join(scores_folder_path, "Scores")),'rb') as f:
            y_scores = pickle.load(f)
        
    if reg_classification: # transform y to binary labels
        y_test = (y_test > 0).astype(int)
    feature_dict[feature] = (y_scores, y_test, indexes)
    return feature_dict

def split_feature_dict_by_indexes(features_dict, indexes_dict, by_mismatch = False):
    """
    Split the features dict {feature: (scores,test,all samples indexes)} by given indexes.
    By defualt it will split by guide indexes.
    If by_mismatch is True, it will split by mismatch indexes.

    Args:
        features_dict (dict): {feature: (scores,test,all samples indexes)}
            dictionary with the features and their scores, labels and indexes.
        indexes_dict (dict): {guide: indexes} - dictionary with the sample indexes of each guide.
        by_mismatch (bool): if to split by mismatch indexes.
            if True, indexes_dict should have:
            {guide: {mismatch: indexes}} - dictionary with the sample indexes of each guide.
     
    Returns: 
        dictionary {guide: {feature : (y_scores, y_test)}}
        if by_mismatch is True, it will return: {mismatch: {guide: {feature : (y_scores, y_test)}}}
    """
    if by_mismatch:
        mismatch_dict = {}
        for mismatch_number, guide_indexes in indexes_dict.items():
            mismatch_dict[mismatch_number] = {}
            for guide, indexes in guide_indexes.items():
                mismatch_dict[mismatch_number][guide] = {}
                for feature, (y_scores, y_test, all_indexes) in features_dict.items():
                    y_indexed_scores, y_indexed_test, indexes = keep_indexes_from_scores_labels_indexes(y_scores, y_test, all_indexes, indexes)
                    mismatch_dict[mismatch_number][guide][feature] = y_indexed_scores, y_indexed_test, indexes
        return mismatch_dict
    guides_dict = {}
    for guide,indexes in indexes_dict.items():
        guides_dict[guide] = {}
        for feature, (y_scores, y_test, all_indexes) in features_dict.items():
            y_indexed_scores, y_indexed_test, indexes = keep_indexes_from_scores_labels_indexes(y_scores, y_test, all_indexes, indexes)
            guides_dict[guide][feature] = y_indexed_scores, y_indexed_test, indexes
    return guides_dict

