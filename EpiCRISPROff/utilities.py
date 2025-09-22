import numpy as np
import os
from itertools import combinations
import pandas as pd

import psutil

# Function to get memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 ** 2)  # Memory in MB

### DATA UTILITIES ###

def keep_positives_by_ratio(X_data, y_data, ratio = None):
    '''This function keeps a subset ratio of the positive data points in the data.
    If ratio is None/<=0/>1 it will through an error.
    Other wise it will keep the ratio of the positive data points for each guide!
    For example if ratio = 0.5 it will keep half of the positive data points for each guide keeping the guide amount the same!
    Args:
    1. X_data - data points - [n_guides,np.array(datapoints,one_hot_vectors)]
    2. y_data - labels - [n_guides,np.array(datapoints,labels)]
    3. ratio - ratio of the positive data points to keep.
    -----------
    Returns: X-data*ratio and y_data*ratio'''
    if ratio is None or ratio <= 0 or ratio >1:
        raise Exception("Ratio must be a number between 0 and 1")
    
    # Sort the data by the labels
    sorted_indices = [] 
    X_data_copy = X_data.copy()
    y_data_copy = y_data.copy()
    for index,labels_array in enumerate(y_data_copy):
        labels_array = labels_array.ravel() # flatten the array
        positive_indices = np.where(labels_array > 0)[0] # keep only the positive labels
        total_amount = len(positive_indices)
        sorted_indices = np.argsort(labels_array[positive_indices])[::-1] # get the indices that sort the labels
        amount_to_keep = int(total_amount*ratio) # amount of positive data points to keep
        if amount_to_keep <= 2 or total_amount <= 2: # keep at least 2 positive data points
            sorted_indices = sorted_indices[:2]
            y_data_copy[index] = y_data_copy[index][sorted_indices] 
            X_data_copy[index] = X_data_copy[index][sorted_indices] 
            continue
        sorted_indices = sorted_indices[:amount_to_keep] # keep the first ratio of the positive data points
        y_data_copy[index] = y_data_copy[index][sorted_indices] # sort the labels
        X_data_copy[index] = X_data_copy[index][sorted_indices] # sort the features
        if len(y_data_copy[index]) != len(X_data_copy[index]):
            raise Exception("The labels and features must have the same amount of data points")
        
    return X_data_copy,y_data_copy

def keep_negatives_by_ratio(X_data, y_data, ratio = None):
    '''This function keeps a subset ratio of the negative data points in the data.
    If ratio is None/<=0/>1 it will through an error.
    Other wise it will keep the ratio of the negative data points for each guide!
    For example if ratio = 0.5 it will keep half of the negative data points for each guide keeping the guide amount the same!
    Args:
    1. X_data - data points - [n_guides,np.array(datapoints,one_hot_vectors)]
    2. y_data - labels - [n_guides,np.array(datapoints,labels)]
    3. ratio - ratio of the negative data points to keep.
    -----------
    Returns: X-data*ratio and y_data*ratio'''
    if ratio is None or ratio <= 0 or ratio >1:
        raise Exception("Ratio must be a number between 0 and 1")
   
    X_data_copy = X_data.copy()
    y_data_copy = y_data.copy()
    for index,labels_array in enumerate(y_data_copy):
        labels_array = labels_array.ravel() # flatten the array
        negative_indice = np.where(labels_array==0)[0] # keep only the negative labels indices
        total_amount = len(negative_indice)
        amount_to_keep = int(total_amount*ratio) # amount of positive data points to keep
        
        if amount_to_keep < 2: # keep at least 2 positive data points
            continue
        shuffled_indices = np.random.permutation(negative_indice)[:amount_to_keep] # shuffle the indices and keep the first ratio of the negative data points
        y_data_copy[index] = y_data_copy[index][shuffled_indices] # sort the labels
        X_data_copy[index] = X_data_copy[index][shuffled_indices] # sort the features
        if len(y_data_copy[index]) != len(X_data_copy[index]):
            raise Exception("The labels and features must have the same amount of data points")
    return X_data_copy,y_data_copy

def keep_positive_OTSs_labels(y_scores, y_test, indexes):
    '''Function keeps only the positive OTSs and their corresponding scores.
    It remove the zero labels to avoid floating point errors.
    Args:
    1. y_test - true labels
    2. y_scores - predicted labels
    3. indexes - indexes of the data points
    -----------
    Returns: positive OTSs labels, scores and indexes'''
    zero_y_test = np.where(y_test == 0)[0] # zero indices
    pos_y_test = np.delete(y_test,zero_y_test) # get only the positive OTSs
    
    pos_y_scores = np.delete(y_scores,axis=1,obj=zero_y_test) # get only the predicted positive OTSs
    pos_indexes = np.delete(indexes,zero_y_test) # get only the positive OTSs indexes
    return pos_y_scores, pos_y_test, pos_indexes
def convert_partition_str_to_list(partition_str):
    '''This function will convert a partition string to a list of integers
    Args:
        partition_str - string with the partition
        -----------
        returns a list of integers
        '''
    partitions_nums = partition_str.split("_")[0]
    num_strings = partitions_nums.split('-')
    return [int(num) for num in num_strings]

def validate_dictionary_input(answer, dictionary):
    '''This function is used to validate the input from the user.
 It checks if the input is a valid number and if it is within the range of the list of options
 given by keys of the dictionary'''
    if not answer: # no input was given
        answer = input(f"Please enter number: {dictionary}\n") # ask for model
    # input was given
    answer = int(answer)
    if answer not in dictionary.keys(): # check if the input is in the dictionary
        raise Exception(f"Invalid input. Please choose from {dictionary.keys()}")
    else :
        return answer


        
'''Function writes 2d array to csv file'''
def write_2d_array_to_csv(np_array, file_path, header, fmt = '%.16e'):
    if np_array.ndim != 2:
        raise Exception("np_array must be 2d")
    if file_path.split(".")[-1] != "csv":
        raise Exception("file_name must end with csv")
    if header: # not None/empty
        if len(header) != np_array.shape[1]:
            raise Exception("header must be the same length as the number of columns in the np_array") 
   
    np.savetxt(file_path, np_array, delimiter=',', fmt=fmt, header=','.join(header), comments='')
def add_row_to_np_array(y_scores, y_test):
    # if y_scores.dtype != y_test.dtype:
    #     raise Exception("y_scores and y_test must have the same dtype")
    if y_scores.ndim != 2 or y_test.ndim != 1:
        raise Exception("y_scores must be 2d and y_test must be 1d")
    if y_scores.shape[1] != y_test.shape[0]:
        raise Exception("y_scores must have the same number of columns as y_test number of values")
    return np.vstack((y_scores, y_test))

def extract_scores_labels_indexes_from_files(paths):
    '''
    Given a list of paths for csv files containing models predicitions scores
extract the scores and combine them into one np array.
The last line the file should contain the indexes of each data point
The second raw from the end the actual labels
 '''
    if isinstance(paths, list):
        if len(paths) == 0:
            raise Exception("Paths list is empty")
    all_scores = []
    for path in paths:
        # Read the file
        scores_lbl_idx = np.genfromtxt(path, delimiter=',')
        scores = scores_lbl_idx[:-2] # keep only the scores
        all_scores.append(scores) # Add the scores to the scores array
    # get indices and labels
    indexes = scores_lbl_idx[-1]
    y_test = scores_lbl_idx[-2]
    # Concate the arrays
    all_scores = np.concatenate(all_scores)
    return all_scores,y_test,indexes           

def get_k_choose_n(n,k):
    indices_combinations = list(combinations(range(1,n+1), k))
    return indices_combinations

def get_X_random_indices(N, k, X):
    '''Create X random combinations sized k out of N
    Example: N = 50, k = 3, X = 200
    Will create 200 random combinations of 3 indices out of 50 without replacement
    --------------
    Returns a list of X random combinations of k indices out of N'''
    # Create a list of indices from 0 to n_models
    indices = list(range(N))
    all_indices = []
    # Create X random combinations of k indices
    for j in range(X):
        random_indices = np.random.choice(indices, k, replace=False)
        all_indices.append(random_indices)
    return all_indices



def save_intersect_guides(dataset_1, dataset_2, target_col_1, target_col_2, output_name):
    '''Given two datasets and the target columns to compare
    Save the intersection of the target columns at output_name'''
    target_1 = set(dataset_1[target_col_1])
    target_2 = set(dataset_2[target_col_2])
    intersect = target_1.intersection(target_2)
    intersect = list(intersect)
    intersect_df = pd.DataFrame(intersect, columns=['Intersect_guides'])
    intersect_df.to_csv(output_name, index=False)
    print(f"Intersect guides saved at {output_name}")


## This function might not be used in the future
def fill_tprs_fpr_by_max_length(tprs, fprs):
    '''This function find the maximum length of the tprs and fprs and fill the arrays that are shorter with their last value.
    This function is for plotting the tprs,fprs togther on the same plot.
    Args:
    1. tprs - list of tprs arrays
    2. fprs - list of fprs arrays
    -----------
    Returns the filled tprs and fprs arrays'''
    # Find max length index
    max_len_index = np.argmax([len(fpr) for fpr in fprs])
    max_length = len(fprs[max_len_index])
    # Fill other indexes with their last value
    for i in range(len(tprs)):
        if i != max_len_index:
            last_tpr = tprs[i][-1]
            last_fpr = fprs[i][-1]
            tprs[i] += [last_tpr] * (max_length - len(tprs[i]))
            fprs[i] += [last_fpr] * (max_length - len(fprs[i]))
    return tprs, fprs

# Function to calculate averages for NchooseK
def calculate_averages_partitions(df, K):
    '''Given a data frame with columns: Positives, Negatives, Guides_amount
    The function will calculate the average positives, negatives and guides amount for each union of K rows
    Args:
        df - data frame with columns: Positives, Negatives, Guides_amount
        K - number of rows to union'''
    combinations_results = list(combinations(df.index, K))
    averages = {"Positives": [], "Negatives": [], "Guides_amount": []}

    for combo in combinations_results:
        selected_rows = df.loc[list(combo)]
        avg_pos = selected_rows['Positives'].sum()
        avg_neg = selected_rows['Negatives'].sum()
        avg_amount = selected_rows['Guides_amount'].sum()
        averages['Positives'].append(avg_pos)
        averages['Negatives'].append(avg_neg)
        averages['Guides_amount'].append(avg_amount)
    averages = {key: np.array(value) for key, value in averages.items()}
    averages = {key: value.mean() for key, value in averages.items()}
    return averages

def union_partitions_stats(data_path):
    '''Given a data path with the partitions stats get the num of rows and print the average stats for each union of k rows
    Args:
        data_path - path to the data'''
    data = pd.read_csv(data_path)
    rows = len(data)
    for k in range(1, rows + 1):
        averages = calculate_averages_partitions(data, k)
        print(f"Union of {k} rows: Positives: {averages['Positives']}, Negatives: {averages['Negatives']}, Guides amount: {averages['Guides_amount']}")

### Numeric Validations
def validate_non_negative_int(number):
    '''
    Returns the number if it is a non-negative integer.
    Else raise an error.
    '''
    if not isinstance(number, int):
        raise Exception("Number must be an integer")
    if number < 1:
        raise Exception("Number must be positive")
    return number


def print_dict_values(columns_config):
    '''This function prints the json columns.'''
    # Access constants from the dictionary
    for key, value in columns_config.items():
        print(f'{key}: {value}')
