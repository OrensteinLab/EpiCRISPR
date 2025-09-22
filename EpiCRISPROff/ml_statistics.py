## script for analysis of ml results
import pandas as pd
import numpy as np
import os
from scipy.stats import wilcoxon,mannwhitneyu,pearsonr,spearmanr



def pearson_correlation(x,y):
    '''This function will return the pearson correlation between x and y.
    Args:
    1. x - array
    2. y - array
    ----------
    Returns: r - pearson correlation, p - p value'''
    r,p = pearsonr(x,y)
    return r,p
def spearman_correlation(x,y):
    '''This function will return the spearman correlation between x and y.
    Args:
    1. x - array
    2. y - array
    ----------
    Returns: r - spearman correlation, p - p value'''
    r,p = spearmanr(x,y)
    return r,p

'''
wilxocon test for paired samples, x-y
recomendation is to give the function only the x-y difrences vector.'''
def get_wilxocon_sign(x,y,test):
    # define test direction
    alternative = get_alternative(test)
    x = x.round(5) # 5 decimels after the dot
    y = y.round(5)
    difrrences = (x-y)
    T_statistic,p_val = wilcoxon(x=difrrences,y=None,alternative=alternative)
    stats_dict = {"T_sts":T_statistic,"P.v":p_val}
    return stats_dict
def get_alternative(test):
    if test == "<":
        alternative = "less"
    elif test == ">":
        alternative = "greater"
    else: alternative = "two-sided"
    return alternative
def extract_ml_data(ml_summary_table):
    # Read all sheets into a dictionary of DataFrames
    all_sheets = pd.read_excel(ml_summary_table, sheet_name=None)
    modified_sheets,main_sheet_name = remove_main_sheet(all_sheets,ml_summary_table)
    results_dict = {key: None for key in modified_sheets.keys()} # set a result dict with keys as sheet names
    # Retrive from each data frame columns 1,2 (auroc) 4,5(auprc)
    for key,df in modified_sheets.items():
        y_auroc = df.iloc[:,0].values # first column for y values
        x_auroc = df.iloc[:,1].values # second column for x values
        y_auprc = df.iloc[:,3].values
        x_auprc = df.iloc[:,4].values
        wilxco,manwhi = run_statistics(x=x_auroc,y=y_auroc,test_sign=">")
        add_to_results_dict(results_dict=results_dict,name=key,stats_vals=wilxco,test="wilcoxon_sign",metric="auroc")
        add_to_results_dict(results_dict=results_dict,name=key,stats_vals=manwhi,test="manwhitenyu",metric="auroc")
        wilxco,manwhi = run_statistics(x=x_auprc,y=y_auprc,test_sign=">")
        add_to_results_dict(results_dict=results_dict,name=key,stats_vals=wilxco,test="wilcoxon_sign",metric="auprc")
        add_to_results_dict(results_dict=results_dict,name=key,stats_vals=manwhi,test="manwhitenyu",metric="auprc")
    write_dict_to_file(results_dict=results_dict,name=main_sheet_name)
def run_statistics(x,y,test_sign):
    stats1 = get_wilxocon_sign(x=x,y=y,test=test_sign)
    stats2 = get_manwhitenyu(x=x,y=y,test=test_sign)
    return stats1,stats2

def get_manwhitenyu(x,y,test):
    alternative= get_alternative(test)
    x = x.round(5) # 5 decimels after the dot
    y = y.round(5)    
    T_stat,P_val = mannwhitneyu(x=x,y=y,alternative=alternative,method="asymptotic") # asymptotic for more then 8 samples
    stats_dict = {"T_sts":T_stat,"P.v":P_val}
    return stats_dict
def add_to_results_dict(results_dict,name,stats_vals,test,metric):
    metric_dict = {metric : stats_vals} # dict for metric and stats values
    test_dict = {test : metric_dict} 

    if results_dict[name] == None: # key hasnt been setted with a value
        results_dict[name] = test_dict
    else: # update exsiting sub dictionary
        sub_dict = results_dict[name] 
        if test in sub_dict.keys(): # if the test has been made on other data 
            old_metric_dict = sub_dict[test]
            old_metric_dict.update(metric_dict)
        else: sub_dict.update(test_dict) 
        
def write_dict_to_file(results_dict,name):
    file_name = "Statistics_" + name + ".txt"
    with open(file_name, 'w') as file:
        for key, sub_dict in results_dict.items(): # key is the name of data
            for test,metric_dict in sub_dict.items(): # test is the statistic test
                base_str = f"{key}: {test}: "
                added_str = ""
                for metric,stats_val in metric_dict.items(): # metric - type of data
                    added_str = added_str + f"{metric}: {stats_val}, " # stats- are values of the test
                added_str = base_str + added_str + "\n"
                file.write(added_str)



  

         
def remove_main_sheet(sheets_dict,ml_path):
    # Get the main sheet name by the path
    main_sheet_name = ml_path.split("/")[-1].replace(".xlsx","")
    print(f"Dict before: {sheets_dict.keys()}")
    temp_dict = sheets_dict.copy()
    removed_value = temp_dict.pop(main_sheet_name, None)
    if removed_value is not None:
        print(f"Dictionary after removing '{main_sheet_name}': {sheets_dict.keys()}")
    else:
        print(f"Key '{main_sheet_name}' not found in the dictionary.")
    return temp_dict,main_sheet_name
## REPRODUCIBILITY ANALYSIS
'''Given a path for a folder eather DATA REPRODUCIBILITY or MODEL REPRODUCIBILITY,
iterate on the files and do the following:
1. DATA REPRODUCIBILITY: the data spliting is the same over all replicates
    a) Each file contains auroc,auprc,n-rank for k data splits.
    b) Calculate the mean and std for each data split over all files.
    c) Calculate the mean and std for all splits over all files.
2. MODEL REPRODUCIBILITY: the model initialization is the same over all replicates
    a) For each file (replicate) calculate the mean and std of the auroc and auprc
    b) Calculate the mean and std over all the given files
    '''
def extract_reproducibility_data(reproducibility_data_paths, k_splits, model_name, repro_type):
    # 1. Init np arrays from the shape (num_files,num_splits) in a dictionary
    keys = ["Auroc","Auprc","N-rank"]
    rows = len(reproducibility_data_paths)
    results_dict = {key: np.zeros((rows,k_splits)) for key in keys}
    
    # 2. Iterate on each file and append the results to the np arrays row wise  
    for i,path in enumerate(reproducibility_data_paths):
        # 2.1. Read the data
        data = pd.read_csv(path)
        # 2.2. Append the data to the np arrays
        for key in keys:
            results_dict[key][i,]= data[key].values
        
    # 3. Add to each row its mean. Add to each column its mean and std
    for key in keys:
        results_dict[key] = get_auroc_auprc_nrank_mean_std(results_dict[key])
        write_reproducibility_results_to_file(results_dict[key],model_name,repro_type,key,k_splits)
''' Given a np array where each row is a file data, each colum is a data split data
return the mean and std for each column (data split), and calculate the mean and std
for all rows (all replicates) and calculate mean and std of these. '''       
def get_auroc_auprc_nrank_mean_std(np_array ):
    # 1. Calculate the mean and std for each row (replicate) and add it to the end of the raw
    mean_replicate = np.mean(np_array,axis=1)
    std_replicate = np.std(np_array,axis=1)
    # 2. Add the mean and std to the end of the row
    np_array = np.column_stack((np_array,mean_replicate))
    np_array = np.column_stack((np_array,std_replicate))
    # 3. Calculate the mean and std for each column (data split)
    mean_data_split = np.mean(np_array,axis=0)
    std_data_split = np.std(np_array,axis=0)
    # 4. Add the mean and std to new row each
    np_array = np.row_stack((np_array,mean_data_split))
    np_array = np.row_stack((np_array,std_data_split))
    
    return np_array
    
def write_reproducibility_results_to_file(np_results, model_name, repro_type, key,k_splits):
    file_name = f"{model_name}_{key}_{repro_type}_Reproducibility_summary.csv"
    # Header is the number of columns -1 and mean
    header_str = ""
    for i in range(k_splits):
        header_str = header_str + f'{i+1},'
    header_str = header_str +  "Mean,Std" 
    np.savetxt(file_name, np_results, delimiter=',', fmt='%.5f', header=header_str, comments='')
def create_paths(folder):
        paths = []
        for path in os.listdir(folder):
            paths.append(os.path.join(folder,path))
        return paths

def get_only_seq_vs_group_ensmbels_stats(ensemble_dict,n_models,compare_to,difference_only = False,compare_to_scores= None,groups_scores=None):
    '''
    This function will return the p val statitcs for the wilcoxon test for given set of features agiasnt a spesific label.
    If difference_only is True, the function will use the group scores as the difference between the group and wanted label.
    Args:
    1. ensemble_dict - dictionary with results as follows: {key -> group_name, val -> {key -> n_models, val -> np array of results}}
    2. n_models - number of models in the ensemble.
    3. compare_to - the group to compare the results to.
    4. difference_only - if True, the function will use the group scores as the difference between the group and wanted label.
    5. compare_to_scores - if given, the function will use this scores as the compare_to scores.
    6. groups_scores - if given, the function will use this scores as the group scores.
    ------
    Returns: a dictionary with the comparison results for each group to the compare_to group.
    '''
    # 1. Get the Only_seq scores
    if compare_to_scores is None:
        if ensemble_dict: # not None
            compare_to_scores = get_values_from_ensmbel_dict(ensemble_dict[compare_to],n_models)
    if difference_only:
        compare_to_scores = None
    if groups_scores is None:
        groups_scores = {group: get_values_from_ensmbel_dict(ensemble_dict[group],n_models) for group in ensemble_dict.keys() if group != compare_to}
    else :
        groups_scores = {group: get_values_from_ensmbel_dict(groups_scores[group],n_models) for group in groups_scores.keys()}    
    # 2. Compare the scores of each group to the only seq
    compare_dict = {} # Init dict that hold the comparison results key: seq vs _, value: stats
    for group,group_score in groups_scores.items():
        stats = extract_roc_prc_nrank_pvals(compare_to_scores,group_score)
        compare_dict[group] = stats
    return compare_dict 


def get_mean_std_from_ensmbel_results(ensmbel_results):
    '''Given a dictionary of ensmbel results:
     {key : folder_name, val: dict{key: n_models_in_ensmbel, val: np array of results}
    calculate the mean and std of the results for each ensmbel.
    ------
    returns a dictionary with the mean and std for each ensmbel:
    classification - [0] -auroc,[1] - auprc,[2] - n-rank 
    regression - [0] r_pearson, [1] pval_persson, [2] r_spearman, [3] pval_spearman, [4] mse'''
    ensmbel_mean_std = {}
    

    for ensmbel,results in ensmbel_results.items():
        if isinstance(results, np.ndarray):
            ensmbel_mean_std[ensmbel] = (np.mean(results, axis = 0), np.std(results,axis = 0))
        else:
            ensmbel_mean_std[ensmbel] = {n_models: (np.mean(results[n_models],axis=0),np.std(results[n_models],axis=0)) for n_models in results.keys()}
    return ensmbel_mean_std

def get_values_from_ensmbel_dict(ensemble_dict, n_models):
    '''This function will return the auroc,auprc,n-rank values for a given number of models
    in ensmbel.'''
    if isinstance(ensemble_dict,np.ndarray):
        # return each column separetly
        return tuple(ensemble_dict[:, col] for col in range(ensemble_dict.shape[1]))
    else:
        # return tuple(ensemble_dict[n_models][:,col] for col in range(len(ensemble_dict)))
        return ensemble_dict[n_models][:,0] , ensemble_dict[n_models][:,1], ensemble_dict[n_models][:,2]
    
def extract_roc_prc_nrank_pvals(compare_to_scores, group_scores):
    '''
    This function will take the compare to scores and the group scores and
    return the p-values of the wilcoxon test for each metric in the scores.
    0 - auroc, 1 - auprc, 2 - nrank - classification
    0 - r_pearson, 1 - r_spearman, 2 - mse - regression'''
    p_vals = []
    if not compare_to_scores: # differnece is given in group scores
        for group_metric in group_scores:
            p_vals.append(wilcoxon(group_metric,alternative="greater")[1])
            
    else:   
        for compare_metric,group_metric in zip(compare_to_scores,group_scores):
            p_vals.append(wilcoxon(compare_metric,group_metric,alternative="less")[1])
            
    return p_vals    

