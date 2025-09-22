import numpy as np
import pandas as pd
import pickle

import os
from evaluation_utilities import *
from file_utilities import find_target_files
from k_groups_utilities import get_partition_information
#from plotting import plot_ensemeble_preformance,plot_ensemble_performance_mean_std,plot_roc, plot_correlation, plot_pr, plot_n_rank, plot_last_tp, plot_subplots
from ml_statistics import get_only_seq_vs_group_ensmbels_stats, get_mean_std_from_ensmbel_results

PATH_TO_STATISTICS_FILE = "Data/guides_statistics.csv"
class evaluation():
    def __init__(self, task, only_pos=False):
        self.results_header = [] # init results header
        self.combi_suffix = "Combi"
        self.reg_classification = False
        self.set_task(task)
        pass
    
    def set_task(self, task):
        if task.lower() == "regression" or task.lower() == "t_regression":
            self.task = "regression"
            self.results_header = ["R_pearson","R_spearman","MSE","P.pearson","P.spearman","PR_STD","PP_STD","SR_STD","SP_STD","MSE_STD"]
            self.set_only_positive(True)
        elif task.lower() == "classification" or task.lower() == "reg_classification":
            self.set_only_positive(False)
            self.task = "classification"
            if task.lower() == "reg_classification":
                self.reg_classification = True
                self.combi_suffix = "Combi_reg"
                #NOTE: CHECK THE ADDDITION OF LAST TP
            self.results_header = ["Auroc","Auprc","N-rank","Last-TP-index","Last-TP-ratio","Auroc_std","Auprc_std","N-rank_std","Last-tp_std","Last-tp_Ratio-std"]
        self.set_data_evaluation_columns()
    def set_only_positive(self, only_pos= False):
        self.only_positive = False
        if only_pos:
            self.only_positive = True
            self.combi_suffix = "Combi_pos"    
    
    def set_features(self, features):
        self.features = features
    def set_data_evaluation_columns(self):
        if self.task.lower() == "classification" or self.task.lower() == "reg_classification":
            #NOTE: CHECK THE ADDDITION OF LAST TP

            self.data_evaluation_columns = ['feature', 'auroc', 'auroc_std','auroc_pval', 'auprc', 'auprc_std','auprc_pval', 'n-rank', 
                                            'n-rank_std','n-rank_pval', 'last-tp', 'last-tp_std','last-tp_pval','last-tp_ratio','last-tp_ratio_std']
            self.k_group_columns = ['partition','auroc', 'auprc', 'n-rank', 'last-tp-ratio','last-tp-index']
            self.k_groups_alternatives = {"auroc":'greater', "auprc":'greater', "n-rank":'greater', "last-tp-ratio":'less', "last-tp-index":'less'}
        elif self.task.lower() == "regression" or self.task.lower() == "t_regression":
            self.data_evaluation_columns = ['feature', 'pearson', 'pearson_std','pearson_pval', 'spearman', 'spearman_std','spearman_pval', 'mse', 'mse_std','mse_pval']
            self.k_group_columns = ['partition','pearson', 'spearman', 'mse']
            self.k_groups_alternatives = {"pearson":'greater', "spearman":'greater', "mse":'less'}
    

    #### Evaluate predictions ####
    def evaluate_k_cross_results(self, ml_results_path, plots_path, save_results = True, evaluate_single_partition = True):
        """
        Evaluate the k-fold cross-validation results.
        Given an ml_results_path, extracts the raw_scores.pkl files for the k_cross partitions.
        Conduct 2 evaluations:
            1) Evaluate each partition by it self comparing all the different models.
                if evaluate_single_partition is False it will avoid this part.
            2) Evaluate the average performance of each model over all partitions.
        Plots the 2 types of evluations.
        
        Args:
            ml_results_path (str): Path to the ML results folder.
            plots_path (str): Path to save the plots.
            save_results (bool): If True, create a results folder in the plots path
                and saves the results in it.
            evaluate_single_partition (bool): If True, evaluate each partition by itself.
        """
        # Get the scores for all the models. Score are in pkl files.
        paths_to_scores = find_target_files(os.path.join(ml_results_path.split('K_cross')[0], "K_cross"), "raw_scores.pkl")
        #paths_to_scores = [path for path in paths_to_scores if 'downstream' not in path]
        #paths_to_scores = [path for path in paths_to_scores if 'downstream_seq' in path]
        if paths_to_scores == []:
            raise FileNotFoundError(f"No raw_scores.pkl files found in the K_cross folder: {ml_results_path}")
        # Open pkl files for each feature.
        features_dict = {}
        for path in paths_to_scores:
            feature_name = os.path.basename(path)
            score_file = os.path.join(path,'raw_scores.pkl')
            with open(score_file, 'rb') as file:
                features_dict[feature_name] = pickle.load(file)
        features_dict['Only-sequence'] = features_dict.pop('Only_sequence') # change key name
        # Feature dict : {feature: {partition: (y_scores, y_test, indexes)}}
        error_file = os.path.join(plots_path, "error.txt")
        if evaluate_single_partition:
            converted_feature_dict = convert_k_cross_dict(features_dict,error_file) # convert to {partition: {feature:(y_scores, y_test, indexes)}}
            # 1. Evaluate each partition by it self comparing all the different models.
            by_partition_path = create_folder(plots_path, "By_partition")
            for partition, partition_dict in converted_feature_dict.items():
                temp_path = create_folder(by_partition_path, f'{partition}_partition')
                plot_evalutions_for_multiple_models(task=self.task,output_path=temp_path,
                                                    scores_dictionary=partition_dict)
        # 2. Evaluate the average performance of each model over all partitions.
        features_results = {}
        for feature, feature_partitions in features_dict.items():

            feature_results = get_k_groups_results(feature_partitions, self.task, self.k_group_columns)
            features_results[feature] = feature_results
        self.k_group_columns.remove('partition')
        #plots_path = os.path.join(plots_path,'withDownstream_in_sequence')
        all_partitions_path = create_folder(plots_path, "All_partitions")
        # Save results                
        with pd.ExcelWriter(os.path.join(all_partitions_path,'results_summary.xlsx'), engine='openpyxl') as writer:
            for col in self.k_group_columns:
                sheet_df = pd.DataFrame()
                for feature_name, df in features_results.items():
                    sheet_df[feature_name] = df[col].values
                sheet_df.to_excel(writer, sheet_name=col, index=False)
        # Average

        compute_average_k_cross_results(features_results, self.k_group_columns, self.k_groups_alternatives,
                                        all_partitions_path,error_file=error_file,save_results=save_results)
        # Ratio
        # compute_average_k_cross_ratios(features_results, self.k_group_columns, self.k_groups_alternatives,
        #                                 all_partitions_path,error_file=error_file,save_results=save_results)    
        
        
        
    
    def plot_k_groups_results(self, k_results_dicionary, plots_path, feature_name):
        plot_evalutions_for_multiple_models(self.task,plots_path,feature_name,None,k_results_dicionary)


    def set_partition_information(self, data_path, partitions, ensembles, models):
        '''This function returns a dictionary with the informatino of the partition.
        If more than one partition is given, the function will sum the positives, negatives, and guides 
        from each partition.'''
        partitions_info = {
            "Ensembles" : ensembles, "Models" : models
        }
        if isinstance(partitions, int):
            partitions_info.update(get_partition_information(data_path, partitions))
            partitions_info["Partition"] = partitions 
        else:
            positives =  negatives =  number_of_guides = 0
            for partition in partitions:  # More than 1 partition
                partition_data = get_partition_information(data_path, partition)
                positives += partition_data["Positives"]
                negatives += partition_data["Negatives"]
                number_of_guides += partition_data["sgRNAs"]
            partitions_info["Positives"] = positives
            partitions_info["Negatives"] = negatives
            partitions_info["sgRNAs"] = number_of_guides
            partitions = [str(partition) for partition in partitions]
            partitions_info["Partitions"] = ",".join(partitions)
        return partitions_info
    
    
    def evaluate_multiple_models_per_feature(self, features_dict):
        '''
        This function accepts a dictionary of features and their scores, labels and indexes.
        Dict: {feature: (y_scores, y_test, indexes)} where y_scores is a 2d array of prediction_scores.
        The function evaluates each prediction_score of each feature and returns a new dictionary with the results.
        Args:
        1. features_dict - Dict: {feature: (y_scores, y_test, indexes)} where y_scores is a 2d array of prediction_scores.
        -----------
        Returns: dictionary {feature: (results)} where results is a 2d array of the evaluation metrics.
        '''
        # Every feature has n_ensemble - each ensebmle is a model
        new_feature_dict = {}
        for feature, (y_scores, y_test, indexes) in features_dict.items():
            feature_results = np.array([
        evaluate_model(y_test, scores, self.task)
        for scores in y_scores
        ])
            new_feature_dict[feature] = feature_results
        return new_feature_dict
    
    def plot_multiple_ensembles_per_guide(self, guides_dict, feature_dict, n_ensembles, plots_path, data_name):
        """ Plot results for multiple ensembles over seperate guides and all guides togther.
        
        Args:
            guides_dict (dict): spesific guide predictions after getting the predictions for that guide.
                 {guide: {feature : (y_scores, y_test)}} - y_score - np.array of shape (n_ensembles, n_samples)
            feature_dict (dict): predictions over all data points. {feature : (y_scores, y_test)} - y_score - np.array of shape (n_ensembles, n_samples).
            n_ensembles (int): Number of ensembles.
            plots_path (str): Path to save the plots.
            data_name (str): Name of the data set.
            """
        ## NOTE: Make multi process
        # for guide,features in guides_dict.items():
        #     print("Checking guide: ",guide)
        #     guides_dict[guide] = self.evaluate_multiple_models_per_feature(features)
        #     try:
        #         p_vals = get_only_seq_vs_group_ensmbels_stats(guides_dict[guide],n_ensembles,compare_to="Only-seq")
        #     except ValueError as e:
        #         print(e)
        #         continue
        #     if self.task == "classification": 
        #         # get_only_seq_vs_group_ensmbels_stats uses mannwhitneyu test which look for less than hypothesis.
        #         # values 3,4 correspond to last tp which should be less than and the hypothesis sign should be changed.
        #         for key, values in p_vals.items():
        #             values[3] = 1 - values[3]  # Update pval4 (index 3)
        #             values[4] = 1 - values[4]  # Update pval5 (index 4)
        #     mean_std = get_mean_std_from_ensmbel_results(guides_dict[guide])
        #     guide_path = os.path.join(plots_path,f'{guide}',f'{n_ensembles}_ensembles')
        #     create_folder(guide_path)
        #     partition_info = get_guide_information(data_name, guide,PATH_TO_STATISTICS_FILE)
        #     args = (mean_std,"all_features",p_vals,guide_path,self.task)
        #     plot_ensembles_by_features_and_task(args,self.task,partition_info)
        # all guides
        all_guides_path = os.path.join(plots_path,"All_guides",f'{n_ensembles}_ensembles')
        create_folder(all_guides_path)
        #partition_info = get_guide_information(data_name, list(guides_dict.keys()),PATH_TO_STATISTICS_FILE)
        feature_dict = self.evaluate_multiple_models_per_feature(feature_dict)
        p_vals = get_only_seq_vs_group_ensmbels_stats(feature_dict,n_ensembles,compare_to="Only-seq")
        if self.task == "classification":
            for key, values in p_vals.items():
                values[3] = 1 - values[3]
                values[4] = 1 - values[4]
        mean_std = get_mean_std_from_ensmbel_results(feature_dict)
        # save the results
        with open(os.path.join(all_guides_path, "all_features.pkl"), 'wb') as f:
            pickle.dump(feature_dict, f)
        with open(os.path.join(all_guides_path, "mean_std.pkl"), 'wb') as f:
            pickle.dump(mean_std, f)
        rows = {k: list(v[0]) + list(v[1]) for k, v in mean_std.items()}
        df = pd.DataFrame.from_dict(rows, orient='index', columns=self.results_header)
        df.to_csv(os.path.join(all_guides_path, "mean_std.csv"), index_label='feature')
        with open(os.path.join(all_guides_path, "p_vals.pkl"), 'wb') as f:
            pickle.dump(p_vals, f)
        rows = {k: v[:2] for k,v in p_vals.items()}
        df = pd.DataFrame.from_dict(rows, orient='index', columns=['AUROC_Pval', 'AUPRC_Pval'])
        df.to_csv(os.path.join(all_guides_path, "p_vals.csv"), index_label='feature')
        # args = (mean_std,"all_features",p_vals,all_guides_path,self.task)
        # plot_ensembles_by_features_and_task(args,self.task,partition_info)
    def evaluate_test_per_guide(self, ml_results_paths, n_ensembles, y_test, indexes, indexes_dict , plots_path, data_name,
                                additional_data = None , if_save_last_tp_gap = True, by_mismatch = False):
        """
        Evaluate the ensemble/s performance over the different guides in the test set and all of them as whole test set.
        Plots 3 plots: AUPR,AUROC,LAST-tp, where all the guides are plotted as sub-plots.
        If by_mismatch is True, it will plot the results for each mismatch.
        
        Given a list of ensemble results paths, uses init_feature_dict_for_all_scores to get
        the predictions of each ensemble and the labels. 
        split the predictions for each guide and evalute the performance.
        if save_last_tp_gap is True, it will save the INDEXES of all the predictions of the otss till the last-tp.
        Args:
            ml_results_paths (list): List of paths to the ML results folders.
            n_ensembles (int): Number of ensembles.
            indexes_dict (dict): {guide: indexes} - dictionary with the sample indexes of each guide.
            plots_path (str): Path to save the plots.
            data_name (str): Name of the data set.
            additional_data (tuple): (name, path) - tuple of additional data to add to the feature dict.
            if_save_last_tp_gap (bool): If to save the last tp gap.
            by_mismatch (bool): If to split the features dict by mismatch indexes.
        ----------
        """
        # get the scores for each ensmbel
        feature_dict = init_feature_dict_for_all_scores(ml_results_paths, n_ensembles, y_test, indexes,
                                                         self.reg_classification, additional_data=additional_data)
        
        #guides_dict = split_feature_dict_by_indexes(feature_dict, indexes_dict, by_mismatch)
        if n_ensembles > 1: 
            self.plot_multiple_ensembles_per_guide(None,feature_dict,n_ensembles,plots_path, data_name)
        return
        if by_mismatch:
            self._evaluate_test_by_mismatch(plots_path, guides_dict, data_name, if_save_last_tp_gap)
            
        else:
            self._evaluate_test_by_guide(guides_dict,feature_dict,data_name,plots_path, if_save_last_tp_gap)
            
    
    def _evaluate_test_by_mismatch(self, plots_path, guides_dict, data_name, if_save_last_tp_gap=True):
        mismatch_path = create_folder(plots_path, "By_mismatch")
        
        error_file = os.path.join(mismatch_path, "error.txt")
        
        for mismatch, guide_dict in guides_dict.items(): # for each mismatch
            temp_path = create_folder(mismatch_path, f'{mismatch}_mismatch')
            mismatch_results = {}
            mismatch_guide_information = {}
            for guide, features in guide_dict.items():
                test_samples = next(iter(features.values()))[1]
                if sum(test_samples>0) == 0:
                    with open(error_file, "a") as f:
                        f.write(f"Guide {guide} has no positives for mismatch {mismatch}\n")
                    continue
                try:
                    guide_info = None
                    #guide_info = get_guide_information(data_name, guide,PATH_TO_STATISTICS_FILE)
                except:
                    guide_info = None
                mismatch_results[guide] = plot_evalutions_for_multiple_models(task= self.task, scores_dictionary=features,return_metrics=True)
                mismatch_guide_information[guide] = guide_info
            if mismatch_results: # Not empty
                #guides_sub_plots_classification(mismatch_results,mismatch_guide_information,output_path=temp_path,
                                               # generall_title='seperated')
                pass
            # merged guide dict into one test,scores,blablaal
        merged_guide_dict = merge_by_mismatches(guides_dict, error_file)
        for mismatch, features_dict in merged_guide_dict.items():
            merged_guide_dict[mismatch] = plot_evalutions_for_multiple_models(task= self.task, 
                                                                                    scores_dictionary=features_dict,return_metrics=True)
        #guides_sub_plots_classification(merged_guide_dict,information=None,
                                            #output_path=mismatch_path,generall_title='All guides')
    
    def _evaluate_test_by_guide(self,guides_dict, feature_dict, data_name, plots_path, if_save_last_tp_gap=True):
        guides_results = {}
        guide_informations = {}
        for guide, features in guides_dict.items(): # for each guide plots and evaluate all metrics
            
            try:
                guide_info =get_guide_information(data_name, guide,PATH_TO_STATISTICS_FILE)
                #guide_info = get_guide_information('Change_seq', guide,PATH_TO_STATISTICS_FILE)
            except:
                guide_info = None
            guides_results[guide] = plot_evalutions_for_multiple_models(task= self.task, scores_dictionary=features,return_metrics=True)
            guide_informations[guide] = guide_info
        
        all_guides_path = create_folder(plots_path,"All_guides")
        if if_save_last_tp_gap:
            guide_indexes_till_last_tp = get_ots_indexes_till_last_tp(guides_results, guides_dict)
            last_tp_index_path = create_folder(all_guides_path, 'last_tp_indexes')
            for guide_seq, indexes_array in guide_indexes_till_last_tp.items():
                np.save(os.path.join(last_tp_index_path, f'{guide_seq}.npy'), indexes_array)
        # sub plot all the guides separtley.
        #guides_sub_plots_classification(guides_results,guide_informations,output_path=all_guides_path,
                                        #generall_title='seperated')
        try:
            guide_lists = list(guides_dict.keys())
            guide_info = get_guide_information(data_name, guide_lists,PATH_TO_STATISTICS_FILE)
        except:
            guide_info = None
        
        print("Checking all guides")
        plot_evalutions_for_multiple_models(task= self.task, output_path=all_guides_path,plot_title="All_guides",
                                                results=None,scores_dictionary=feature_dict, information=guide_info)
    
    





def plot_evalutions_for_multiple_models(  task, output_path = None, plot_title = None, results = None,
                                         scores_dictionary = None, information=None, return_metrics = False):
    """
    This function iterates the scores dictionary and extract the evlaution for each model in the dict.
    Than plots the results of each model on the same plot.
    If return_metrics is True the function will return the metrics dictionary for each model.
    
    Args:
        task (str): The task of the model. (classification, regression)
        output_path (str): The path to save the plot.
        plot_title (str): The title for the plot.
        results (dict): The results of the models. If None the function will evaluate the models.
        scores_dictionary (dict): The scores dictionary with the scores for each model. model_name: predictions,tests,idx
        information (dict): The information of the models.
            i.e. dict with keys and values like positives:, negatives:, so on.
        return_metrics (bool): If True, return the metrics dictionary and model names.
    
    Returns:
        if return_metrics is True:
        metrics_dict (dict): The metrics dictionary for each model.
        {model: (fprs,tprs,aucs,percs,recalls,auprcs,n_ranks,last_fn_values)} for classification
        {model: (pearson_r, spearman_r, mse)} for regression"""
    # NOTE: NEED TO ARANAGE THIS PLOTTING.
    if scores_dictionary is None:
        raise RuntimeError("No scores dictionary was given")
    if results is None:
        pass # evaluate the models
    metrics_dict = get_metrics_by_task(task)
    model_names = []
    for model_name, scores in scores_dictionary.items():
        predictions, test, indexes = scores
        metrics_dict = append_values_metrics_by_task(test,predictions,metrics_dict,task)
        model_names.append(model_name)
    if return_metrics:
        return metrics_dict, model_names
    else:
        if information is None:
            if task.lower() == "classification":
                information = {'Positives': sum(test>0), 'Negatives': sum(test==0)}
            elif task.lower() == "regression":
                pass
        #plot_multiple_models_by_task(metrics_dict, model_names, task, output_path, plot_title, information)


def append_values_metrics_by_task(test,prediction,metrics_dict = None, task = None):
    '''This function appends the values of the test and prediction to the metrics dict.
    Args:
    1. test - the actual labels.
    2. prediction - the predicted scores.
    3. metrics_dict - the metrics dictionary to append the values to.
    4. task - the task of the model. (classification, regression)
    ------------
    Returns: the updated metrics dict.
    '''
    if task.lower() == "classification":
        return append_values_to_classification_metrics(test, prediction, metrics_dict)
    elif task.lower() == "regression":
        pass
    else:
        raise RuntimeError(f"Task: {task} is not supported")

def append_values_to_classification_metrics( test, predictions, metrics_dict = None):
    if metrics_dict is None:
        metrics_dict = init_classification_metrics()
    results, rates_dict = evaluate_classification(test, predictions, return_rates=True)
    auroc, auprc, n_rank_, last_fn_index, last_fn_ratio = results
    fpr,tpr,precision,recall = rates_dict.values()
    n_tpr = get_tpr_by_n_expriments(None,None,1000,tpr)
    n_rank = (n_rank_, n_tpr)
    fn_values = (last_fn_index,last_fn_ratio,tpr[:last_fn_index+1])

    metrics_dict["fprs"].append(fpr)
    metrics_dict["tprs"].append(tpr)
    metrics_dict["aucs"].append(auroc)
    metrics_dict["percs"].append(precision)
    metrics_dict["recalls"].append(recall)
    metrics_dict["auprcs"].append((auprc,np.sum(test[test > 0]) / len(test)))
    metrics_dict["n_ranks"].append(n_rank) 
    metrics_dict["last_fn_values"].append(fn_values)
    return metrics_dict
def init_classification_metrics():
    return {
        "fprs": [],
        "tprs": [],
        "aucs": [],
        "percs": [],
        "recalls": [],
        "auprcs": [],
        "n_ranks": [],
        "last_fn_values": []
    }
def init_regression_metrics():
    return {
        "pearsons": [],
        "spearman": [],
        "mses": []
    }  
def get_metrics_by_task(task):
    if task.lower() == "classification":
        return init_classification_metrics()
    elif task.lower() == "regression":
        return init_regression_metrics()
    else:
        raise RuntimeError(f"Task: {task} is not supported")



### METRICS HELPER FUNCTIONS ###
def get_percision_baseline(y_test):
    '''
    This function calculates the percision baseline for the given test values.
    Args:
    1. y_test - test values.
    ------------
    Returns: percision baseline. positives/
    '''
    total_positives = np.sum(y_test > 0)
    return total_positives / len(y_test)


