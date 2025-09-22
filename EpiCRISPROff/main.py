
from multiprocessing import Pool
from typing import Tuple
from file_management import File_management
from file_utilities import create_paths, keep_only_folders
from evaluation import evaluation
from utilities import   get_memory_usage
from data_constraints_utilities import with_bulges
from k_groups_utilities import  get_k_groups_guides
from features_engineering import generate_features_and_labels, get_otss_labels_and_indexes
from features_and_model_utilities import get_features_columns_args_ensembles, parse_feature_column_dict, get_feature_column_suffix
from parsing import features_method_dict, cross_val_dict, model_dict,encoding_dict, early_stoping_dict
from parsing import class_weights_dict,off_target_constrians_dict, get_minimal_parser, parse_args, validate_main_args
from time_loging import log_time, save_log_time, set_time_log
from ensemble_utilities import ensemble_parmas, get_scores_combi_paths_for_ensemble 
from run_models import run_models
import os
import numpy as np
import sys
import time
import atexit
import traceback
import pickle

global ARGS, PHATS, COLUMNS, TRAIN, TEST, MULTI_PROCESS
DOWNSTREAM_SEQ = False



#NOTE: in setup_runner gave if downstream and # additional features line 160~. 

def set_args(argv):
    visible_args = ['--model', '--cross_val', '--features_method', '--features_columns', '--job', '--exclude_guides','--test_on_other_data']
    parser = get_minimal_parser(visible_args)
    args = parse_args(argv, parser)
    
    global ARGS, PHATS, COLUMNS
    ARGS, PHATS, COLUMNS = validate_main_args(args)
    time.sleep(2)
    print("Config files:\n")
    # print_dict_values(PHATS)
    # print_dict_values(COLUMNS)
    # time.sleep(1)

def set_multi_process(gpu_availability):
    '''
    This function set multi process to True if the gpu is NOT available.
    Other wise use the GPU without multiprocessing.
    '''
    global MULTI_PROCESS
    if not gpu_availability:
        MULTI_PROCESS = True
    else:
        MULTI_PROCESS = False

        

def set_cross_val_args(file_manager, train = False, test = False, cross_val_params = None, cross_val = None ):
    '''
    This function sets the cross validation arguments for the file manager.
    If no cross vaildation method given the fucntion will use the argument given to main.py
    For leave_one_out will set the parameters for file manager needed for leave one out
    For k_groups will set the parameters for file manager needed for k_groups
    For ensemble will set the parameters for file manager needed for ensemble amd return the t_guides, n_models, n_ensmbels
    '''
    if not cross_val:
        cross_val = ARGS.cross_val
    if not cross_val_params:
        cross_val_params = (ARGS.n_models, ARGS.n_ensmbels, ARGS.partition)
    if cross_val == 1: # leave_one_out
        pass
    elif cross_val == 2: # K_groups
        if ARGS.test_on_other_data: ### Testing on other data -> testing on all the guides in that data
            return None, None, None
        guide_path = file_manager.get_guides_partition_path(train,test)
        if len(ARGS.partition) == 1:
            partition_numbers = ARGS.partition[0]
            if isinstance(partition_numbers,str) and partition_numbers.lower() == 'all': # Cross val all partitions in the folder
                partitions_numbers = np.arange(1,len(os.listdir(guide_path))+1)
            elif isinstance(partition_numbers,int):
                partitions_numbers = np.arange(1,partition_numbers+1)
        elif isinstance(ARGS.partition,list): # Cross val specific partitions
            partitions_numbers = ARGS.partition
        t_guides = get_k_groups_guides(guide_path,partitions_numbers,train,test)
        return t_guides, None, None
    elif cross_val == 3: # Ensemble
        ens_parms = ensemble_parmas(*cross_val_params, ARGS.job)
        file_manager,n_models,n_ensmbels = ens_parms.set_file_manager_ensemble_params(file_manager, train, test)
        t_guides = file_manager.get_guides_partition()
        return t_guides, n_models, n_ensmbels




def set_job(args):
    train = test = process = evaluation = False
    if args.job.lower() == 'train':
        train = True
    elif args.job.lower() == 'test':
        test = True
    elif args.job.lower() == 'process':
        process = True
    elif args.job.lower() == 'evaluation':
        evaluation = True
    else: raise ValueError("Job must be either Train/Test/Evaluation/Process")
    return train, test, process, evaluation


 

### INITIARS ### 
def init_file_management(params=None, phats = None):
    '''This function creates a file management object with the given parameters.'''
    global PHATS
    if not phats: # None
        phats = PHATS
    file_manager = File_management(models_path=phats["Model_path"], ml_results_path=phats["ML_results_path"], 
                                   guides_path=phats["Guides_path"], vivo_silico_path=phats["Vivo-silico"], 
                                   vivo_vitro_path=phats["Vivo-vitro"], epigenetics_bed=phats["Epigenetic_folder"], 
                                   epigenetic_bigwig=phats["Big_wig_folder"], 
                                   partition_information_path=phats["Partition_information"], plots_path=phats["Plots_path"],
                                   job=ARGS.job)
    if not params: # None
        ml_name, cross_val, feature_type,epochs_batch,early_stop = model_dict()[ARGS.model], cross_val_dict()[ARGS.cross_val], features_method_dict()[ARGS.features_method], ARGS.deep_params, ARGS.early_stoping[0]
    else:
        ml_name, cross_val, feature_type,epochs_batch,early_stop = params
    early_stop = early_stoping_dict()[early_stop]
    cw,encoding_type,ots_constraints = class_weights_dict()[ARGS.class_weights], encoding_dict()[ARGS.encoding_type], off_target_constrians_dict()[ARGS.off_target_constriants]
    file_manager.set_model_parameters(data_type=ARGS.data_type, model_task=ARGS.task, cross_validation=cross_val, 
                                      model_name=ml_name, epoch_batch=epochs_batch,early_stop=early_stop,
                                      features=feature_type,class_weight=cw,encoding_type=encoding_type,
                                        ots_constriants=ots_constraints,transformation=ARGS.transformation,
                                        exclude_guides=ARGS.exclude_guides, test_on_other_data=ARGS.test_on_other_data, 
                                        with_downstream=ARGS.downstream)
    return file_manager

def init_model_runner(ml_task = None, model_num = None, cross_val = None, features_method = None):
    '''This function creates a run_models object with the given parameters.
    If no parameters are given, the function will use the default parameters passed in the arguments to main.py.
    --------- 
    Returns the run_models object.'''
    from run_models import tf_clean_up
    atexit.register(tf_clean_up)
    model_runner = run_models()
    if not ml_task: # Not none
        ml_task = ARGS.task
    if not model_num:
        model_num = ARGS.model
    if not cross_val:
        cross_val = ARGS.cross_val
    if not features_method:
        features_method = ARGS.features_method
    
    model_runner.setup_runner(ml_task=ml_task, model_num=model_num, cross_val=cross_val,
                               features_method=features_method,cw=ARGS.class_weights, encoding_type=ARGS.encoding_type,
                                 if_bulges=with_bulges(ARGS.off_target_constriants), early_stopping=ARGS.early_stoping
                                 ,deep_parameteres=ARGS.deep_params,if_down_stream=False)
    set_multi_process(model_runner.get_gpu_availability())
    if ARGS.downstream: # Add additional features
        additional_length = ARGS.downstream_length * 4
        model_runner.set_additional_flanking_features(additional_features_num=additional_length)
    return model_runner

def init_model_runner_file_manager(model_params = None) -> Tuple[run_models,File_management]:
    if model_params:
        model_runner = init_model_runner(*model_params)
    else :
        model_runner = init_model_runner()
    params = model_runner.get_parameters_by_names()
    file_manager = init_file_management(params)
    model_runner.set_big_wig_number(file_manager.get_number_of_bigiwig())
    
    return model_runner, file_manager

def init_run(model_params = None,  cross_val_params = None):
    '''This function inits the model runner and file manager.
    It sets the cross validation arguments for the file_manager'''
    model_runner, file_manager = init_model_runner_file_manager(model_params)
    train,test,process,evaluation = set_job(ARGS)
    t_guides, ensembles, models = set_cross_val_args(file_manager, train, test, cross_val_params)
    x_features, y_features, all_guides = get_x_y_data(file_manager, model_runner.get_model_booleans())
    return model_runner, file_manager, x_features, y_features, all_guides, t_guides, ensembles, models

def get_x_y_data(file_manager, model_runner_booleans, features_columns = None):
    '''Given a file manager, model runner booleans and model runner codings
    The function will generate the features and labels for the model used by the path to data in the file manager.
    The function will return the x_features, y_features and the guide set.''' 
    
    if_only_seq, if_bp, if_seperate_epi, if_features_by_columns, data_reproducibility, model_repro = model_runner_booleans
    if not features_columns: # if feature columns not given set for the defualt columns
        #### CHANGE to validate column with booleans
        features_columns = ARGS.features_columns
    log_time(f"Features_generation_{encoding_dict()[ARGS.encoding_type]}_start")
    x,y,guides=  generate_features_and_labels(file_manager.get_merged_data_path() , file_manager,
                                         if_bp, if_only_seq, if_seperate_epi,
                                         ARGS.epigenetic_window_size, features_columns, 
                                         data_reproducibility,COLUMNS, ARGS.transformation.lower(),
                                         sequence_coding_type=ARGS.encoding_type, if_bulges= with_bulges(ARGS.off_target_constriants),
                                         exclude_guides = ARGS.exclude_guides,test_on_other_data=file_manager.get_train_on_other_data(),
                                         if_downstream=ARGS.downstream,downstream_length=ARGS.downstream_length, downstream_in_seq = DOWNSTREAM_SEQ)
    print(f"Memory Usage features: {get_memory_usage():.2f} MB")
    log_time(f"Features_generation_{encoding_dict()[ARGS.encoding_type]}_end")
    return x,y,guides
################################################################################


def run():
    
    set_args(sys.argv)
    

    global ARGS
    log_time("Main_Run_start")
    train,test,process,evaluation = set_job(ARGS)
    cross_val_dict = cross_val()
    cross_val_dict[ARGS.cross_val](train,test,process,evaluation,ARGS.features_method)
    log_time("Main_Run_end")

def cross_val():
    function_dict = {
        
        2: run_k_groups,
        3: run_ensemble,
        
    }
    return function_dict
def run_ensemble(train = False, test = False,process=False,evaluation=False, method = None):
    if train:
        train_dict = train_ensemble()
        train_dict[method]()
    elif test:
        test_dict = test_ensemble()
        test_dict[method]()
    
    elif evaluation:
        if ARGS.test_on_other_data:
            evaluate_ensemble_by_guides_in_other_data()
        else:
            raise ValueError("Trying to evalaute ensemble on the training data, please give an --test_on_other_data argument")
    else: 
        raise ValueError("Job must be either Train or Test")
   
def train_ensemble():
    return  {
        1: create_ensmble_only_seq,
        2: create_ensembels_by_all_feature_columns,
        
    }
    
def test_ensemble():
    return {
        1: test_ensemble_via_onlyseq_feature,
        2: test_ensemble_by_features,
    }



def evaluate_ensemble_by_guides_in_other_data():
    file_manager = init_file_management()
    ml_results_path = file_manager.get_ml_results_path()
    scores_combi_paths = get_scores_combi_paths_for_ensemble(ml_results_path,ARGS.n_ensmbels,ARGS.n_models,True)
    if not scores_combi_paths:
        raise RuntimeError("No scores found in the ml_results_path")
    
    eval_obj = evaluation(ARGS.task)
    by_mismatch = False
    # guide_indexes = keep_indexes_per_guide(data_frame=file_manager.get_merged_data_path(), target_column=COLUMNS["TARGET_COLUMN"],
    #                                        ot_constrain=ARGS.off_target_constriants, mismatch_column=COLUMNS["MISMATCH_COLUMN"],
    #                                        bulges_column=COLUMNS["BULGES_COLUMN"], by_mismatch=by_mismatch)
    additional_data = None
    y_indexes, y_test = get_otss_labels_and_indexes(data_frame=file_manager.get_merged_data_path(), ot_constrain=ARGS.off_target_constriants,
                                                     mismatch_column=COLUMNS["MISMATCH_COLUMN"],
                                           bulges_column=COLUMNS["BULGES_COLUMN"], label_column= COLUMNS["Y_LABEL_COLUMN"])
    
    eval_obj.evaluate_test_per_guide(scores_combi_paths,ARGS.n_ensmbels,y_test,y_indexes,None,
                                     file_manager.get_plots_path(),ARGS.data_name,
                                     additional_data,by_mismatch=by_mismatch)


#####################################################################
   
def run_k_groups(train = False, test = False,process=False,evaluation=False, method = None):
    if train:
        train_dict = train_k_groups()
        train_dict[method]()
    elif test:
        test_dict = test_k_groups()
        test_dict[method]()
    elif evaluation:
        evaluate_k_groups()

def train_k_groups():
    return {
        1: train_k_groups_only_seq,
        2: train_k_groups_by_features,
    }
def test_k_groups():
    return {
        1: test_k_groups_only_seq,
        2: test_k_groups_by_features,
    }
def evaluate_k_groups():
    file_manager = init_file_management()
    ml_results_path = file_manager.get_ml_results_path()
    plots_path = file_manager.get_plots_path()
    evaluation_obj = evaluation(ARGS.task)
    evaluation_obj.evaluate_k_cross_results(ml_results_path,plots_path,save_results=True, evaluate_single_partition=False)


def train_k_groups_by_features(feature_dict = None, all_epigenetics = True):
    runner,file_manager = init_model_runner_file_manager()
    train_guides, n_models, n_ensmbels = set_cross_val_args(file_manager, train = True, test = False)
    model_base_path, ml_results_base_path = file_manager.get_model_path(), file_manager.get_ml_results_path()
    if not feature_dict: # None
        ### NOTE: ONLY EPIGENETICS IS SET TO TRUE!!!
        features_dict = parse_feature_column_dict(ARGS.features_columns, only_epigenetics=True)
    else:
        features_dict = feature_dict
    args = get_features_columns_args_ensembles(runner=runner,file_manager=file_manager,t_guides=train_guides,
                                               model_base_path=model_base_path,ml_results_base_path=ml_results_base_path,
                                               n_models=None,n_ensmbels=None,features_dict=features_dict,multi_process=False)
    for arg in args:
        group, feature,runner, file_manager,t_guides,model_base_path,ml_results_base_path, n_models, n_ensmbels,multi_process = arg
        log_time(f'Create_ensmbels_with_epigenetic_features_{group}_{feature}_start')
        print(f"Training K cross for group: {group} with feature: {feature}")
        temp_suffix = get_feature_column_suffix(group,feature) # set path to epigenetic data type - binary, by score, by enrichment.
        if not all_epigenetics: # Dont run all epigenetics
            if "All-epigenetics" in temp_suffix:
                return
        x_features,y_features,all_guides = get_x_y_data(file_manager, runner.get_model_booleans(),  feature)
        file_manager.set_models_path(model_base_path) # set model path
        file_manager.set_ml_results_path(ml_results_base_path) # set ml results path
        file_manager.add_type_to_models_paths(temp_suffix) # add path to train ensmbel
        runner: run_models
        runner.set_additional_epigenetic_features(len(feature)) # set feature   
        train_k_groups_only_seq(runner,file_manager,x_features,y_features,all_guides,t_guides)
def train_k_groups_only_seq(runner = None, file_manager = None, x_features= None,
                             y_features = None, all_guides=None, guides = None):
    '''
    This function trains a model for each partition.
    In total k models will be created with the suffix partition_number.keras
    '''
    if runner is None or file_manager is None:
        runner,file_manager = init_model_runner_file_manager()
    if guides is None:
        guides, n_models, n_ensmbels = set_cross_val_args(file_manager, train = True, test = False)
    if x_features is None or y_features is None:
        x_features, y_features, all_guides = get_x_y_data(file_manager, runner.get_model_booleans())
    models_path = file_manager.get_model_path()
    seed=10
    args = [(os.path.join(models_path, f"{partition}.keras"),guides_partition,seed,x_features,y_features,all_guides )for partition,guides_partition in guides.items()]
    if MULTI_PROCESS:
        processes = min(os.cpu_count(), len(ARGS.partition))
        with Pool(processes=processes) as pool:
            pool.starmap(runner.create_model, args)
    else:
        models = len(args)
        for m_,arg in enumerate(args):
            print(f'train model number: {m_+1}/{models}')
            runner.create_model(*arg)       
def test_k_groups_only_seq(runner = None, file_manager = None,
                            x_features= None, y_features = None, all_guides=None, 
                            guides = None, save_raw_scores = True ):
    """

    This functions tests every model in k_cross partition and calculate it evaluation metric.
    It uses evaluation object to evalute the model preformance.
    It can use evaluation object to plot the results.
    If return_raw_scores is True, the function will return the raw scores of the models.
    This is used by external function like different feature testing so one can use all the different
    features raw results and evalaute tham togther.

    Args:
        
        runner (object): The model runner object.
        file_manager (object): The file manager object.
        x_features (array): The features used for training.
        y_features (array): The labels used for training.
        all_guides (array): The guides used for training.
        guides (dict): The guides used for testing.
        save_raw_scores (bool): If True, the function will save the raw scores of the models.
    
    Returns:
        if_return_raw_scores (bool): If True, the function will return the raw scores of the models.
        scores_dictionary (dict): The scores of the models.
    NOTE: ADD ARGUMENTS FOR FEATURES AND MULTIPROCESSING
    NOTE: DEAL WITH PLOT PATH FROM FILE MANAGER AS IT NOT SET FOR TESTING!
    """
    if runner is None or file_manager is None:
        runner,file_manager = init_model_runner_file_manager()
    if guides is None:
        guides, n_models, n_ensmbels = set_cross_val_args(file_manager, train = False, test = True)
    if x_features is None or y_features is None:
        x_features, y_features, all_guides = get_x_y_data(file_manager, runner.get_model_booleans())
    
    models_path = file_manager.get_model_path()
    scores_dictionary = {}
    if ARGS.test_on_other_data: # Test on other data so 
        models = create_paths(models_path)
        for model in models:
            scores, test, idx = runner.test_model(model, guides, x_features, y_features, all_guides)
            scores_dictionary[model] = (scores,test,idx)
    else:
        partitions = len(guides)
        for partition,guides_in_partition in guides.items():
            print(f'Testing partition number: {partition}/{partitions}')
            temp_path = os.path.join(models_path, f"{partition}.keras")
            scores, test, idx = runner.test_model(temp_path, guides_in_partition, x_features, y_features, all_guides)
            scores_dictionary[partition] = (scores,test,idx)
    if save_raw_scores:
        ml_results = file_manager.get_ml_results_path()
        raw_scores_path = os.path.join(ml_results, "raw_scores.pkl")
        with open(raw_scores_path, 'wb') as f:
            pickle.dump(scores_dictionary, f)
        return
    
def test_k_groups_by_features(feature_dict = None, all_epigenetics = True):
    runner,file_manager = init_model_runner_file_manager()
    train_guides, n_models, n_ensmbels = set_cross_val_args(file_manager, train = False, test = True)
    model_base_path, ml_results_base_path = file_manager.get_model_path(), file_manager.get_ml_results_path()
    if not feature_dict: # None
        ### NOTE: ONLY EPIGENETICS IS SET TO TRUE!!!
        features_dict = parse_feature_column_dict(ARGS.features_columns, only_epigenetics=True)
    else:
        features_dict = feature_dict
    args = get_features_columns_args_ensembles(runner=runner,file_manager=file_manager,t_guides=train_guides,
                                               model_base_path=model_base_path,ml_results_base_path=ml_results_base_path,
                                               n_models=None,n_ensmbels=None,features_dict=features_dict,multi_process=False)
    epi_results = {}
    for arg in args:
        group, feature,runner, file_manager,t_guides,model_base_path,ml_results_base_path, n_models, n_ensmbels,multi_process = arg
        log_time(f'Create_ensmbels_with_epigenetic_features_{group}_{feature}_start')
        print(f"Testing K cross for group: {group} with feature: {feature}")
        temp_suffix = get_feature_column_suffix(group,feature) # set path to epigenetic data type - binary, by score, by enrichment.
        if not all_epigenetics: # Dont run all epigenetics
            if "All-epigenetics" in temp_suffix:
                return
        x_features,y_features,all_guides = get_x_y_data(file_manager, runner.get_model_booleans(),  feature)
        file_manager.set_models_path(model_base_path) # set model path
        file_manager.set_ml_results_path(ml_results_base_path) # set ml results path
        file_manager.add_type_to_models_paths(temp_suffix) # add path to train ensmbel
        runner: run_models

        runner.set_additional_epigenetic_features(len(feature)) # set feature  
        epi_results[temp_suffix.split('/')[1]] = test_k_groups_only_seq(runner=runner,file_manager=file_manager,
                               x_features=x_features,y_features=y_features,all_guides=all_guides,
                               guides=t_guides,save_raw_scores=True)       
    




## ENSMBEL
# Only sequence
def create_ensmble_only_seq(  model_params = None,cross_val_params=None,multi_process=True,group_dir = None,):
    '''The function will create an ensmbel with only sequence features'''
    log_time("Create_ensmble_only_seq_start")
    if not model_params and not cross_val_params: # None
        runner, file_manager , x_features, y_features, all_guides, guides, n_models, n_ensmbels = init_run()
    else:
        runner, file_manager , x_features, y_features, all_guides, guides, n_models, n_ensmbels = init_run(model_params, cross_val_params)
    if group_dir:
        file_manager.add_type_to_models_paths(group_dir)
    if n_ensmbels == 1:
        create_n_ensembles(n_ensmbels, n_models, guides, file_manager, runner, x_features, y_features, all_guides)
    else: # more then 1 ensembles.
        create_n_ensembles(n_ensmbels, n_models, guides, file_manager, runner, x_features, y_features, all_guides, multi_process=True)
    log_time("Create_ensmble_only_seq_end")
    del x_features, y_features
## - EPIGENETICS:
    ## 1. Creation
def create_ensembels_by_all_feature_columns(model_params = None,cross_val_params=None, multi_process = True, feature_dict = None):
    '''
    This function trains ensembles for each epigenetic feature column.
    The function splits the columns them into groups of the epigenetic value estimation i.e binary, score, enrichment.
    The function will create ensmbels for each group and for each feature in the group.
    NOTE: multiprocess each feature when n_ensmbels = 1. if n_ensembles > 1 the subfunction will multiprocess the ensmbels.
    ARGS:
    1. model_params: tuple - model parameters for the runner
    2. cross_val_params: tuple - cross val parameters for the file manager
    3. multi_process: bool - if True the function will multiprocess the features in the group.'''
    if not feature_dict: # None
        features_dict = parse_feature_column_dict(ARGS.features_columns, only_epigenetics=True)
    else:
        features_dict = feature_dict
    arg_list = []
    if not model_params and not cross_val_params: # No paramaeteres are given.
        runner, file_manager = init_model_runner_file_manager()
        train_guides, n_models, n_ensmbels = set_cross_val_args(file_manager, train = True, test = False)
    else:
        runner, file_manager = init_model_runner_file_manager(model_params)
        train_guides, n_models, n_ensmbels = set_cross_val_args(file_manager, train = True, test = False, cross_val_params = cross_val_params)
    model_base_path, ml_results_base_path = file_manager.get_model_path(), file_manager.get_ml_results_path()
    
    if n_ensmbels == 1 and multi_process and MULTI_PROCESS:  # multiprocess bool for activating this function from another function.
        arg_list = get_features_columns_args_ensembles(runner= runner, file_manager = file_manager, t_guides = train_guides, 
                                             model_base_path = model_base_path, ml_results_base_path = ml_results_base_path,
                                               n_models = n_models, n_ensmbels = n_ensmbels, features_dict = features_dict, multi_process = False)
        with Pool(processes=10) as pool:
            pool.starmap(create_ensembels_for_a_given_feature, arg_list) 
    
    else: # multi_process = True/False and n_ensmbels > 1
        arg_list = get_features_columns_args_ensembles(runner= runner, file_manager = file_manager, t_guides = train_guides, 
                                             model_base_path = model_base_path, ml_results_base_path = ml_results_base_path,
                                               n_models = n_models, n_ensmbels = n_ensmbels, features_dict = features_dict, multi_process = multi_process)
        for args in arg_list:
            create_ensembels_for_a_given_feature(*args)
            



def create_ensembels_for_a_given_feature(group, feature,runner:run_models, file_manager,train_guides,model_base_path,ml_results_base_path,
                                         n_models=50, n_ensmbels=10, multi_process = False,all_epigenetics = False):
    '''This function create a ensemble for a given group of features and A feature in that group by utilizing the create_n_ensembles function.
    It extracts the x_features, y_features and all_guides from the file manager given the specific feature and create the ensembles.
    ARGS:
    1. group: str - group name
    2. feature: list of one feature
    3. runner: run_models object
    4. file_manager: file_manager object
    5. train_guides: list of guides to train the model on
    6. model_base_path: str - model path before the group and feature
    7. ml_results_base_path: str - ml results path " " " " " " " " ..
    8. n_models: int - number of models in each ensemble
    9. n_ensmbels: int - number of ensembles
    10. multi_process: bool - passed to create_n_ensembles function to multiprocess the ensembles.
    11. all_epigenetics: bool - default True, if False the function will not use all epigenetic features in the group.
    '''
    log_time(f'Create_ensmbels_with_epigenetic_features_{group}_{feature}_start')
    temp_suffix = get_feature_column_suffix(group,feature) # set path to epigenetic data type - binary, by score, by enrichment.
    if not all_epigenetics: # Dont run all epigenetics
        if "All-epigenetics" in temp_suffix:
            return
    x_features,y_features,all_guides = get_x_y_data(file_manager, runner.get_model_booleans(),  feature)
    file_manager.set_models_path(model_base_path) # set model path
    file_manager.set_ml_results_path(ml_results_base_path) # set ml results path
    file_manager.add_type_to_models_paths(temp_suffix) # add path to train ensmbel
    
    
    runner.set_additional_epigenetic_features(len(feature)) # set feature   
    create_n_ensembles(n_ensmbels, n_models, train_guides, file_manager, runner,x_features,y_features,all_guides, multi_process)
    log_time(f'Create_ensmbels_with_epigenetic_features_{group}_{feature}_end')
    # Delete data free memory
    del x_features, y_features
def create_n_ensembles(n_ensembles, n_models, guides, file_manager, runner, x_, y_, all_guides, multi_process = False, start_from = 1): 
    '''This function creates n ensembles with n models for each ensemble.
    It will use the file manager to create train folders for each ensmbel.
    It will use the model runner to train the model in that folder.
    ARGS:
    1. n_ensembles: int - number of ensembles
    2. n_models: int - number of models in each ensemble
    3. guides: list of guides to train the model on
    4. file_manager: file_manager object
    5. runner: run_models object
    6. x_: np.array - features
    7. y_: np.array - labels
    8. all_guides: list of all guides in the data
    9. multi_process: bool - if True and the number of ensmebles is bigger than 1, the function will multiprocess the ensembles.
    10. start_from: (int) - default 2, the function will start from the given ensemble number so totaly will create n_ensebmles - startfrom ensembles.
    '''
    # Generate argument list for each ensemble
    if n_ensembles == 1:
        ensemble_args_list = [(n_models, file_manager.create_ensemble_train_folder(1), guides,(1*10),x_,y_,all_guides) ]
    elif n_ensembles >= start_from: # more than 1 ensemble to create. validate start from is lower than number of ensembles
        ensemble_args_list = [(n_models, file_manager.create_ensemble_train_folder(i), guides,(i*10),x_,y_,all_guides) for i in range(start_from, n_ensembles+1)]
    else:
        raise ValueError("Start from is bigger than number of ensembles")
    # Create_ensmbel accpets - n_models, output_path, guides, additional_seed for reproducibility
    if multi_process and n_ensembles > 1 and MULTI_PROCESS:
        # Create a pool of processes
        cpu_count = os.cpu_count()
        num_proceses = min(cpu_count, n_ensembles)
        
        with Pool(processes=num_proceses) as pool:
            pool.starmap(runner.create_ensemble, ensemble_args_list)
    else : 
        for args in ensemble_args_list:
            runner.create_ensemble(*args)


## 2. ENSMBEL SCORES/Predictions


def test_ensemble_via_onlyseq_feature(model_params = None,cross_val_params=None,multi_process=True,different_test_folder_path = None, different_test_path = None, group_dir = None):
    '''This function init a model runner, file manager and the x,y features and testing guide for an ensmeble.
    It will pass these arguments to test_ensmbel_scores function to test the ensmbel on the test guides.
    If more than 1 ensemble to check, the function will multiprocess if it wasnt activated from a multiprocess program.
    ARGS:
    1. model_params: tuple - model parameters for the runner
    2. cross_val_params: tuple - cross val parameters for the file manager
    3. multi_process: bool set to True, when another process subprocess this function it should set to False to avoid sub multiprocessing.'''
    if not model_params and not cross_val_params: # None
        runner, file_manager , x_features, y_features, all_guides, tested_guides, n_models, n_ensmbels = init_run()
    else :
        runner, file_manager , x_features, y_features, all_guides, tested_guides, n_models, n_ensmbels = init_run(model_params, cross_val_params)
    if different_test_folder_path: # Not None
        file_manager.set_ml_results_path(different_test_folder_path)
    if different_test_path:
        file_manager.set_seperate_test_data(different_test_path[0],different_test_path[1])    
    if group_dir:
        file_manager.add_type_to_models_paths(group_dir)
    score_path, combi_path = file_manager.create_ensemble_score_nd_combi_folder()
    ensmbels_paths = create_paths(file_manager.get_model_path())  # Create paths for each ensmbel in partition
    if ensmbels_paths is None or len(ensmbels_paths) == 0:
        raise ValueError(f"No ensembles found in the given path. Please check the model path.\n{file_manager.get_model_path()}")
    ensmbels_paths = keep_only_folders(ensmbels_paths)  # Keep only folders
    args = [(runner, ensmbel, tested_guides, score_path, x_features, y_features, all_guides) for ensmbel in ensmbels_paths]
    if n_ensmbels>1 and multi_process and MULTI_PROCESS: 
        with Pool(processes=10) as pool:
            pool.starmap(test_enmsbel_scores, args)
    else:
        for ensmbel in ensmbels_paths:
            test_enmsbel_scores(runner, ensmbel, tested_guides, score_path, x_features, y_features, all_guides)


def test_enmsbel_scores(runner, ensmbel_path, test_guides, score_path, x_features, y_labels, all_guides):
    '''Given a path to an ensmbel, a list of test guides and a score path
    the function will test the ensmbel on the test guides and save the scores in the score path.
    Each scores will be added with the acctual label and the index of the data point.'''
    
    print(f"Testing ensmbel {ensmbel_path}")
    models_path_list = create_paths(ensmbel_path)
    models_path_list.sort(key=lambda x: int(x.split(".")[-2].split("_")[-1]))  # Sort model paths by models number
    y_scores, y_test, test_indexes = runner.test_ensmbel(models_path_list, test_guides, x_features, y_labels, all_guides)
    # Save raw scores in score path
    temp_output_path = os.path.join(score_path,f'{ensmbel_path.split("/")[-1]}.pkl')
    y_scores = np.mean(y_scores,axis=0)
    sorted_indexes = np.argsort(test_indexes)
    y_scores = y_scores[sorted_indexes]
    with open(temp_output_path, 'wb') as f:
        pickle.dump(y_scores, f)
    # y_scores_with_test = add_labels_and_indexes_to_predictions(y_scores, y_test, test_indexes)
    #write_2d_array_to_csv(y_scores_with_test,temp_output_path,[])






def test_ensemble_by_features(model_params= None, cross_val_params= None, multi_process = True, features_dict = None):
    if not model_params and not cross_val_params: # None
        runner, file_manager  = init_model_runner_file_manager()
        t_guides, n_models, n_ensmbels = set_cross_val_args(file_manager, train = False, test = True)
    else : 
        runner, file_manager  = init_model_runner_file_manager(model_params)
        t_guides, n_models, n_ensmbels = set_cross_val_args(file_manager, train = False, test = True, cross_val_params = cross_val_params)
    if not features_dict: 
        features_dict = parse_feature_column_dict(ARGS.features_columns,only_epigenetics=True)
    else:
        features_dict = features_dict
    
    model_base_path, ml_results_base_path = file_manager.get_model_path(), file_manager.get_ml_results_path()
    arg_list = []
    if n_ensmbels == 1 and multi_process and MULTI_PROCESS: # multiprocess each feature
        arg_list = get_features_columns_args_ensembles(runner, file_manager, t_guides, model_base_path, ml_results_base_path, n_models, n_ensmbels, features_dict, multi_process = False)
        with Pool(processes=10) as pool:
            pool.starmap(test_ensemble_via_epi_feature_2, arg_list)
    else:
        arg_list = get_features_columns_args_ensembles(runner, file_manager, t_guides, model_base_path, ml_results_base_path, n_models, n_ensmbels, features_dict, multi_process = multi_process)
        for arg in arg_list:
            test_ensemble_via_epi_feature_2(*arg)
    
def test_ensemble_via_epi_feature_2(group, feature, runner : run_models, file_manager, t_guides, model_base_path, ml_results_base_path,n_models, n_ensmbels, multi_process):
    # NOTE: ALL EPIGENETIS!
    skip_all_epigenetics = True
    group_epi_path = get_feature_column_suffix(group,feature)
    if "All-epigenetics" in group_epi_path:
        if skip_all_epigenetics:
            return
    file_manager.set_models_path(model_base_path)
    file_manager.set_ml_results_path(ml_results_base_path)
    file_manager.add_type_to_models_paths(group_epi_path)
    
    runner.set_additional_epigenetic_features(len(feature))
    x_features, y_features, all_guides = get_x_y_data(file_manager, runner.get_model_booleans(),  feature)
    score_path, combi_path = file_manager.create_ensemble_score_nd_combi_folder() # Create score and combi folders
    ensmbels_paths = create_paths(file_manager.get_model_path())  # Create paths for each ensmbel in partition
    ensmbels_paths = keep_only_folders(ensmbels_paths)  # Keep only folders
    args = [(runner, ensmbel, t_guides, score_path,x_features,y_features,all_guides) for ensmbel in ensmbels_paths]
    if multi_process and MULTI_PROCESS: # can run multiprocess
        with Pool(processes=10) as pool:
            pool.starmap(test_enmsbel_scores, args)
    else: 
        for arg in args:
            test_enmsbel_scores(*arg)
    del x_features, y_features






if __name__ == "__main__":
    set_time_log(keep_time=True,time_logs_paths="Time_logs")
    try:
        run()
    except Exception as e:
        print(e)
        traceback.print_exc()  # Print the full traceback

    save_log_time(ARGS)
    
    