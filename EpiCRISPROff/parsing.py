import argparse
import os
import json
PRINTS = ''
USER_ARGS = ['model','corss_val','features_method','features_columns','job','exclude_guides','test_on_other_data']
## NOTE: Remove 'ALL' when evaluating
def features_method_dict():
    '''A dictionary for the feature incorporate in the model.'''
    return {
            1: "Only_sequence",
            2: "With_features_by_columns",
            
            
        }

def cross_val_dict():
    '''A dictionary for the cross validation methods.'''
    return {
            
            2: "K_cross",
            3: "Ensemble"
        }

def model_dict():
    '''A dictionary for the models to use.'''
    return {
            1: "LOGREG",
            2: "XGBOOST",
            3: "XGBOOST_CW",
            4: "CNN",
            5: "RNN",
            6: "GRU-EMB"
        }

def encoding_dict():
    '''
    A dictionary for the sequence encoding types.
    '''
    return {
        1: "PiCrispr_encoding",
        2: "Full_encoding"
    }

def off_target_constrians_dict():
    '''
    A dictionary for the off-target constraints.
    '''
    return {
        1: "No_constraints",
        2: "Mismatch_only",
        3: "Bulges_only"
    }

def class_weights_dict():
    '''
    A dictionary for the class weights.
    '''
    return {
        1: "CW",
        2: "No_CW"
    }

def early_stoping_dict():
    '''
    A dictionary for the early stoping.
    '''
    return {
        1: "Early_stop",
        2: "No_early_stop"
    }
def get_minimal_parser(visible_args):
    full_parser = main_argparser()

    # New empty parser
    minimal_parser = argparse.ArgumentParser(description='''Python script to init a model and train it on off-target dataset.
                                     Different models, feature types, cross_validations and tasks can be created.
                                     ''')

    for action in full_parser._actions:
        if any(opt in visible_args for opt in action.option_strings):
            minimal_parser._add_action(action)
        else:
            # Clone action but suppress help
            action.help = argparse.SUPPRESS
            minimal_parser._add_action(action)
    
    return minimal_parser
def main_argparser():
    parser = argparse.ArgumentParser(description='''Python script to init a model and train it on off-target dataset.
                                     Different models, feature types, cross_validations and tasks can be created.
                                     ''',add_help=False)
    parser.add_argument('--model','-m', type=int, 
                        help='''Model number: 1 - Logistic regression, 2 - XGBoost,
                          3 - XGBoost with class weights, 4 - CNN - PiCRISPR, 5 - RNN, 6- GRU-EMB''',
                         required=True, default=4)
    parser.add_argument('--cross_val','-cv', type=int,
                         help='''Cross validation type: 1 - Leave one out, 
                         2 - K cross validation, 3 - Ensmbel, 4 - K cross with ensemble''',
                         required=True, default=1)
    parser.add_argument('--features_method','-fm', type=int,
                         help='''Features method: 1 - Only_sequence, 2 - With_features_by_columns, 
                         3 - Base_pair_epigenetics_in_Sequence, 4 - Spatial_epigenetics''', 
                        required=True, default = 1)
    parser.add_argument('--features_columns', '-fc', type=str,
                     help='Features columns - path to a dict with keys as feature type and values are the columns names', required=False)
    parser.add_argument('--epigenetic_window_size','-ew', type=int, 
                        help='Epigenetic window size - 100,200,500,2000', required=False, default=2000)
    parser.add_argument('--epigenetic_bigwig','-eb', type=str,
                         help='Path for epigenetic folder with bigwig files for each mark.', required=False)
    parser.add_argument('--task','-t', type=str, help='Task: Classification/Regression/T_Regression - T = Transformed', required=False, default='Classification')
    parser.add_argument('--transformation','-tr', type=str, help='Transformation type: Log/MinMax/Z_score', required=False, default=None)
    parser.add_argument('--job','-j', type=str, help='Job type: Train/Test/Evaluation/Process', required=True)
    parser.add_argument('--over_sampling','-os', type=str, help='Over sampling: y/n', required=False)
    parser.add_argument('--seed','-s', type=int, help='Seed for reproducibility', required=False)
    parser.add_argument('--data_reproducibility','-dr', type=str, help='Data reproducibility: y/n', required=False, default='n')
    parser.add_argument('--model_reproducibility','-mr', type=str, help='Model reproducibility: y/n', required=False, default='n')
    parser.add_argument('--config_file','-cfg', type=str, 
                        help='''Path to a json config file with the next dictionaries:
                        1. Data columns:
                        target_column, offtarget_column, chrom_column, start_column, end_column, binary_label_column,regression_label_column
                        2. Data paths:
                        Train_guides, Test_guides, Vivo-silico, Vivo-vitro, Model_path, ML_results, Data_name
                        ''',
                         required=False)
    parser.add_argument('--data_name','-jd', type=str,
                         help='''Dictionary names: 1 - Change_seq, 2 - Hendel, 3 - Hendel_Changeseq
                        The name of the data dict need to parse from the json file''', required=False)
    parser.add_argument('--data_type','-dt', type=str, help='''Data type: silico/vitro''', required=False)
    parser.add_argument('--partition','-p', type=str_or_int, nargs='+',help='Partition number given via list', required=False)
    parser.add_argument('--n_models','-nm', type=int, help='Number of models in each ensmbel', required=False)
    parser.add_argument('--n_ensmbels','-ne', type=int, help='Number of ensmbels', required=False)
    parser.add_argument('--encoding_type','-et', type=int, help='Sequence encoding type: 1 - PiCrispr, 2 - Full', default=1)
    parser.add_argument('--off_target_constriants','-otc', type=int, help='Off-target constraints: 1 - No_constraints, 2 - Mismatch_only, 3 - Bulges_only', default=1)
    parser.add_argument('--class_weights','-cw', type=int, help='Class weights: 1 - CW, 2 - No_CW', default=1)
    parser.add_argument('--deep_params','-dp', nargs='+',type=int, help='Deep learning parameters - epochs, batch', default=None)
    parser.add_argument('--early_stoping','-es', nargs='+',type=int, help='''Early stoping[0]: 1 - Early_stop, 2 - No_early_stop
                        Early stoping[1]: paitence size''', default=None)
    parser.add_argument('--exclude_guides','-eg', type=str, nargs='+', help='[List] Path to a data frame with sgRNAs, column/s to get.', default=None)
    parser.add_argument('--test_on_other_data','-tood', type=str, nargs= '+', help='''Test on other data:
                        [list] l[0] - path to json file (dictionary with keys as data names and values paths to data)
                        l[1] - data name to test on - should match the keys in the json file.''', default=None)
    parser.add_argument('--downstream', action='store_true', help='If given will add downstream sequences to the features')
    parser.add_argument('--downstream_length', type=int, default=9, help='Length of the downstream sequences to add, default is 9')
    
    
    # Set defualt arguments, not altered in EpiCRISPROff
    parser.set_defaults(
    epigenetic_window_size=2000,
    epigenetic_bigwig='/path/to/bigwig/folder',
    task='Classification',
    over_sampling='y',
    data_reproducibility='n',
    model_reproducibility='n',
    seed=42,
    config_file='Jsons/Data_columns_and_paths.json',
    data_name='Change_seq',
    data_type='vivo-silico',
    partition=['All'],
    n_ensmbels=10,
    n_models=50,
    encoding_type=2,
    off_target_constriants=1,
    class_weights=2,
    deep_params=[5, 1024],
    early_stoping=[1, 10],
    downstream=False,
    downstream_length=9
    
)
    
    return parser

def parse_args(argv,parser):
    
    if '--argfile' in argv:
        argfile_index = argv.index('--argfile') + 1
        argfile_path = argv[argfile_index]
     # Read the arguments from the file
        with open(argfile_path, 'r') as f:
            file_args = f.read().split()
        
        # Parse args with the file arguments included
            args = parser.parse_args(file_args)
    else:
    # Parse normally if no config file is provided
        args = parser.parse_args()    
    # Read the JSON file and load it as a dictionary
    return args

# Custom function to handle both int and str types
def str_or_int(value):
    try:
        # Attempt to convert to an integer
        return int(value)
    except ValueError:
        # If conversion fails, return the original string
        return value

def validate_main_args(args):
    if not os.path.exists(args.config_file):
        raise ValueError("Data columns config file does not exist") 
    if args.data_type not in ['vivo-silico','vivo-vitro','vitro-silico']:
        raise ValueError("Data type must be either vivo-silico/vivo-vitro/vitro-silico")
    if args.task.lower() not in ['classification','regression','t_regression','reg_classification']:
        raise ValueError("Task must be either Classification, Regression or T_Regression")
    if args.task.lower() == 't_regression' and (args.transformation is None or args.transformation.lower() not in ["log","minmax","z_score"]):
        raise ValueError("Transformation must be given for transformed regression or must be either Log, MinMax or Z_score")
    if args.job.lower() not in ['train','test','evaluation','process']:
        raise ValueError("Job must be either Train\Test\Evaluation")
    if args.deep_params is not None and len(args.deep_params) != 2:
        raise ValueError("Deep learning parameters must be given as a list of 2 integers - epochs and batch size")
    if args.early_stoping is not None and len(args.early_stoping) != 2:
        raise ValueError("Early stoping parameters must be given as a list of 2 integers - early stoping and patience")
    args.exclude_guides =  validate_exclude_guides(args.exclude_guides)
    args = validate_test_on_other_data(args)
    ## Print all args:
    print("The arguments are:")
    for arg, value in vars(args).items():
        if arg in USER_ARGS:
            print(f'{arg}: {value}') 
    with open(args.config_file, 'r') as f:
        print("Parsing config file")
        configs = json.load(f)
        data_columns = configs["Columns_dict"]
        data_configs = configs[args.data_name]
        args,data_columns = set_task_label(args, data_columns)
        args = set_method(args)
        print(PRINTS)
        return args, data_configs, data_columns
    
def set_test_on_other_data(test_on_other_data = None):
    '''
    Validate the test_on_other_data argument.
    Argument is [list] - [0] path to json file.
    [1] - key in the json file. 
    This function will return the tuple of (path_to_the_data, data_name) from the arg.
    '''
    if test_on_other_data is None:
        return None
    dictionary_datas_path = test_on_other_data[0]
    if not os.path.exists(dictionary_datas_path):
        raise ValueError(f"Test on other data dictionary dont exists: {dictionary_datas_path}")
    with open(dictionary_datas_path, 'r') as f:
        data_dict = json.load(f)
    data_name = test_on_other_data[1]
    data_path = data_dict[data_name]
    if not os.path.exists(data_path):
        raise ValueError(f"Data path does not exist: {data_path}")
    return (data_path,data_name)

def validate_test_on_other_data(args):
    '''
    This function validates that if job is test and partitions is all then test_on_other_data must be given.'''
    global PRINTS
    if isinstance(args.partition[0],str):
        if args.job.lower() == "train":
            args.test_on_other_data = None
        elif args.partition[0].lower() == 'all': # job is test/evaluation/process
            
        # other data should be given
            try:
                args.test_on_other_data = set_test_on_other_data(args.test_on_other_data)
            except ValueError as e:
                
                    
                print(e)
                raise ValueError("Test on other data must be given for partition == all and job == test")
            if args.test_on_other_data is None:
                if args.cross_val ==2:
                    PRINTS = PRINTS + "CROSS VALIDATION IS K-CROSS, PARTITION IS 'ALL' AND JOB IS TEST.\nOTHER TESTING DATA HAS NOT BEEN GIVEN TESTING ALL THE PARTITIONS IN THE PARTITIONS FOLDER!\n"
                    return args
                raise ValueError("Test on other data must be given for partition == all and job == test")
        else : # partition is not all/ job is not test
            args.test_on_other_data = None # set to None
    else: # partition is int
        args.test_on_other_data = None
    return args

def validate_exclude_guides(exclude_guides = None):
    '''
    Validate the exclude_guides- List of [path, column...,...]
    This function will return the tuple of (guide_dict_name, data_path, data_column) from the arg.
    '''
    if exclude_guides is None:
        return None
    data_path = exclude_guides[0]
    if not os.path.exists(data_path):
        raise ValueError("Data path does not exist")
    data_columns = exclude_guides[1:]
    sgrna_description = "_".join(data_columns)
    return (sgrna_description, data_path, data_columns)
    
def set_task_label(args, columns):
    '''
    This function will set the Y_LABEL_COLUMN in the columns dict according to the task.
    '''
    if args.task.lower() == 'classification':
        columns["Y_LABEL_COLUMN"] = columns["BINARY_LABEL_COLUMN"]
        args.transformation = ""
    else: 
        columns["Y_LABEL_COLUMN"] = columns["REGRESSION_LABEL_COLUMN"]  
    return args, columns

def set_method(args):
    if args.features_method == 2 and args.features_columns is None:
        raise ValueError("Features columns must be given for features by columns method")
    else:
        try:
            with open(args.features_columns, 'r') as f: # Read the features columns dict json file
                args.features_columns = json.load(f)
        except:
            raise ValueError("Features columns must be json file")
    if args.features_method == 3 and args.epigenetic_bigwig is None:
        raise ValueError("Epigenetic bigwig folder must be given for base pair epigenetics in sequence")
    if args.features_method == 4 and (args.epigenetic_bigwig is None or args.epigenetic_window_size is None):
        raise ValueError("Epigenetic bigwig folder and epigenetic window size must be given for spatial epigenetics")
    
    return args


