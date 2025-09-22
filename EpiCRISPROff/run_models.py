
FORCE_CPU = False
from features_engineering import  extract_features, get_guides_indexes

from models import get_cnn, get_logreg, get_xgboost, get_xgboost_cw, get_gru_emd, argmax_layer
from utilities import validate_dictionary_input, get_memory_usage
from parsing import features_method_dict, cross_val_dict, model_dict, class_weights_dict
from features_and_model_utilities import get_encoding_parameters
from train_and_test_utilities import split_to_train_and_val, split_by_guides
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler, SMOTE
import numpy as np
import time
import os
import signal
import tensorflow as tf


def tf_clean_up():
    tf.keras.backend.clear_session()
    print("GPU memory cleared.")
def set_signlas_clean_up():
    signal.signal(signal.SIGINT, tf_clean_up)
    signal.signal(signal.SIGSTOP, tf_clean_up)

class run_models:
    def __init__(self) -> None:
        self.ml_type = self.ml_name = self.ml_task = None
        
        self.shuffle = True
        self.if_os = False
        self.os_valid = False
        self.init_encoded_params = False
        self.init_booleans()
        self.init_model_dict()
        self.init_cross_val_dict()
        self.init_features_methods_dict()
        self.set_computation_power(FORCE_CPU)
        self.epigenetic_window_size = 0
        self.additional_epigenetic_features = 0
        self.additional_flanking_sequence_features = 0

    ## initairs ###
    # This functions are used to init necceseray parameters in order to run a model.
    # If the parameters are not setted before running the model, the program will raise an error.

    def set_computation_power(self, force_cpu=False):
        '''
        This function checks if there is an available GPU for computation.
        If GPUs are available, it enables memory growth. 
        If force_cpu is True, it forces the usage of CPU instead of GPU.
        '''
        if force_cpu:
            # Forcing CPU usage by hiding GPUs
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            self.gpu_available = False
            print("Forcing CPU computation, no GPU will be used.")
            return
        gpus = tf.config.list_physical_devices('GPU')  # Stable API for listing GPUs
        if gpus:
            try:
                # Enabling memory growth for each GPU
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')  # Logical devices are virtual GPUs
                print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
                gpu_available = True  # Indicating that GPU is available
                #set_signlas_clean_up()
            except RuntimeError as e:
                # Handle the error if memory growth cannot be set
                print(f"Error enabling memory growth: {e}")
                gpu_available = False
        else:
            gpu_available = False
            print("No GPU found. Using CPU.")
        self.gpu_available = gpu_available
    # def set_gpu(self, gpu_number):
    #     if not self.gpu_available:
    #         raise RuntimeError("No GPU available")
    #     if gpu_number < 0:
    #         raise ValueError("GPU number must be a non negative integer")
    #     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)
    def init_booleans(self):
        '''Features booleans'''
        self.if_only_seq = self.if_seperate_epi = self.if_bp = self.if_features_by_columns = False  
        self.method_init = False
    

    def init_deep_parameters(self, epochs = 5, batch_size = 1024, verbose = 2):
        '''Deep learning hyper parameters'''
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.hyper_params = {'epochs': self.epochs, 'batch_size': self.batch_size, 'verbose' : self.verbose, 'callbacks' : [], 'validation_data': None}# Add any other fit parameters you need
    
    def init_early_stoping(self):
        '''Early stopping for deep learning models'''
        
        if self.early:
            if self.paitence > 0:
                early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.paitence, restore_best_weights=True)
            else : early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.epochs, restore_best_weights=True)
            
        self.hyper_params.setdefault('callbacks', []).append(early_stopping)
   


    def init_model_dict(self):
        ''' Create a dictionary for ML models'''
        self.model_dict = model_dict()
        self.model_type_initiaded = False
    
    def init_cross_val_dict(self):
        ''' Create a dictionary for cross validation methods'''
        self.cross_val_dict = cross_val_dict()
        self.cross_val_init = False
    
    def init_features_methods_dict(self):
        ''' Create a dictionary for running methods'''
        self.features_methods_dict = features_method_dict()
        self.method_init = False
    
    def validate_initiation(self):
        if not self.model_type_initiaded:
            raise RuntimeError("Model type was not set")
        elif not self.method_init:
            raise RuntimeError("Method type was not set")
        elif not self.cross_val_init:
            raise RuntimeError("Cross validation type was not set")
        elif not self.os_valid:
            raise RuntimeError("Over sampling was not set")
        elif not self.ml_task:
            raise RuntimeError("ML task - classification/regression was not set")
        elif not self.init_encoded_params:
            raise RuntimeError("Encoded parameters were not set")
        

    def setup_runner(self, ml_task = None, model_num = None, cross_val = None, features_method = None, 
                     over_sampling = None, cw = None, encoding_type = None, if_bulges = None , early_stopping = None , deep_parameteres = None,
                     train = False, test =False, if_down_stream = False, downstream_length = 9):
        """
        This function sets the parameters for the model.

        Args:
            ml_task (str): classification or regression
            model_num (int): number of the model to use
            cross_val (int): cross validation method
            features_method (int): what feature included in the model
            over_sampling (str): if to use over sampling/downsampling
            cw (int): class weighting - 1- true, 2 - false
            encoding_type (int): encoding type for the model and features
            if_bulges (bool): if to include bulges in the encoding
            early_stopping (Tuple[bool, int]): if to use early stopping - Tuple (Bool, number of epochs)
            deep_parameteres (Tuple[int, int, int]): Tuple of epochs, batch size, verbose

        """
        self.set_model_task(ml_task)
        self.set_model(model_num, deep_parameteres)
        self.set_cross_validation(cross_val)
        self.set_features_method(features_method)
        self.set_encoding_parameters(encoding_type, if_bulges, if_down_stream,downstream_length)

        self.set_over_sampling('n') # set over sampling
        self.set_class_wieghting(cw)
        self.set_early_stopping(early_stopping[0],early_stopping[1]) if early_stopping else self.set_early_stopping()
        
        self.set_data_reproducibility(False) # set data, model reproducibility
        self.set_model_reproducibility(False)
        
        self.validate_initiation()
        self.init = True
    
    '''Set reproducibility for data and model'''
    def set_data_reproducibility(self, bool):
        self.data_reproducibility = bool
    def set_model_reproducibility(self, bool):
        self.model_reproducibility = bool
        if self.model_reproducibility:
            if self.ml_type == "DEEP":
                self.set_deep_seeds()
            else : 
                self.set_ml_seeds()
        else : # set randomness
            self.set_random_seeds(False)
    
    def set_class_wieghting(self, cw):
        answer = validate_dictionary_input(cw, class_weights_dict())
        if answer == 1:
            self.cw = True
        else : self.cw = False
    
    def set_early_stopping(self, if_early = 1,paitence = None):
        self.early = False
        self.paitence = 0
        if if_early == 1:
            self.early = True
            if paitence and paitence > 0:
                self.paitence = int(paitence)
            self.init_early_stoping()
    
    def set_encoding_parameters(self,enconding_type, if_bulges, if_downstream, downstream_length):
        self.guide_length, self.bp_presntation = get_encoding_parameters(enconding_type, if_bulges,if_downstream, downstream_length)
        if self.ml_name == "GRU-EMB":
            self.bp_presntation = self.bp_presntation**2
        self.encoded_length =  self.guide_length * self.bp_presntation
        self.init_encoded_params = True
    '''Set seeds for reproducibility'''
    def set_deep_seeds(self,seed=42):
        self.seed = seed
        tf.random.set_seed(seed) # Set seed for Python's random module (used by TensorFlow internally)
        tf.keras.utils.set_random_seed(seed)  # sets seeds for base-python, numpy and tf
        tf.config.experimental.enable_op_determinism()
    def set_ml_seeds(self):
        #np.random.seed(42) # set np seed
        self.random_state = 42
    def set_random_seeds(self,seed):
        loc_seed =int(time.time())
        if seed:
            loc_seed = seed
        tf.random.set_seed(loc_seed) # Set seed for Python's random module (used by TensorFlow internally)
        tf.keras.utils.set_random_seed(loc_seed)  # sets seeds for base-python, numpy and tf
        self.seed = loc_seed
    ## Features booleans setters
    def set_only_seq_booleans(self):
        self.if_bp = False
        self.if_only_seq = True
    
    def set_bp_in_seq_booleans(self):
        self.if_only_seq = False
        self.if_bp = True
        
    def set_epi_window_booleans(self):
        self.if_only_seq = self.if_bp = self.if_features_by_columns= False
        self.if_seperate_epi = True
    
    def set_features_by_columns_booleans(self):
        self.if_only_seq = self.if_bp = self.if_seperate_epi = False
        self.if_features_by_columns = True

    def get_model_booleans(self):
        '''Return the booleans for the model
        ----------
        Tuple of - only_seq, bp, seperate_epi, epi_features, data_reproducibility, model_reproducibility'''
        return self.if_only_seq, self.if_bp, self.if_seperate_epi, self.if_features_by_columns, self.data_reproducibility, self.model_reproducibility
   
    ## Model setters ###
    # This functions are used to set the model parameters.
    def set_hyper_params_class_wieghting(self, y_train):
        if self.ml_task == "Classification":
            if self.cw:
                class_weights = compute_class_weight(class_weight='balanced',classes= np.unique(y_train),y= y_train)
                class_weight_dict = dict(enumerate(class_weights))
                self.hyper_params['class_weight'] = class_weight_dict
        else :  return # no class wieghting for regression
    
    def set_model(self, model_num_answer = None , deep_parameters = None):
        if self.model_type_initiaded:
            return # model was already set
        if not self.ml_task:
            raise RuntimeError("Model task need to be setted before the model")
        model_num_answer = validate_dictionary_input(model_num_answer, self.model_dict)
        '''Given an answer - model number, set the ml_type and ml name'''
    
        if model_num_answer < 4: # ML models
            self.ml_type = "ML"
        else : # Deep models
            self.ml_type = "DEEP"
            self.init_deep_parameters(*deep_parameters) if deep_parameters else self.init_deep_parameters()
            
        self.ml_name = self.model_dict[model_num_answer]
        self.model_type_initiaded = True
            
    
    def set_model_task(self, task):
        '''This function set the model Task - classification or regression'''
        if self.ml_task:
            return # task was already set
        if task.lower() == "classification":
            self.ml_task = "Classification"
        elif task.lower() == "regression" or task.lower() == "t_regression":
            self.ml_task = "Regression"
        else : raise ValueError("Task must be classification or regression/t_regression")

    def set_cross_validation(self, cross_val_answer = None):
        if not self.model_type_initiaded:
            raise RuntimeError("Model type need to be setted before cross val type")
        if self.cross_val_init:
            return # cross val was already set
        ''' Set cross validation method and k value if needed.'''
        cross_val_answer = validate_dictionary_input(cross_val_answer, self.cross_val_dict)
        if cross_val_answer == 1:
            self.cross_validation_method = "Leave_one_out"
            self.k = ""
        elif cross_val_answer == 2:
            self.cross_validation_method = "K_cross"
            # self.k = int(input("Set K (int): "))
        elif cross_val_answer == 3:
            self.cross_validation_method = "Ensemble"
        self.cross_val_init = True
    
    def set_features_method(self, feature_method_answer = None):  
        if not self.model_type_initiaded and not self.cross_val_init:
            raise RuntimeError("Model type and cross val need to be setted before features method")
        if self.method_init:
            return # method was already set
        '''Set running method'''
        feature_method_answer = validate_dictionary_input(feature_method_answer, self.features_methods_dict)
        booleans_dict = {
            1: self.set_only_seq_booleans,
            2: self.set_features_by_columns_booleans,
            3: self.set_bp_in_seq_booleans,
            4: self.set_epi_window_booleans
            
        }   
        booleans_dict[feature_method_answer]()
        self.feature_type = self.features_methods_dict[feature_method_answer]
        self.method_init = True

    '''Set features columns for the model'''
    def set_additional_epigenetic_features(self, additional_features_num =0):
        if additional_features_num < 0:
            raise ValueError("Number of additional features must be a non negative integer")
        else :
            self.additional_epigenetic_features = additional_features_num
    
    def set_additional_flanking_features(self,additional_features_num =0):
        if additional_features_num < 0:
            raise ValueError("Number of additional features must be a non negative integer")
        else :
            self.additional_flanking_sequence_features = additional_features_num

    def set_big_wig_number(self, number):
        if isinstance(number,int) and number >= 0:
            self.bigwig_numer = number 
        else : raise ValueError("Number of bigwig files must be a non negative integer")
    def get_parameters_by_names(self):
        '''This function returns the following atributes by their names:
        Ml_name, Cross_validation_method, Features_type, epochs, batch_size'''
        return self.ml_name, self.cross_validation_method, self.feature_type, [self.epochs,self.batch_size], self.early
    def get_gpu_availability(self):
        return self.gpu_available
    ## Over sampling setter
    def set_over_sampling(self, over_sampling):
        if self.os_valid:
            return # over sampling was already set
        if not over_sampling:
            if_os = input("press y/Y to oversample, any other for more\n")
        else : if_os = over_sampling
        if if_os.lower() == "y":
            self.sampler = self.get_sampler('auto')
            self.if_os = True
            
        else: 
            self.sampler_type = ""
            self.sampler = None
        self.os_valid = True
    '''Tp are minority class, set the inverase ratio for xgb_cw
        args are 5 element tuple from get_tp_tn()'''
    def set_inverase_ratio(self, tps_tns):
        tprr, tp_test, tn_test, tp_train, tn_train = tps_tns # unpack tuple
        self.inverse_ratio = tn_train / tp_train

    ## Output setters: Features used, file name, model evalution and results table ##

    ## 1. File description based on booleans.
   
    def set_features_output_description(self):
        '''Create a feature description list'''
        if self.if_only_seq: # only seq
            self.features_description  = ["Only-Seq"]
        elif self.if_bp: # with base pair to gRNA bases or epigenetic window
            self.features_description = [file_name[0] for file_name in self.file_manager.get_bigwig_files()]
        elif self.if_seperate_epi: # window size epenetics
            self.features_description = [file_name[0] for file_name in self.file_manager.get_bigwig_files()]
            self.features_description.append(f'window_{self.epigenetic_window_size}')
        else : self.features_description = self.additional_epigenetic_features.copy() # features are added separtley
    
 
   
    
    

    ## Assistant RUN FUNCTIONS: DATA, MODEL, REGRESSION, CLASSIFICATION, 
    
    # OVER SAMPLING:
    '''1. Get the sampler instace with ratio and set the sampler string'''
    def get_sampler(self,balanced_ratio):
        sampler_type = input("1. over sampeling\n2. synthetic sampling\n")
        if sampler_type == "1": # over sampling
            self.sampler_type = "ROS"
            return RandomOverSampler(sampling_strategy=balanced_ratio, random_state=42)
        else : 
            self.sampler_type = "SMOTE"
            return SMOTE(sampling_strategy=balanced_ratio,random_state=42)

    
        
    ## MODELS:
    '''1. GET MODEL - from models.py'''
    def get_model(self):
        additional_features = self.additional_epigenetic_features + self.additional_flanking_sequence_features
        if self.ml_name == "LOGREG":
            return get_logreg(self.random_state, self.data_reproducibility)
        elif self.ml_name == "XGBOOST":
            return get_xgboost(self.random_state)
        elif self.ml_name == "XGBOOST_CW":
            return get_xgboost_cw(self.inverse_ratio, self.random_state,self.data_reproducibility)
        elif self.ml_name == "CNN":
            return get_cnn(self.guide_length, self.bp_presntation, self.if_only_seq, self.if_bp, 
                           self.if_seperate_epi, additional_features, self.epigenetic_window_size, self.bigwig_numer, self.ml_task)
        elif self.ml_name == "RNN":
            pass
        elif self.ml_name == "GRU-EMB":
            return get_gru_emd(task=self.ml_task,input_shape=(self.guide_length,self.bp_presntation),num_of_additional_features=additional_features,if_flatten=True)
    '''2. Training and Predicting with model:'''
    ## Train model: if Deep learning set class wieghting and extract features
    def train_model(self,X_train, y_train):
        if not self.init:
            raise RuntimeError("Trying to trian a model without a setup - please re run the code and use setup_runner function")
        # time_test(X_train,y_train)
        model = self.get_model()
        if self.ml_type == "DEEP":
            self.set_hyper_params_class_wieghting(y_train= y_train)
            if self.if_seperate_epi or (not (self.if_only_seq or self.if_bp)) or self.additional_flanking_sequence_features>0: 
                # if seperate epi/ only_seq=bp=false --> features added to seq encoding
                # extract featuers/epi window from sequence enconding 
                X_train = extract_features(X_train, self.encoded_length)
            if self.early: # split to train and val
                X_train,y_train,x_y_val = split_to_train_and_val(X_train,y_train,self.ml_task, seed=self.seed)
            self.hyper_params["validation_data"] = x_y_val
            model.fit(X_train,y_train,**self.hyper_params)
        else :
            model.fit(X_train,y_train)
        print(f"Memory Usage train model: {get_memory_usage():.2f} MB")
        tf.keras.backend.clear_session()
        return model
    
    def predict_with_model(self, model, X_test):
        if not self.init:
            raise RuntimeError("Trying to predict with a model without a setup - please re run the code and use setup_runner function")
        if self.ml_type == "DEEP":
            if self.if_seperate_epi or (not (self.if_only_seq or self.if_bp)) or self.additional_flanking_sequence_features>0:
                X_test = extract_features(X_test,self.encoded_length)
            y_pos_scores_probs = model.predict(X_test,verbose = 2,batch_size=self.hyper_params['batch_size'])
        else :
            y_scores_probs = model.predict_proba(X_test)
            y_pos_scores_probs = y_scores_probs[:,1]
        return y_pos_scores_probs

    ## RUNNERS: LEAVE ONE OUT, K-CROSS VALIDATION, ENSEMBLE
    
    
   
    

    
    ## ENSEMBLE:
    def create_ensemble(self, n_models, output_path, guides_train_list, seed_addition = 0, x_features=None, y_labels=None,guides=None):
        '''This function create ensemble of n_models and save them in the output path.
        The models train on the guide list given in guides_train_list.
        Each model created with diffrenet intitaion seed + a seed addition. This can be usefull to reproduce the model.
        Positive ratio is the ratio of positive labels in the training set, if None all the positive labels will be used.
        Args:
        1. n_models - number of models to create.
        2. output_path - path to save the models.
        3. guides_train_list - list of guides to train on.
        4. seed_addition - int to add to the seed for reproducibility.
        5. positive_ratio - list of ratios for positive labels in the training set.
        if positive ratio given, for each ratio a new folder will be created in the output path.
        6. X_train, y_train - if given will be used for training the models.
        ----------
        Saves: n trained models in output_path.
        Example: create_ensebmle(5,"/models",["ATT...TGG",...],seed_addition=10,positive_ratio=[0.5,0.7,0.9])'''
        
        for j in range(n_models):
            temp_path = os.path.join(output_path,f"model_{j+1}.keras")
            print(f'Creating model {j+1} out of {n_models}')
            self.create_model(output_path=temp_path,guides_train_list=guides_train_list,seed_addition=(j+1+seed_addition),x_features=x_features,y_labels=y_labels,guides=guides)

    
    def create_model(self, output_path, guides_train_list, seed_addition = 10, x_features=None, y_labels=None,guides=None,shared=False):
        if x_features is None or y_labels is None or guides is None:
            raise RuntimeError("Cannot create ensemble without data : x_features, y_labels, guides")
        else: 
           x_train,y_train,g_idx = split_by_guides(guides, guides_train_list, x_features, y_labels)
        self.set_deep_seeds(seed = seed_addition) # repro but random init (j+1 not 0)
        model = self.train_model(X_train=x_train,y_train=y_train)
        model.save(output_path)
    def test_ensmbel(self, ensembel_model_list, tested_guide_list,x_features=None, y_labels=None,guides=None):
        '''This function tests the models in the given ensmble.
        By defualt it test the models on the tested_guide_list, If test_on_guides is False:
        it will test on the guides that are not in the tested_guide_list
        Args:
        1. ensembel_model_list - list of paths to the models
        2. tested_guide_list - list of guides to test on
        3. test_on_guides - boolean to test on the given guides or on the diffrence guides'''
        # Get data
        if x_features is None:
            raise RuntimeError("Cannot test ensemble without x_features!")
        if y_labels is None or guides is None:
            y_test, all_guides_idx = None,None
            print('testing ensemble only with features!')
            x_test = x_features
            if isinstance(x_features,list):
                points = x_features[0].shape[0]
            else:
                points = x_features.shape[0]
            y_scores_probs = np.zeros(shape=(len(ensembel_model_list), points))
        else:
            x_test, y_test, guides_idx = split_by_guides(guides, tested_guide_list, x_features, y_labels)
            all_guides_idx = get_guides_indexes(guide_idxs=guides_idx) # get indexes of all grna,ots
            # init 2d array for y_scores 
            # Row - model, Column - probalities
            y_scores_probs = np.zeros(shape=(len(ensembel_model_list), len(y_test))) 
        for index,model_path in enumerate(ensembel_model_list): # iterate on models and predict y_scores
            model = tf.keras.models.load_model(model_path, custom_objects={'argmax_layer': argmax_layer})
            # self.set_random_seeds(seed = (index+1+additional_seed))
            model_predictions = self.predict_with_model(model=model,X_test=x_test).ravel() # predict and flatten to 1d
            y_scores_probs[index] = model_predictions
        return y_scores_probs, y_test, all_guides_idx
    
    def test_model(self, model_path, tested_guide_list,x_features=None, y_labels=None,guides=None):
        x_test, y_test, guides_idx = split_by_guides(guides, tested_guide_list, x_features, y_labels)
        all_guides_idx = get_guides_indexes(guide_idxs=guides_idx) # get indexes of all grna,ots
        #model = tf.keras.models.load_model(model_path, safe_mode=False)

        model = tf.keras.models.load_model(model_path, custom_objects={'argmax_layer': argmax_layer})
        model_predictions = self.predict_with_model(model=model,X_test=x_test).ravel() # predict and flatten to 1d
        return model_predictions, y_test, all_guides_idx
    
     

