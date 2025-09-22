'''
The file_management class validates the following:
PARTITIONS: 
In the ensemble and k fold cross validation models the model needs to train on partitions.
Related functions: self.add_partition_path, set_partition, get_guides_partition.

OFF-Target data:
This module validates the off-target data paths and the type of data.
Type: vivo-silico, vivo-vitro, vitro-silico.
Type related functions: set_silico_vitro_bools, add_data_type_toguides_path, set_data_set_path.

Job type: test or train.

Epigenetic data:
.....

'''


import os

#import pyBigWig
from utilities import validate_non_negative_int
from k_groups_utilities import create_guides_list
from file_utilities import create_paths, get_ending
#import pybedtools

class File_management:
    # Positive and negative are files paths, pigenetics_bed and bigwig are folders path
    def __init__(self, models_path = None , ml_results_path = None, guides_path = None,
                    epigenetics_bed = None , epigenetic_bigwig = None,
                    vivo_silico_path = None, vivo_vitro_path = None,vitro_silico_path=None, partition_information_path = None,
                    plots_path = None, job = None) -> None: 
        self.set_paths = False
        if job:
            self.job = job.lower()
            if self.job == 'interpertation':
                return
        else:
            raise Exception("Job type not set")
        self.set_all_paths(models = models_path, ml_results = ml_results_path, guides_path = guides_path,
                            vivo_silico = vivo_silico_path, vivo_vitro = vivo_vitro_path,
                              vitro_silico=vitro_silico_path, epi_folder = epigenetics_bed, bigiw_folder = epigenetic_bigwig,
                             partition_information_path=partition_information_path, plots_path = plots_path)
        
        
        
    ## Getters:
    def get_positive_path(self):
        return self.positive_path
    def get_negative_path(self):
        return self.negative_path
    def get_merged_data_path(self):
        if self.merged_data_path:
            return self.merged_data_path
        elif self.vivo_silico_path is None and self.vivo_vitro_path is None:
            raise RuntimeError("No merged data path set")
        else:
            raise RuntimeError("No silico/vitro bools data path set")
    def get_plots_path(self):
        return self.plots_path

    def get_epigenetics_folder(self):
        return self.bed_folder_path
    def get_bigwig_folder(self):
        return self.bigwig_folder_path
    def get_number_of_bigiwig(self):
        return 0
        return self.bigwig_amount
    def get_model_path(self):
        return self.models_path
    def get_ml_results_path(self):
        return self.ml_results_path
    def get_number_of_bed_files(self):
        return self.bed_files_amount
    def get_ensmbel_path(self):
        return self.ensmbel_train_path
    def get_bigwig_files(self):
        if self.bigwig_amount > 0 :
            return self.bigwig_files.copy() # keep original list
        else: raise RuntimeError("No bigwig files")
    def get_bed_files(self):
        if self.bed_files_amount > 0 :
            return self.bed_files.copy()
        else: raise RuntimeError("No bedfiles setted")
    def get_global_max_bw(self):
        if len(self.glb_max_dict) > 0 :
            return self.glb_max_dict
        else : raise RuntimeError("No max values setted for bigwig files")
    
    def get_partition_information_path(self):
        return self.partition_information_path
    def get_guides_partition_path(self, train = False, test = False):
        
        if train:
            if not "Train" in self.guides_partition_path:
                self.add_data_type_toguides_path("Train_guides")
        elif test:
            if not "Test" in self.guides_partition_path:
                self.add_data_type_toguides_path("Test_guides")
        else: raise RuntimeError("No test/train flag given")
        return self.guides_partition_path
    def get_train_on_other_data(self):
        return self.other_test_data_initiated
    ## Setters:
    def set_all_paths(self, models, ml_results, guides_path, vivo_silico, vivo_vitro,vitro_silico, epi_folder, bigiw_folder,partition_information_path, plots_path):
        self.set_models_path(models)
        self.set_ml_results_path(ml_results)
        self.set_guide_paths(guides_path)
        
        # self.set_epigenetic_paths(epi_folder, bigiw_folder)
        self.set_data_set_path(vivo_silico, vivo_vitro, vitro_silico)
        #self.set_partition_information_path(partition_information_path)
        self.set_plot_path(plots_path)
        self.other_test_data_initiated = False
        self.set_paths = True

    def set_models_path(self, models_path):
        self.validate_path_exsits(models_path)
        self.models_path = models_path
    def set_ml_results_path(self, ml_results_path):
        self.validate_path_exsits(ml_results_path)
        self.ml_results_path = ml_results_path
    def set_guide_paths(self, guide_path):
        '''
        Guides path is a path to folder containing guide for each partition'''
        self.validate_path_exsits(guide_path)
        self.guides_partition_path = guide_path
    
    def add_data_type_toguides_path(self, data_type):
        '''
        This function adds the data type to the training/testing guides path.
        The orignal path is the guides_partition_path.

        Job: adding Train or Test the path. - diffrenet guides for training and testing.
        Type: adding vivo or vitro to the path. - different data types may hold different guides partitions.
        '''
        self.guides_partition_path = os.path.join(self.guides_partition_path,data_type)
        self.validate_path_exsits(self.guides_partition_path)

    def set_bigwig_folder_path(self, bigwig_folder_path):
        self.validate_path_exsits(bigwig_folder_path)
        self.bigwig_folder_path = bigwig_folder_path
    
    def set_bed_folder_path(self, bed_folder_path):
        self.validate_path_exsits(bed_folder_path)
        self.bed_folder_path = bed_folder_path

    def set_epigenetic_paths(self, epigenetics_bed, bigwig):
        self.validate_path_exsits(epigenetics_bed)
        self.validate_path_exsits(bigwig)
        self.bed_folder_path = epigenetics_bed
        self.bigwig_folder_path = bigwig
        #self.create_bigwig_files_objects()
        #self.set_global_bw_max()
        #self.create_bed_files_objects()
    def set_partition_information_path(self, partition_information_path):
        self.validate_path_exsits(partition_information_path)
        self.partition_information_path = partition_information_path
    
    def set_plot_path(self, plots_path):
        self.validate_path_exsits(plots_path)
        self.plots_path = plots_path
    ## Data paths - silico/vitro negative OTSs ##
    '''These functions dont validate the path, a dataset can have only one of the paths.
    When choosing to use one of the datasets then the path will be validated.'''
    def set_data_set_path(self, vivo_silico_path, vivo_vitro_path, vitro_silico_path):
        self.vivo_silico_path = vivo_silico_path
        self.vivo_vitro_path = vivo_vitro_path
        self.vitro_silico_path = vitro_silico_path


    def set_model_parameters(self,data_type, model_task, cross_validation, model_name,epoch_batch,early_stop, 
                             features,class_weight,encoding_type,ots_constriants, transformation = None,
                               exclude_guides = None, test_on_other_data = None, with_downstream = False):
        '''
        
        This function sets the model parameters to save the model in the corresponding path.
        
        Path: .../Models/{data_type}/{model_task}/{cross_validation}/{model_name}/{features}
              .../Ml_results/{data_type}/{model_task}/{cross_validation}/{model_name}/{features}
        '''
        if data_type == "vivo-silico":
            self.set_silico_vitro_bools(silico_bool = True)
        #     self.add_data_type_toguides_path("vivo")
        # elif data_type == "vivo-vitro":
        #     self.set_silico_vitro_bools(vitro_bool = True)
        #     self.add_data_type_toguides_path("vivo")
        # elif data_type == "vitro-silico":
        #     self.set_silico_vitro_bools(vitro_bool = True, silico_bool = True)
        #     self.add_data_type_toguides_path("vitro")
        # else:
        #     raise RuntimeError("Data type not set")
        # if model_task.lower() == "reg_classification":
        #     model_task = "T_Regression"
        #     plots_model_task = "Reg_classification"
        # else:
        #     plots_model_task = ""
        # epoch_batch = f'{epoch_batch[0]}epochs_{epoch_batch[1]}_batch'
        self.add_exlucde_guides(exclude_guides)
        self.set_seperate_test_data(test_on_other_data)
        # if transformation:
            
        #     full_path = os.path.join(model_task,transformation,ots_constriants,encoding_type,class_weight,model_name,epoch_batch,early_stop,cross_validation,features)
        #     plots_path = os.path.join(plots_model_task,model_task,transformation,ots_constriants,encoding_type,class_weight,model_name,epoch_batch,early_stop,cross_validation)
        plots_addition = ""
        if with_downstream:
            features = f'{features}_with_downstream'
            plots_addition = "withDownstream"
        full_path = os.path.join(model_name,cross_validation,features)
        plots_path = os.path.join(model_name,cross_validation,plots_addition)
        self.add_type_to_models_paths(full_path)
        self.add_type_to_plots_path(plots_path)        
        self.task = model_task
        
    def add_exlucde_guides(self, exclude_guides = None):
        if exclude_guides:
            exclude_guides = "Exclude_" + exclude_guides[0]
            self.add_type_to_models_paths(exclude_guides)
            self.add_type_to_plots_path(exclude_guides)
              


    def set_silico_vitro_bools(self, silico_bool = False, vitro_bool = False):
        '''This function sets the merged data path to the vivo_silico_path or vivo_vitro_path.
        As well adds to model and model results path the vivo-silico/vivo-vitro suffix.
        Args:
        1. silico_bool - bool, default False
        2. vitro_bool - bool, default False
        -----------
        Returns: Error if both bools are false'''
        suffix_str = ""
        if silico_bool and vitro_bool: # vitro-silico
            self.validate_path_exsits(self.vitro_silico_path)
            self.merged_data_path = self.vitro_silico_path
            suffix_str = "vitro-silico"
        elif silico_bool:
            self.validate_path_exsits(self.vivo_silico_path)
            self.merged_data_path = self.vivo_silico_path
            suffix_str = "vivo-silico"
        elif vitro_bool:
            self.validate_path_exsits(self.vivo_vitro_path)
            self.merged_data_path = self.vivo_vitro_path
            suffix_str = "vivo-vitro"
        
        # else:
        #     raise RuntimeError("No silico vitro bools were given data path set")
        # if (suffix_str not in self.models_path) and (suffix_str not in self.ml_results_path):
        #     self.add_type_to_models_paths(suffix_str)
        # if (suffix_str not in self.plots_path):
        #     self.add_type_to_plots_path(suffix_str)
        # else:
        #     raise Exception(f"Suffix {suffix_str} already in model or results paths:\n {self.models_path}\n{self.ml_results_path}")
    
    def set_seperate_test_data(self, other_data_tuple = None):
        '''This function sets the data path and for a different test data then the initiated one.
        If the other data is None, do nothing.
        If given add the ML_results path the name of the data to test on.
        Args:
        1. other_data_tuple - tuple () - 0 - path to the test data, 1 - name of the data
        '''
        if not other_data_tuple: # None do nothing
            return
        path, name = other_data_tuple
        name = f'on_{name}'
        self.validate_path_exsits(path)
        self.merged_data_path = path
        self.ml_results_path = self.add_to_path(self.ml_results_path,name)
        self.add_type_to_plots_path(name)
        self.other_test_data_initiated = True
    
    def add_type_to_models_paths(self, type):
        '''
        Given a type create folders in ML_results and Models with the type
        if job is train create only the models path.
        type will be anything to add:
        1. model type - cnn,rnn...
        2. cross val type -  k_fold, leave_one_out,ensmbel,
        3. features - only_seq, epigenetics, epigenetics_in_seq, spatial_epigenetics'''
        self.validate_ml_results_and_model()
       # create folders
        if self.job == "test":
            self.models_path = self.add_to_path(self.models_path,type)
            self.ml_results_path = self.add_to_path(self.ml_results_path,type)
        elif self.job == "train":      
            self.models_path = self.add_to_path(self.models_path,type)
        elif self.job == "evaluation" or self.job == "process":
            self.ml_results_path = self.add_to_path(self.ml_results_path,type)

    def add_type_to_plots_path(self, type):
        '''
        Adds a type to the plots path if the job is setted to evaluation.
        '''
        if self.job == "evaluation":
            self.plots_path = self.add_to_path(self.plots_path,type)
        
           
  
    ## ENSMBELS and K fold partitions and paramaters ##    
    def add_partition_path(self, partition_str = None):
        '''
        This function adds the partition number to the model and model results phats.
        If partition_str is given the function will use it.
        if not it will use the partition list setted.
        '''
        if partition_str:
            self.add_type_to_models_paths(partition_str)
        else:
            partition_str = "-".join(map(str,self.partition)) # Join the partition numbers into str seperated by '-'
            self.add_type_to_models_paths(f'{partition_str}_partition')
    
    def set_n_ensembels(self, n_ensembels):
        validate_non_negative_int(n_ensembels)
        self.n_ensembels = n_ensembels
        if self.partition:
            self.add_type_to_models_paths(f'{self.n_ensembels}_ensembels')
        else :
            raise ValueError("Cannot set ensembels without partition")
    
    def set_n_models(self, n_models):
        validate_non_negative_int(n_models)
        if self.n_ensembels:
            self.add_type_to_models_paths(f'{n_models}_models')
        else:
            raise ValueError("Cannot set models number without ensembels or partition")
    
     
    def add_to_path(self, path, path_to_add):
        '''
        This function take two paths, concatenate tham togther, create a folder and return the new path.
        '''
        temp_path = os.path.join(path, path_to_add)
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        return temp_path
   
    
    
    def set_partition(self, partition_list, train = False, test = False):
        '''
        This function sets the partition of the train/test fold.
        It it used in ensemble models and k fold cross validation where each ensmble/model is trained on a different partition.

        The partition number argument is a list of numbers or a list of one string "All".
        if numbers: Each number is a partition number.
        The fuction check if the partition number is in the range of the number of partitions
        If so it set the partition number list other wise it raise an exception.

        If the partition is "All" than the function sets the partition to "All" and no partition is needed.
        I.E. all guides are for training or testing.
        If the partition is All and job is Test, there are no more guides to test on in the original data, than the test_on_other_data should be given!

        Args: 
        1. partition_list - [list] of ints, partitions numbers. If the first partition is "All" than no partition is needed.
        2. train - bool, default False, if True set the train partition
        3. test - bool, default False, if True set the test partition
        
        '''
                
        if self.check_if_partition_is_all(partition_list): # partition is all
            if test: # All + test means that the other test data should beeen initiated
                if self.other_test_data_initiated:
                    return
                else:
                    raise Exception("Test on other data not set while partition is All and jon is Test.")
        if self.partition == "All": # partition already set to all in prevouis function
            return        
        
        if train:
            self.guide_prefix = "Train"
            self.add_data_type_toguides_path("Train_guides")
        elif test:
            self.guide_prefix = "Test"
            self.add_data_type_toguides_path("Test_guides")
        else:
            raise ValueError('eather test or train must be set to True')
        
        # Check for number of partitions
        if os.path.exists(self.guides_partition_path):
            self.partition = []
            
            for partition in partition_list:
                
                # check for partition
                if partition > 0 and partition <= len(os.listdir(self.guides_partition_path)):
                    self.partition.append(partition)
                else:
                    self.partition = [] # clear the list
                    raise Exception(f'Partition number {partition} is out of range')
        else: 
            raise Exception('Guides path not set')
        self.add_partition_path()


    def check_if_partition_is_all(self, partition_list):
        '''
        This function first validate the partition list is not an empty list.
        Than checks if the first element is a string and if it is "All"'''
        if isinstance(partition_list,list) and len(partition_list) > 0:
            if isinstance(partition_list[0],str):
                if partition_list[0].lower() == "all":
                    self.partition = "All"
                    #self.add_partition_path("All_guides")
                    return True
                else : raise Exception('If first partition is STR it must be "All"')  
            return False
        else: raise Exception('Partition list is empty')
    def get_partition(self):
        if self.partition:
            return self.partition
        else: raise Exception('Partition not set')
    
    
    
    
    
    def get_guides_partition(self):
        '''
        This function returns the guide for the partition once setted in the self.partition argument.
        If the self.partition/path not setted, raises an exception.

        Else sort the list of guides by partition number and return the GUIDES in the partition guides path.

        If the partition is "All" than return None - functions that use the guides will know to use all guides.

        Returns: [list] of guides or None.
        '''
        guides = []
        if self.partition == "All":
            return None
        
        ## Checks path existence and set the guides paths.
        if self.guides_partition_path:
            guides_list = os.listdir(self.guides_partition_path)
            if self.partition:
                guides_path = []
                for partition in self.partition: # concatenate to the guide path the test/train prefix and the partition numner
                    guides_txt = f'{self.guide_prefix}_guides_{partition}_partition.txt'
                    if guides_txt not in guides_list:
                        raise RuntimeError(f'Guides for partition {partition} not found')
                    guides_path.append(os.path.join(self.guides_partition_path,guides_txt))
            else : raise Exception('Partition not set')
        else : raise Exception('Guides path not set')
        ## Create the guides list from the guides paths.
        for guide_path in guides_path:
            guides += create_guides_list(guide_path, 0)
        return guides
        
    
    def set_bigwig_files(self,bw_list):
        flag = False
        if bw_list: #bw list isnt empty
            flag = True
            # check if bw/bedgraph
            for file_name,file_object in bw_list:
                if not file_object.isBigWig():
                    # not bigwig throw error
                    flag = False
                    raise Exception(f'trying to set bigwig files with other type of file\n{file_name}, is not bw file')
                else : 
                    continue # check next file
        if flag: # not empty list + all files are big wig
           #self.close_big_wig(only_bw_object)
           self.bigwig_files = bw_list
           self.bigwig_amount = len(self.bigwig_files)
        else : # flag is false list is empty
            raise Exception('Trying to set bigwig files with empty list, try agian.') 


    # Functions to create paths from folders
    

    '''Create pyBigWig objects list of the bigwig files'''
    def create_bigwig_files_objects(self):
        self.bigwig_files = []
        for path in create_paths(self.bigwig_folder_path):
            name = get_ending(path) # retain the name of the file (includes the marker)
            try:
                name_object_tpl = (name,pyBigWig.open(path))
                self.bigwig_files.append(name_object_tpl)
            except Exception as e:
                print(e)
        self.bigwig_amount = len(self.bigwig_files) # set amount
        
    def create_bed_files_objects(self):
        self.bed_files = []
        for path in create_paths(self.bed_folder_path):
            name = get_ending(path) # retain the name of the file (includes the marker)
            try:
                name_object_tpl = (name,pybedtools.BedTool(path))
                self.bed_files.append(name_object_tpl)
            except Exception as e:
                print(e)
        self.bed_files_amount = len(self.bed_files) # set amount
    '''Function to close all bigwig objects'''
    def close_big_wig(self,new_bw_object_list):
        if self.bigwig_files: # not empty
            for file_name,file_object in self.bigwig_files:
                if file_object in new_bw_object_list: # setting new list with objects from old one - dont close the file
                    continue
                else:
                    try:
                        file_object.close()
                    except Exception as e:
                        print(e)
    def close_bed_files(self):
        if self.bed_files:
            pybedtools.cleanup(remove_all=True)
    def set_global_bw_max(self):
        self.glb_max_dict = {}
        for bw_name,bw_file in self.bigwig_files:
            # get chroms
            chroms = bw_file.chroms()
            max_list = []
            for chrom,length in chroms.items():
                # get max
                max_val = bw_file.stats(chrom,0,length,type='max')[0]
                max_list.append(max_val)
            self.glb_max_dict[bw_name] = max(max_list)

    '''Function to save machine learning results to file
    '''
    def save_ml_results(self, results_table, model_name): 
        # concatenate self.model_results_output_path with model_name
        output_path = os.path.join(self.model_results_output_path,f'{model_name}.csv')
        # save results to file
        results_table.to_csv(output_path)
    def create_ensemble_train_folder(self, i_ensmbel):
        output_path = os.path.join(self.models_path,f'ensemble_{i_ensmbel}')
        # create dir output_path if not exsits
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        return output_path
    
    def create_ensemble_result_folder(self, i_ensmbel):
        output_path = os.path.join(self.ensmbel_result_path,f'ensemble_{i_ensmbel}')
        # create dir output_path if not exsits
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        return output_path
    
    def create_ensemble_score_nd_combi_folder(self):
        if not self.ml_results_path:
            raise Exception("Ensemble result path not set")
        score_path = os.path.join(self.ml_results_path,"Scores")
        combi_path = os.path.join(self.ml_results_path,"Combi")
        # if self.task.lower() == "regression" or self.task.lower() == "t_regression":
        #     pos_combi_path = os.path.join(self.ml_results_path,"Pos_combi") 
        # else: pos_combi_path = None
        if not os.path.exists(score_path):
            os.makedirs(score_path)
        if not os.path.exists(combi_path):
            os.makedirs(combi_path)
        # if not os.path.exists(pos_combi_path):
        #     os.makedirs(pos_combi_path)
        return score_path,combi_path #pos_combi_path
        
    ## Validations
    '''Function to validate the paths'''
    def validate_path_exsits(self,path):
        assert os.path.exists(path), f"{path}Path does not exist"
    def validate_ml_results_and_model(self):
        if not self.ml_results_path:
            raise Exception("ML results path not set")
        if not self.models_path:
            raise Exception("Models path not set")
    '''dtor'''
    def __del__(self):
        #self.close_big_wig([])
        #self.close_bed_files()
        # call more closing
        pass



