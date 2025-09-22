'''
Module for ensemble utilities.
'''
from file_utilities import find_target_folders
from utilities import validate_non_negative_int
class ensemble_parmas:
    def __init__(self, n_models = None, n_ensembles = None, partition_num = None, job = None):
        self.n_models = validate_non_negative_int(n_models)
        self.n_ensembles = validate_non_negative_int(n_ensembles)
        self.partition_num = partition_num
        

    def set_file_manager_ensemble_params(self, file_manager = None, train = None, test = None):
        '''
        This function sets the file manager ensemble parameters.
        '''
        if not file_manager:
            raise ValueError("file_manager is None")
        file_manager.set_partition(self.partition_num, train, test)
        file_manager.set_n_ensembels(self.n_ensembles)
        file_manager.set_n_models(self.n_models)
        return file_manager, self.n_models, self.n_ensembles
        

def get_scores_combi_paths_for_ensemble(ml_results_path,n_ensembles,n_modles, all_features=False  ):
    """
    This function extracts from the ml_results_path paths that contianing the scores and combi folder.
    Path with scores, combi folders indicating there is a model that was trained and tested.
    
    Args:
        ml_results_path (str): path to the ml_results folder
        n_ensembles (int): number of ensembles
        n_modles (int): number of models
        all_features (bool): if True, will return all the paths that contain the n_ensembles and n_models

    Returns:
        (list) of paths to that contain scores and combi folders that match the n_ensemble and n_models.
    """
    if all_features:
        ml_results_path = ml_results_path.split("Ensemble")[0]
    scores_combis_paths =  find_target_folders(ml_results_path, ["Scores"])
    # remove folder that dont hold the n_ensembles and n_models
    filtered_paths = []
    for path in scores_combis_paths:
        if f'{n_ensembles}_ensembels' in path and f'{n_modles}_models' in path:
            filtered_paths.append(path)
    return filtered_paths



def copy_ml_results_from_1_to_10_ensembles(base_10, base_1):
    import os
    import shutil
    """
    Running 10_ensebmles ussualy done after running one ensmble. Creating 9 ensebmles and adding the first one is a good time saver.
    This function copies the ensemble_1.csv files from subdirectories in base_1 to corresponding
    subdirectories in base_10 if they exist.
    
    Args:
        base_10 (str): Base directory for 10_ensembles.
        base_1 (str): Base directory for 1_ensembles.
    Example usage
    base_10 = "ML_results/Change-seq/vivo-silico/Exclude_Refined_TrueOT/on_Refined_TrueOT_Lazzarroto/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/With_features_by_columns/All_guides/10_ensembels/50_models/Binary_epigenetics"
    base_1 = "ML_results/Change-seq/vivo-silico/Exclude_Refined_TrueOT/on_Refined_TrueOT_Lazzarroto/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/With_features_by_columns/All_guides/1_ensembels/50_models/Binary_epigenetics"

    copy_ml_results_from_1_to_10_ensembles(base_10, base_1)
    """
    for root, dirs, files in os.walk(base_1):
        if "Scores" in root:
            # Extract subdirectory name
            subdir = os.path.basename(os.path.dirname(root))
            
            # Define source and destination paths
            src = os.path.join(root, "ensemble_1.csv")
            dest_dir = os.path.join(base_10, subdir, "Scores")
            dest = os.path.join(dest_dir, "ensemble_1.csv")
            
            # Check if the source file exists
            if os.path.isfile(src):
                # Create target directory if it doesn't exist
                os.makedirs(dest_dir, exist_ok=True)
                
                # Copy the file
                shutil.copy2(src, dest)
                print(f"Copied {src} to {dest}")
            else:
                print(f"Source file {src} does not exist. Skipping.")

def copy_ensemble_model_from_1_to_10(base_10, base_1):
    import os
    import shutil
    """
    Running 10_ensebmles ussualy done after running one ensmble. Creating 9 ensebmles and adding the first one is a good time saver.
    This function copies the ensemble_1 folder files from subdirectories in base_1 to corresponding
    subdirectories in base_10 if they exist.
    
    Args:
        base_10 (str): Base directory for 10_ensembles.
        base_1 (str): Base directory for 1_ensembles.
    Example usage
    base_10 = "Models/Change-seq/vivo-silico/Exclude_Refined_TrueOT/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/With_features_by_columns/All_guides/10_ensembels/50_models/Binary_epigenetics"
    base_1 = "Models/Change-seq/vivo-silico/Exclude_Refined_TrueOT/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/With_features_by_columns/All_guides/1_ensembels/50_models/Binary_epigenetics"

    copy_ensemble_model_from_1_to_10(base_10, base_1)
    """
    for root, dirs, files in os.walk(base_1):
        if "ensemble_1" in dirs:
            # Extract subdirectory name
            subdir = os.path.basename(root)
            # Define source and destination paths
            src = os.path.join(root, "ensemble_1")
            dest_dir = os.path.join(base_10, subdir, "ensemble_1")
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            # Copy the 'ensemble_1' directory
            shutil.copytree(src, dest_dir, dirs_exist_ok=True)
            print(f"Copied {src} to {dest_dir}")

