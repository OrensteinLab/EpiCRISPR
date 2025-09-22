# File utilities
import os
import re
from shutil import rmtree
## FILES
def remove_dir_recursivly(dir_path):
    try:
        rmtree(dir_path)
        print(f"Directory '{dir_path}' and its contents have been removed.")
    except Exception as e:
        print(f"Error: {e}") 

def create_paths(folder):
    '''
    Create list off all the files/folders in the given folder.
    If the given path is not a folder return the path itself.
    Args:
        folder (str) - path to the folder
    Returns:
        list of paths
    '''
    if not os.path.isdir(folder):
        return [folder]
    paths = []
    for path in os.listdir(folder):
        paths.append(os.path.join(folder,path))
    return paths


def create_folder(path, extend = None):
    '''
    Create folder in spesefic path
    Args:
        path (str) - path to the folder
        extend (str) - name of the folder to create inside the path
    Returns:
        path to the created folder'''
    if not os.path.exists(path):
        try:
            os.makedirs(path)
            print("created new folder: ", path)
        except Exception as e:
            print(f"Error: {e}")
    if not extend is None:
        path = os.path.join(path,extend)
        if not os.path.exists(path):
            try:
                os.makedirs(path)
                print("created new folder: ", path)
            except Exception as e:
                print(f"Error: {e}")
    return path

def keep_only_folders(paths_list):
    '''Given list of paths return only folders from the list'''
    return [path for path in paths_list if os.path.isdir(path)]

def validate_path(path):
    '''Validate if path exists or not'''
    return os.path.exists(path)

def find_target_folders(root_dir, target_subdirs):
    '''This function will iterate the root directory and return paths to the target folders
    '''
    target_folders = []
    for current_dir, dirs, files in os.walk(root_dir):
        # Check if both "Scores" and "Combi" are in the current directory
       if all(subdir in dirs for subdir in target_subdirs):
            target_folders.append(current_dir)
    return target_folders

def find_target_files(root_dir, target_file):
    """
    Find paths where file/s with the target name are located.
    Returns all the paths that contain that file/s from the given root dir.
    
    Args:
        root_dir (str): Path to the root directory to search in.
        target_file (list): list of files to look for.
    Returns:
        list: List of paths where all target files are found.
    """
    if not isinstance(target_file, list):
        target_file = [target_file]
    target_folders = []
    for current_dir, dirs, files in os.walk(root_dir):
        if all(filename in files for filename in target_file):
            target_folders.append(current_dir)
    return target_folders


def extract_ensmbel_combi_inner_paths(base_path):
    '''This function will iterate the base path:
    Base path -> partitions -> inner folders (number of ensmbels) - > Combi
    --------
    Returns a list of paths to the Combi folders from each inner folder'''
    path_lists = []
    for partition in os.listdir(base_path): # iterate partition
        partition_path = os.path.join(base_path,partition)
        for n_ensmbels_path in os.listdir(partition_path): # iterate inner folders
            parti_ensmbels_path = os.path.join(partition_path,n_ensmbels_path)
            if os.path.isdir(os.path.join(parti_ensmbels_path,"Combi")): # if Combi folder exists
                path_lists.append(parti_ensmbels_path)
    return path_lists


def get_bed_folder(bed_parent_folder):
    ''' function iterate on bed folder and returns a list of tuples:
    each tuple: [0] - folder name [1] - list of paths for the bed files in that folder.'''  
    # create a list of tuples - each tuple contain - folder name, folder path inside the parent bed file folder.
    subfolders_info = [(entry.name, entry.path) for entry in os.scandir(bed_parent_folder) if entry.is_dir()]
    # Create a new list of tuples with folder names and the information retrieved from the get bed files
    result_list = [(folder_name, get_bed_files(folder_path)) for folder_name, folder_path in subfolders_info]
    return result_list

def get_bed_files(bed_files_folder):
    '''
    Return a list of the bed files in the folder.
    Args:
        bed_files_folder (str): path to the folder
    Returns:
        list of bed files paths'''
    bed_files = []
    for foldername, subfolders, filenames in os.walk(bed_files_folder):
        for name in filenames:
            # check file type the narrow,broad, bed type. $ for ending
            if re.match(r'.*(\.bed|\.narrowPeak|\.broadPeak)$', name):
                bed_path = os.path.join(foldername, name)
                bed_files.append(bed_path)
    return bed_files

def get_bigwig_files(bigwig_folder):
    '''
    Return a list of BigWig file paths in the given folder (recursively).
    
    Args:
        bigwig_folder (str): Path to the folder containing BigWig files.
        
    Returns:
        list of str: Full paths to .bw or .bigWig files found.
    '''
    bigwig_files = []
    for foldername, subfolders, filenames in os.walk(bigwig_folder):
        for name in filenames:
            # Match files ending with .bw or .bigWig (case-insensitive)
            if re.match(r'.*\.(bw|bigWig)$', name, re.IGNORECASE):
                bigwig_path = os.path.join(foldername, name)
                bigwig_files.append(bigwig_path)
    return bigwig_files

def get_bed_and_bigwig_files(folder_path):
    '''
    Recursively search a folder and return lists of BED and BigWig file paths.

    Args:
        folder_path (str): Path to the folder to search.

    Returns:
        tuple:
            - list of str: Paths to .bed, .narrowPeak, or .broadPeak files.
            - list of str: Paths to .bw or .bigWig files.
    '''
    bed_files = []
    bigwig_files = []

    for dirpath, _, filenames in os.walk(folder_path):
        for name in filenames:
            if re.search(r'\.(bed|narrowPeak|broadPeak)$', name, re.IGNORECASE):
                bed_files.append(os.path.join(dirpath, name))
            elif re.search(r'\.(bw|bigWig)$', name, re.IGNORECASE):
                bigwig_files.append(os.path.join(dirpath, name))

    return bed_files, bigwig_files

def get_ending(txt):
    ending = txt.split("/")[-1].split(".")[0]
    return ending

def copy_ensmebles():
    import os
    import shutil

    # Define the source and destination base directories
    source_base = "ML_results/Change-seq/vivo-vitro/Classification/CNN/Ensemble/Epigenetics_by_features/7_partition/7_partition_50/binary"
    dest_base = "ML_results/Change-seq/vivo-vitro/Classification/CNN/Ensemble/Epigenetics_by_features/7_partition/1_ensembels/50_models/binary"

    # Iterate through all folders in the source binary directory
    for folder in os.listdir(source_base):
        source_scores_dir = os.path.join(source_base,f'{folder}/Scores' )
        dest_scores_dir = os.path.join(dest_base, f'{folder}/Scores')
        
        # Check if the source Scores directory exists
        if os.path.exists(source_scores_dir):
            source_file = os.path.join(source_scores_dir, "ensmbel_1.csv")
            # Check if the file exists before copying
            if os.path.exists(source_file):
                # Create the destination Scores directory if it doesn't exist
                os.makedirs(dest_scores_dir, exist_ok=True)
                # Copy the file to the destination directory
                shutil.copy(source_file, dest_scores_dir)
                print(f"Copied {source_file} to {dest_scores_dir}")
            else:
                print(f"File not found: {source_file}")
        else:
            print(f"Scores directory not found in: {os.path.join(source_base, folder)}")
