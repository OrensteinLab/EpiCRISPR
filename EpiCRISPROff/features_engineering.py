'''
This is a module to create feature and labels from data.
'''

import pandas as pd
import numpy as np
import os
from pybedtools import BedTool
from sklearn.utils import shuffle
import itertools
from features_and_model_utilities import get_encoding_parameters, transform_labels
from data_constraints_utilities import return_constrained_data
from utilities import get_k_choose_n
ALL_INDEXES = [] # Global variable to store indexes of data points when generating features

## FUNCTIONS:
# 1.
'''Args: 
1. whole data table - positives and negativs
Function: takes the data table, create a unique list of gRNAs, Split the data into seperate data frames
Based on gRNA
Outputs: 1. Dictionray - {gRNA : Data frame} 2. unique gRNA set
'''
def create_data_frames_for_features(data, if_data_reproducibility, target_column, 
                                    exclude_guides = None, test_on_other_data = False, 
                                    if_bulges = False, bulges_column = None):
    """
    Creates a dictionary of DataFrames, where keys are gRNA names and values are corresponding DataFrames.
    
    Args:
        data (str): Path to the data file.
        if_data_reproducibility (bool): If True, the data will be sorted for reproducibility.
        target_column (str): The name of the column containing the gRNA names.
        exclude_guides (tuple): (guides_description, path to guides to exclude from the data, target_column)
        test_on_other_data (bool): If True, the guides will not be excluded from the data.
        if_bulges (bool): If True, bulges will be included in the data."""
    if isinstance(data,str):
        data_table = pd.read_csv(data) # open data
    else: data_table = data
    if not if_bulges: # if bulges are not included
        data_table = data_table[data_table[bulges_column] == 0] # remove bulges
    if exclude_guides: # exlucde not empty
        if not test_on_other_data: # 
            data_table = return_df_without_guides(data_table, exclude_guides, target_column)
    # set unquie guide identifier, sorted if reproducibilty is need with data spliting
    if if_data_reproducibility:
        guides = sorted(set(data_table[target_column])) 
    else : 
        guides = list(set(data_table[target_column]))
        guides = shuffle(guides)
        # Create a dictionary of DataFrames, where keys are gRNA names and values are corresponding DataFrames
    df_dict = {grna: group for grna, group in data_table.groupby(target_column)}
    # Create separate DataFrames for each gRNA in the set
    result_dataframes = {grna: df_dict.get(grna, pd.DataFrame()) for grna in guides}
    return (result_dataframes, guides)

def return_df_without_guides(data_frame, guide_to_exlucde, data_frame_column):
    '''
    Return a dataframe without the guides in guides_to_exclude.
    Args:
        data_frame: A dataframe containing the data
        guide_to_exclude: (Tuple) (guides_description, path to guides to exclude from the data, target_columns)
    '''
    
    description, path, target_columns = guide_to_exlucde
    guides_to_exclude = set()
    guides_data = pd.read_csv(path)
    for column in target_columns:
        guides_to_exclude.update(guides_data[column].dropna().unique())  # Remove NaN values and add unique guides
    
    # Return the dataframe without the excluded guides
    return data_frame[~data_frame[data_frame_column].isin(guides_to_exclude)]

def add_downstream_to_sgRNa_OT_sequences(data, target_column, off_target_column, downstream_column, downstream_length):
    """
    Appends a specified number of downstream nucleotides to both target and off-target sequences in a DataFrame.

    Parameters:
    -----------
    data : pandas.DataFrame
        The input DataFrame containing sequence information.
    
    target_column : str
        The name of the column containing target (sgRNA) sequences to be extended.
    
    off_target_column : str
        The name of the column containing off-target sequences to be extended.
    
    downstream_column : str
        The name of the column containing downstream sequence context (e.g., genomic flanking regions).
    
    downstream_length : int
        The number of nucleotides to extract from the start of the downstream sequence and append.

    Returns:
    --------
    pandas.DataFrame
        The DataFrame with updated target and off-target sequence columns extended by the downstream sequence.
    
    Notes:
    ------
    This function modifies the original `data` DataFrame in-place and returns it.
    Assumes that all specified columns contain string-like sequence data.
    """
    if isinstance(data,str):
        data = pd.read_csv(data)
    data[target_column] = data[target_column] + data[downstream_column].str[:downstream_length]
    data[off_target_column] = data[off_target_column] + data[downstream_column].str[:downstream_length]
    return data


def generate_features_and_labels(data_path, manager, if_bp, if_only_seq , 
                                 if_seperate_epi, epigenetic_window_size, features_columns, if_data_reproducibility,
                                 columns_dict, transform_y_type = False, sequence_coding_type = 1, if_bulges = False,
                                 exclude_guides = None, test_on_other_data = False, return_otss = False,exclude_ontarget=False, 
                                 if_downstream = False, downstream_length = 9, downstream_in_seq = True):
    '''
    This function generates x and y data for gRNAs and their corresponding off-targets.
    For each (gRNA, OTS) pair it one-hot encodes the sequences and adds epigenetic data if required.
    For each pair it will return their corresponding y values (1/0/read_count).
    It iterates on each gRNA : Data frame and extract the data.
    Uses internal functions for seq encoding, epigentic encoding, and bp intersection of epigenetics with seq.
    
    Args:
        Data path (str): Path to the data file.
        manager (FileManager): File manager instance to get epigenetic files and their data.
        if_bp (bool): If True, adds epigenetic data for each base pair.
        if_only_seq (bool): If True, uses only sequence encoding.
        if_seperate_epi (bool): If True, uses epigenetic data in a separate vector.
        epigenetic_window_size (int): The size of the window for epigenetic data.
        features_columns (list): List of columns to add as features.
        if_data_reproducibility (bool): If True, sorts the data for reproducibility.
        columns_dict (dict): Dictionary of columns to use in the data frame with the following keys:
            TARGET_COLUMN, OFFTARGET_COLUMN, CHROM_COLUMN, START_COLUMN, END_COLUMN, 
            BINARY_LABEL_COLUMN, REGRESSION_LABEL_COLUMN, Y_LABEL_COLUMN
        transform_y_type (str): the type of transformation to apply to the y values.
        sequence_coding_type (int): the type of sequence encoding to use -  defualt is 1 - PiCRISPR style. 2 -  nuc*nuc per base pair.
        if_bulges (bool): If True, includes bulges in the sequence encoding.
        exclude_guides (tuple): (guides_description, path to guides to exclude from the data, target_column)
        test_on_other_data (bool): If True, does not exclude guides from that data.
        return_otss (bool): If True, returns the OTS and gRNA sequences.
        exclude_ontarget (bool): If True, excludes the ontarget from the data.
        
    Returns:
    A tuple containing:
        x_data_all, y_labels_all, guides, grna_otss_dict (if return_otss is True)
    where:
        x_data_all - list of sgRNA,OTS encoded pairs.
        y_labels_all - list of labels for each gRNA, OTS pair.
        guides - list of unique gRNAs.
        grna_otss_dict - dictionary of gRNA and their corresponding OTS sequences (if return_otss is True).
    
'''
    #NOTE: Adding downstream to sgrna and off_target_sequences!
    if if_downstream:
        if downstream_in_seq:
            data_path = add_downstream_to_sgRNa_OT_sequences(data_path, target_column=columns_dict["REALIGNED_COLUMN"],
                                                        off_target_column=columns_dict["OFFTARGET_COLUMN"],
                                                        downstream_column=columns_dict['DOWNSTREAM_COLUMN'],
                                                        downstream_length=downstream_length)
    splited_guide_data,guides = create_data_frames_for_features(data_path, if_data_reproducibility,
                                                                columns_dict["TARGET_COLUMN"],exclude_guides,test_on_other_data,
                                                                if_bulges,columns_dict["BULGES_COLUMN"])
    x_data_all = []  # List to store all x_data
    y_labels_all = []  # List to store all y_labels
    ALL_INDEXES.clear() # clear indexes
    seq_len,nuc_num = get_encoding_parameters(sequence_coding_type,if_bulges,downstream_in_seq,downstream_length) # get sequence encoding parameters
    encoded_length = seq_len * nuc_num # set encoded length
    grna_otss_dict = {} # init dict for sgRNA and its OTSS
    for guide_data_frame in splited_guide_data.values(): # for every guide get x_features by booleans
        if exclude_ontarget:
            guide_data_frame = guide_data_frame[~((guide_data_frame[columns_dict["MISMATCH_COLUMN"]] == 0) & 
                                                  (guide_data_frame[columns_dict["BULGES_COLUMN"]] == 0))]
        # get seq info - represented in all!
        if sequence_coding_type == 1: # PiCRISPR style
            seq_info = PiCRISPR_one_hot(data=guide_data_frame, encoded_length = encoded_length, bp_presenation = nuc_num,
                                    off_target_column=columns_dict["OFFTARGET_COLUMN"],
                                    target_column=columns_dict["REALIGNED_COLUMN"]) #int8
        elif sequence_coding_type == 2: # Full encoding
            seq_info = full_one_hot_encoding(dataset_df=guide_data_frame, n_samples=len(guide_data_frame), seq_len=seq_len, nucleotide_num=nuc_num,
                                  off_target_column=columns_dict["OFFTARGET_COLUMN"], target_column=columns_dict["REALIGNED_COLUMN"])

        if if_bp: # epigentic value for every base pair in gRNA
            big_wig_data = get_bp_for_one_hot_enconded(data = guide_data_frame, encoded_length = encoded_length, manager = manager,
                                                        bp_presenation = nuc_num, chr_column = columns_dict["CHROM_COLUMN"],
                                                        start_column = columns_dict["START_COLUMN"], end_column = columns_dict["END_COLUMN"])
            seq_info = seq_info.astype(np.float32) 
            x_data = seq_info + big_wig_data
           
        elif if_seperate_epi: # seperate vector to epigenetic by window size
            epi_window_data = get_seperate_epi_by_window(data = guide_data_frame, epigenetic_window_size = epigenetic_window_size, 
                                                         manager = manager, chr_column = columns_dict["CHROM_COLUMN"],
                                                         start_column = columns_dict["START_COLUMN"])
            seq_info = seq_info.astype(np.float32)
            x_data = np.append(seq_info, epi_window_data ,axis=1)
        elif if_only_seq:
            x_data = seq_info
        else : # add features into 
            x_data = guide_data_frame[features_columns].values.astype(np.int8)
            x_data = np.append(seq_info, x_data, axis = 1)
        #NOTE: adding downstream by additional vector. now adding by sequence.
        if if_downstream: # add downstream sequence
            if not downstream_in_seq:
                    
                down_stream_seq = guide_data_frame[columns_dict["DOWNSTREAM_COLUMN"]].values
                down_stream_seq = oneHot_sequences(down_stream_seq, length=downstream_length, size=4)
                down_stream_seq = down_stream_seq.reshape(down_stream_seq.shape[0], downstream_length*4)
                x_data = np.append(x_data, down_stream_seq, axis = 1)
        if "Index" in guide_data_frame.columns:
            ALL_INDEXES.append(guide_data_frame["Index"])
        else:
            ALL_INDEXES.append(guide_data_frame.index)
        x_data_all.append(x_data)
        
        y_labels_all.append(guide_data_frame[[columns_dict["Y_LABEL_COLUMN"]]].values) # add label values by extracting from the df by series values.
        if return_otss:
            grna_otss = guide_data_frame[[columns_dict["REALIGNED_COLUMN"],columns_dict["OFFTARGET_COLUMN"]]].values
            grna_otss_dict[guide_data_frame[columns_dict["TARGET_COLUMN"]].iloc[0]] = grna_otss
            
    del splited_guide_data # free memory
    
    if transform_y_type:
        y_labels_all = transform_labels(y_labels_all, transform_y_type)
    if return_otss:
        return (x_data_all,y_labels_all,guides,grna_otss_dict)
    return (x_data_all,y_labels_all,guides)

def oneHot_sequences(sequences, length, size=4):
    """
    Truncate sequences to a fixed length and return one-hot encoded numpy array.

    Args:
        sequences (list or pandas Series): List of DNA sequences (strings).
        length (int): Desired fixed length to truncate each sequence.
        size (int): Alphabet size (default 4 for ACGT).

    Returns:
        np.ndarray: One-hot encoded array of shape (n_sequences, length, size)
    """
    # Translation: A → 0, C → 1, G → 2, T → 3
    translator = str.maketrans("ACGT", "0123")

    # Step 1: Truncate and translate
    sequence_as_ints = [
        list(map(int, seq[:length].translate(translator)))
        for seq in sequences
    ]
    # Step 2: Convert to numpy and one-hot encode
    int_array = np.array(sequence_as_ints, dtype=int)
    one_hot = np.eye(size)[int_array]  # Shape: (N, length, size)
    one_hot = one_hot.astype(np.int8)  # Convert to int8 for memory efficiency
    return one_hot

    
    


def PiCRISPR_one_hot(data, encoded_length, bp_presenation, off_target_column, target_column):
    """
    Generate one-hot encoded representation of gRNA and off-target sequences.

    Args:
        data (pd.DataFrame): DataFrame containing sequence data.
        encoded_length (int): The length of the encoded sequence.
        bp_presenation (int): Vector size for each base pair representation.
        off_target_column (str): Column name for off-target sequences in the DataFrame.
        target_column (str): Column name for gRNA sequences in the DataFrame.

    Returns:
        np.ndarray: Array of shape (num_data_points, encoded_length) with one-hot encoded sequences.
    """
    seq_info = np.ones((data.shape[0], encoded_length),dtype=np.int8)
    for index, (otseq, grnaseq) in enumerate(zip(data[off_target_column], data[target_column])):
        otseq = enforce_seq_length(otseq, 23)
        grnaseq = enforce_seq_length(grnaseq, 23)
        otseq = otseq.upper()
        seq_info[index] = partial_one_hot_enconding(otseq, grnaseq,encoded_length,bp_presenation)
    return seq_info
def get_bp_for_one_hot_enconded(data, encoded_length, manager, bp_presenation, chr_column, start_column, end_column):
    """
    Generate epigenetic-augmented representation for gRNA sequences.

    Args:
        data (pd.DataFrame): DataFrame containing genomic information for gRNA.
        encoded_length (int): Length of the encoded representation.
        manager: File manager object responsible for handling access to epigenetic signal files.
        bp_presenation (int): Vector size for each base pair representation.
        chr_column (str): Column name for chromosome identifier.
        start_column (str): Column name for start position of the gRNA.
        end_column (str): Column name for end position of the gRNA.

    Returns:
        np.ndarray: Array of shape (num_data_points, encoded_length) where each base pair is encoded 
                    with epigenetic features (e.g., peak values) from the given files.
    
    Function:
        Initializes a matrix filled with ones of shape (number of data points, encoded_length). 
        For each data point, epigenetic features corresponding to the gRNA's genomic region are 
        extracted using the file manager and used to populate the encoded vector.
    """
    bigwig_info = np.ones((data.shape[0],encoded_length))
    for index, (chrom, start, end) in enumerate(zip(data[chr_column], data[start_column], data[end_column])):
        if not (end - start) == 23:
            end = start + 23
        bigwig_info[index] = bws_to_one_hot(file_manager=manager,chr=chrom,start=start,end=end,encoded_length=encoded_length,bp_presenation=bp_presenation)
    bigwig_info = bigwig_info.astype(np.float32)
    return bigwig_info

def get_seperate_epi_by_window(data, epigenetic_window_size, manager, chr_column, start_column):
    """
    Generate separate epigenetic feature vectors by applying a window around each off-target site.

    Args:
        data (pd.DataFrame): DataFrame containing genomic coordinates of gRNA off-target sites.
        epigenetic_window_size (int): Total size of the epigenetic window (i.e., number of base pairs to extract).
                                      Each feature will be centered on the off-target start location,
                                      extending half the window size in both directions.
        manager: File manager object that provides access to epigenetic signal files (e.g., bigWig files).
        chr_column (str): Column name specifying the chromosome for each gRNA.
        start_column (str): Column name specifying the start position of the off-target location.

    Returns:
        np.ndarray: Array of shape (num_data_points, epigenetic_window_size * num_epigenetic_tracks), 
                    where for each data point, epigenetic signal values from all files are concatenated.
    
    Function:
        Initializes an array of ones with shape based on the number of data points and total epigenetic 
        dimensions (window size multiplied by number of tracks). For each gRNA/off-target position, 
        extracts epigenetic signal values in the specified window and stores them in the array.
    """
    epi_data = np.ones((data.shape[0],epigenetic_window_size * manager.get_number_of_bigiwig())) # set empty np array with data points and epigenetics window size
    for file_index, (bw_epi_name, bw_epi_file) in enumerate(manager.get_bigwig_files()): # get one or more files 
        #glb_max = manager.get_global_max_bw()[bw_epi_name] # get global max all over bigwig
        filler_start = file_index * epigenetic_window_size
        filler_end = (file_index + 1) * epigenetic_window_size
        for index, (chrom, start) in enumerate(zip(data[chr_column], data[start_column])):
        
            epi_data[index,filler_start:filler_end] = get_epi_data_bw(epigenetic_bw_file=bw_epi_file,chrom=chrom,center_loc=start,window_size=epigenetic_window_size,max_type = 1)
        print(epi_data[0])
    epi_data = epi_data.astype(np.float32)
    return epi_data
## ONE HOT ENCONDINGS:
def partial_one_hot_enconding(sequence, seq_guide, encoded_length, bp_presenation):
    """
    Perform partial one-hot encoding for a pair of sequences (gRNA and off-target).

    Args:
        sequence (str): Off-target DNA sequence (OTS), expected to be uppercase and of fixed length.
        seq_guide (str): Guide RNA (gRNA) sequence aligned to the off-target.
        encoded_length (int): Total length of the encoded output vector.
        bp_presenation (int): Size of the representation for each base pair. 
                              Typically 6: 4 bits for base identity (A, T, C, G), 
                              and 2 bits to indicate sequence origin.

    Returns:
        np.ndarray: Flattened vector of length `encoded_length`, where each base is encoded 
                    with 4 bits for nucleotide identity and 2 bits indicating whether 
                    it belongs to the gRNA or off-target sequence.

    Function:
        Constructs a feature vector of length `encoded_length`. For each base:
          - The first 4 elements represent a one-hot encoding of the base (A, T, C, G).
          - The next 2 elements indicate the source: gRNA, off-target, or both.
        The final output is a flattened vector across all base positions.
    """
    bases = ['A', 'T', 'C', 'G']
    onehot = np.zeros(encoded_length, dtype=np.int8) # init encoded length zeros vector (biary vec, int8)
    sequence_length = len(sequence)
    for i in range(sequence_length): # for each base pair 
        for key, base in enumerate(bases): # set by key of [A-0,T-1,C-2,G-3]
            if sequence[i] == base: # OTS
                onehot[bp_presenation * i + key] = 1 
            if seq_guide[i] == base: # gRNA
                onehot[bp_presenation * i + key] = 1
        if sequence[i] != seq_guide[i]:  # Mismatch
            try: # Set direction of mismatch
                if bases.index(sequence[i]) < bases.index(seq_guide[i]):
                    onehot[bp_presenation * i + 4] = 1
                else:
                    onehot[bp_presenation * i + 5] = 1
            except ValueError:  # Non-ATCG base found
                pass
    return onehot

def full_one_hot_encoding(dataset_df, n_samples, seq_len, nucleotide_num, off_target_column, target_column):
    """
    Creates a one-hot encoding of sgRNA and off-target sequences.

    Args:
        dataset_df (pd.DataFrame): Dataset Dataframe. Must contain columns "SG_RNA_SEQ" and "OFF_TARGET".
        n_samples (int): Total number of samples in the dataset.
        seq_len (int): Length of the sequences.
        nucleotide_num (int): Number of distinct nucleotides (5 when including bulges).

    Returns:
        np.ndarray: One-hot encoded array, shape: (n_samples, seq_len, nucleotide_num ** 2)
    """
    nucleotides_to_position_mapping = create_nucleotides_to_position_mapping()

    one_hot_arr = np.zeros((n_samples, seq_len, nucleotide_num, nucleotide_num), dtype=np.int8)
    for i, (sg_rna_seq, off_seq) in enumerate(zip(dataset_df[target_column], dataset_df[off_target_column])):
        if len(off_seq) != len(sg_rna_seq):
            raise ValueError("len(off_seq) != len(sg_rna_seq)")
        actual_seq_size = len(off_seq)
        if actual_seq_size > seq_len:
            raise ValueError("actual_seq_size > seq_len")

        size_diff = seq_len - actual_seq_size
        for j in range(seq_len):
            if j >= size_diff:
                # note that it is important to take (sg_rna_seq_j, off_seq_j) as old models did the same.
                matrix_positions = nucleotides_to_position_mapping[(sg_rna_seq[j-size_diff], off_seq[j-size_diff])]
                one_hot_arr[i, j, matrix_positions[0], matrix_positions[1]] = 1
    # reshape to [n_samples, seq_len, nucleotide_num**2]
    one_hot_arr = one_hot_arr.reshape((n_samples, seq_len, nucleotide_num**2))
    one_hot_arr = one_hot_arr.reshape(n_samples, -1) # flatten the array
    return one_hot_arr

def reversed_ont_hot_to_seq(one_hot, bp_presenation):
    '''1.2 Reverse one hot encoding to sequence'''

    bases = ['A', 'T', 'C', 'G']
    sequence = ''
    guide_seq = ''
    for i in range(int(len(one_hot) / bp_presenation)):
        base_indices = np.nonzero(one_hot[i * bp_presenation:i * bp_presenation + 4])[0] # get indices of 1's
        # Check mismatchess
        if one_hot[i*bp_presenation + 4] == 1: # mismatch
            # First base is ots second is gRNA
            sequence += bases[base_indices[0]]
            guide_seq += bases[base_indices[1]]
        elif one_hot[i*bp_presenation + 5] == 1: # mismatch
             # First base is gRNA second is ots
            sequence += bases[base_indices[1]]
            guide_seq += bases[base_indices[0]]
        else : # no mismatch add to both sequences the same value
            sequence += bases[base_indices[0]]
            guide_seq += bases[base_indices[0]]
    return sequence, guide_seq


def bws_to_one_hot(file_manager, chr, start, end,encoded_length,bp_presenation):
    '''2. bigwig (base pair epigentics) to one hot
Fill vector sized |encoded length| with values from bigwig file. 

'''
    # back to original bp presantation
    indexing = bp_presenation - file_manager.get_number_of_bigiwig()
    epi_one_hot = np.zeros(encoded_length,dtype=np.float32) # set epi feature with zeros
    try:
        for i_file,file in enumerate(file_manager.get_bigwig_files()):
            values = file[1].values(chr, start, end) # get values of base pairs in the coordinate
            for index,val in enumerate(values):
                # index * BP =  set index position 
                # indexing + i_file the gap between bp_presenation to each file slot.
                epi_one_hot[(index * bp_presenation) + (indexing + i_file)] = val
    except ValueError as e:
        return None
    return epi_one_hot

## Functions to obtain epienetic data for each base pair


def get_epi_data_bw(epigenetic_bw_file, chrom, center_loc, window_size,max_type):
    """
    Get epigenetic mark values per base via big wig file.
    Given chromosome, center location and window size.
    
    Args:
        epigenetic_bw_file (py.bigwig object): The bigwig file object.
        chrom (str): The chromosome name.
        center_loc (int): The center location of the window in that chromosome.
        window_size (int): The size of the window.
        max_type (int): The type of maximum value to use:
            if None: Normalize by the local max.
            if >1: Normalize by the given value max.

    """
    positive_step = negative_step = int(window_size / 2) # set steps to window/2
    if (window_size % 2): # not even
        positive_step += 1 # set pos step +1 (being rounded down before)

        
    chrom_lim =  epigenetic_bw_file.chroms(chrom)
    indices = np.arange(center_loc - negative_step, center_loc + positive_step)
    # Clip the indices to ensure they are within the valid range
    indices = np.clip(indices, 0, chrom_lim - 1)
    # Retrieve the values directly using array slicing
    y_values = epigenetic_bw_file.values(chrom, indices[0], indices[-1] + 1)
    # Get local min and local max
    min_val = epigenetic_bw_file.stats(chrom,indices[0],indices[-1] + 1,type="min")[0] 
    if max_type: # None for local max, other 1 or global
        if max_type <= 0:
            raise ValueError("max_type should be positive")
        max_val = max_type
    else :
        max_val = epigenetic_bw_file.stats(chrom,indices[0],indices[-1] + 1,type="max")[0] 
        if max_val == 0.0: # max val is 0 then all values are zero
            return np.zeros(window_size,dtype=np.float32) 
    # Create pad_values using array slicing
    pad_values_beginning = np.full(max(0, positive_step - center_loc), min_val)
    pad_values_end = np.full(max(0, center_loc + negative_step - chrom_lim), min_val)

    # Combine pad_values with y_values directly using array concatenation
    y_values = np.concatenate([pad_values_beginning, y_values, pad_values_end])
    y_values = y_values.astype(np.float32)
    y_values[np.isnan(y_values)] = min_val # replace nan with min val
    y_values /= max_val # devide by max [local/global/1].
    return y_values



def get_epi_data_bed(epigenetic_bed_file, chrom, center_loc,window_size):
    '''2. CREATE epigenetic data with 1/0 values via bed file (interval) information.
uses help function - update_y_values_by_intersect()'''
    positive_step = negative_step = int(window_size / 2) # set steps to window/2
    if (window_size % 2): # not even
        positive_step += 1 # set pos step +1 (being rounded down before)
    start = center_loc - negative_step # set start point
    end = center_loc + positive_step # set end point
    string_info = f'{chrom} {start} {end}' # create string for chrom,start,end
    ots_bed = BedTool(string_info,from_string=True) # create bed temp for OTS
    intersection = ots_bed.intersect(epigenetic_bed_file) 
    if not len(intersection) == 0: # not empty
        y = update_y_values_by_intersect(intersection, start, window_size)
    else : 
        y = np.zeros(window_size,dtype=np.int8)
    os.remove(intersection.fn)
    os.remove(ots_bed.fn)
    return y
    
## Assistant functions:
def enforce_seq_length(sequence, requireLength):
    if (len(sequence) < requireLength): sequence = '0'*(requireLength-len(sequence))+sequence # in case sequence is too short, fill in zeros from the beginning (or sth arbitrary thats not ATCG)
    return sequence[-requireLength:] # in case sequence is too long
'''help function for get_epi_by_bed'''
def update_y_values_by_intersect(intersect_tmp, start, window_size):
    print(intersect_tmp.head())
    y_values = np.zeros(window_size,dtype=np.int8) # set array with zeros as window size i.e 2000
    for entry in intersect_tmp:
        intersect_start = entry.start # second field (0-based) is start
        intersect_end = entry.end # 3rd fiels is end
    # get indexs for array values allways between 0 and window size
        if intersect_start == start:
            start_index = 0
        else : start_index = intersect_start - start - 1
        end_index = intersect_end - start - 1
    
        y_values[start_index:end_index] = 1 # set one for intersection range
    y_values[0] = 0 # for better respresnation - setting 0,0
    return y_values


def create_nucleotides_to_position_mapping():
    """
    Creates a mapping of nucleotide pairs (sgRNA, off-target) to their numerical positions.
    This mapping includes positions for "N" nucleotides (representing any nucleotide).

    Returns:
        dict: A dictionary where keys are tuples of nucleotides ("A", "T"), ("G", "C"), etc.,
              and values are tuples representing their (row, column) positions in a matrix.
    """
    # matrix positions for ("A","A"), ("A","C"),...
    # tuples of ("A","A"), ("A","C"),...
    nucleotides_product = list(itertools.product(*(["ACGT-"] * 2)))
    # tuples of (0,0), (0,1), ...
    position_product = [(int(x[0]), int(x[1]))
                        for x in itertools.product(*(["01234"] * 2))]
    nucleotides_to_position_mapping = dict(
        zip(nucleotides_product, position_product))

    # tuples of ("N","A"), ("N","C"),...
    n_mapping_nucleotides_list = [("N", char) for char in ["A", "C", "G", "T", "-"]]
    # list of tuples positions corresponding to ("A","A"), ("C","C"), ...
    n_mapping_position_list = [nucleotides_to_position_mapping[(char, char)]
                               for char in ["A", "C", "G", "T", "-"]]

    nucleotides_to_position_mapping.update(
        dict(zip(n_mapping_nucleotides_list, n_mapping_position_list)))

    # tuples of ("A","N"), ("C","N"),...
    n_mapping_nucleotides_list = [(char, "N") for char in ["A", "C", "G", "T", "-"]]
    # list of tuples positions corresponding to ("A","A"), ("C","C"), ...
    n_mapping_position_list = [nucleotides_to_position_mapping[(char, char)]
                               for char in ["A", "C", "G", "T", "-"]]
    nucleotides_to_position_mapping.update(
        dict(zip(n_mapping_nucleotides_list, n_mapping_position_list)))

    return nucleotides_to_position_mapping
## Data and features manipulation

def get_guides_indexes(guide_idxs):
    '''
    Function to extract the guides indexes given which guides to keep.
The guides to keep are given as a list of indexes.
The function take the guides_indexes and return from ALL_INDEXES and spesific guide indexes
'''
    choosen_indexes = [index for idx in guide_idxs for index in ALL_INDEXES[idx]]
    choosen_indexes = np.array(choosen_indexes)
    return choosen_indexes


def synthesize_mismatch_off_targets(sgRNA_seqeunce, num_missmatches= 0, if_flatten = False,
                                    if_bulge_encoding = True, sample_data = False):
    '''
    Generates all potential off-targets for a given sgRNA sequence with a given number of mismatches/bulges.
    Args:
    sgRNA_seqeunce (str): The sgRNA sequence.
    num_missmatches (int): The number of mismatches.
    num_bulges (int): The number of bulges.
    sample_data (bool): If True, the data will be sampled.
    Returns:
    nd.array (n_samples, encoded off target): The one-hot encoded off-target sequences.
    '''    
    # First choose optional positions by chossing k positions from n long sequence.
    if num_missmatches <=0 or num_missmatches > 7:
        raise RuntimeError("mismatch number should between 1-7")
    n = len(sgRNA_seqeunce)
    if num_missmatches > n:
        raise RuntimeError(f"mismatch number ({num_missmatches}) is bigger than length of sequence ({n})")
    if n > 24:
        raise RuntimeError(f"sequence length ({n}) is bigger than 24")
    mismatches_indexes = np.array(get_k_choose_n(n, num_missmatches)) - 1 # -1 for 0-based indexing
    if sample_data:
        # number of indicies*3^num_missmatches* num_features is the total number of samples
        # number of features is unkown therefore we want indices*3^num_missmatches < 50000
        if len(mismatches_indexes) * 3 ** num_missmatches > 25000:
            number_of_indices = int(25000 // (3 ** num_missmatches))
            random_incides = np.random.choice(mismatches_indexes.shape[0], number_of_indices, replace=False)
            mismatches_indexes = mismatches_indexes[random_incides]
            # get the number of indices needed to get 2000 samples
            
    mismatches_tuples_dict = get_mismatches_tuples()
    matching_tuples_dict = {'A': (0,0), 'C': (1,1), 'G': (2,2), 'T': (3,3)}
    nucleotide_num =  4 if not if_bulge_encoding else 5
    all_off_targets = []
    #all_even_dis_off_targets = np.zeros((len(mismatches_indexes),n,nucleotide_num,nucleotide_num),dtype=np.float32)
    #even_dis_val = 1/3
    indexes_set = set(i for i in range(0,n))
    for j,index in enumerate(mismatches_indexes): 
        matching_indexes = np.array(list(indexes_set - set(index))) 
        matching_tuples = np.array([matching_tuples_dict[sgRNA_seqeunce[pos]] for pos in matching_indexes])
        mismatch_tuples = np.array(list(itertools.product(*[mismatches_tuples_dict[sgRNA_seqeunce[pos]] for pos in index])))
        m = len(mismatch_tuples)
        mismatch_array = np.zeros((m,n,nucleotide_num,nucleotide_num),dtype=np.int8)
        mismatch_array[:,matching_indexes,matching_tuples[:,0],matching_tuples[:,1]] = 1 # assign matching positions
        mismatch_array[np.arange(m)[:, None], index, mismatch_tuples[:, :, 0], mismatch_tuples[:, :, 1]] = 1 # assign mismatching positions
        # even_dis = np.zeros((n,nucleotide_num,nucleotide_num),dtype=np.float32)
        # even_dis[matching_indexes,matching_tuples[:,0],matching_tuples[:,1]] = 1
        # even_dis[index,mismatch_tuples[:, :, 0], mismatch_tuples[:, :, 1]] = even_dis_val
        # all_even_dis_off_targets[j] = even_dis
        all_off_targets.append(mismatch_array)
    
    all_off_targets = np.concatenate(all_off_targets, axis=0)
    if if_bulge_encoding: # append zeros in the first location 24 length instead of 23
        zeros = np.zeros((all_off_targets.shape[0], 1, nucleotide_num, nucleotide_num),dtype=np.int8)
        all_off_targets = np.concatenate([zeros, all_off_targets], axis=1)
        n = 24
        # zeros_ = np.zeros((all_even_dis_off_targets.shape[0], 1, nucleotide_num, nucleotide_num),dtype=np.float32)
        # all_even_dis_off_targets = np.concatenate([zeros_, all_even_dis_off_targets], axis=1)
    if if_flatten:
        all_off_targets = all_off_targets.reshape((all_off_targets.shape[0],n,nucleotide_num**2))
        all_off_targets = all_off_targets.reshape(all_off_targets.shape[0],-1)
        # all_even_dis_off_targets = all_even_dis_off_targets.reshape((all_even_dis_off_targets.shape[0],n,nucleotide_num**2))
        # all_even_dis_off_targets = all_even_dis_off_targets.reshape(all_even_dis_off_targets.shape[0],-1)
    return all_off_targets
    return all_off_targets,all_even_dis_off_targets

            
def synthesize_all_mismatch_off_targets(sgrna_sequence, mismatch_limit = 6, 
                                        if_range = True, if_flatten = False, sample_data = False):
    """
    Creates a dictionary with {mismatch number: synthesized off targets}

    Args:
        sgrna_sequence (str): the guide seqeunce
        mismatch_limit (int): the higher limit of the mismatches number to synthesize. defualt -6.
        if_range (bool): True - syntesize off target with 1 - mismatch_limit.
            False - syntesize off target with mismatch_limit only.
        if_flatten (bool): True - flatten the off target array.
        sample_data (bool): defualt False, True - sample the data
    Returns:
        dictionary of np.arrays: {mismatch_number: off targets}
    """
    if mismatch_limit < 0:
        raise RuntimeError('Mismatches should be positive')
    missmatches = [i for i in range(1,mismatch_limit+1)] if if_range else [mismatch_limit]
    return {mismatch_num: synthesize_mismatch_off_targets(sgrna_sequence,mismatch_num,if_flatten,sample_data=sample_data) for mismatch_num in missmatches}
                

    # Return the one-hot-encoded off-targets

def get_mismatches_tuples():
    '''
    Generates a dicionary of all possible mismatches for each nucleotide.
    For example {A: (0,1), (0,2), (0,3)..}
    Returns:
    dict: A dictionary of all possible mismatches for each nucleotide.
    '''
    nucleotide_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    # Generate the mismatch dictionary
    mismatch_dict = {
        letter: [(nucleotide_to_index[letter], index) for other, index in nucleotide_to_index.items() if letter != other]
        for letter in nucleotide_to_index
    }
    return mismatch_dict


def extract_features(X_train,encoded_length):
    '''
    This function splits one hot encoding into seq encoding and features encoding.
    '''
    seq_only = X_train[:, :encoded_length].astype(np.int8)
    features = X_train[:, encoded_length:]
    return [seq_only,features]




def get_tp_tn(y_test,y_train):
    tp_train = np.count_nonzero(y_train) # count 1's
    tn_train = y_train.size - tp_train # count 0's
    if not tn_train == np.count_nonzero(y_train==0):
        print("error")
    tp_test = np.count_nonzero(y_test) # count 1's
    tn_test = y_test.size - tp_test #count 0's
    if not tn_test == np.count_nonzero(y_test==0):
        print("error")
    tp_ratio = tp_test / (tp_test + tn_test)
    return (tp_ratio,tp_test,tn_test,tp_train,tn_train)



def get_otss_labels_and_indexes(data_frame = None, ot_constrain = 1, mismatch_column = None, bulges_column = None,
                         label_column = None):
    """
    Get the otss labels and indexes.
    
    Sort the indexes in ascending order correspoding to the predictions values on this dataset.
    
    Args:
        data_frame (str -path/ panda df): The data frame containing the data.
        ot_constrain (int): The constrain for the data -
            (1) all off target
            (2) only mismatch
            (3) bulges
        mismatch_column (str): The name of the mismatch column.
        bulges_column (str): The name of the bulges column.
        label_column (str): The name of the label column.
    Returns:
        (np.array, np.array): The indexes and labels of the otss."""
    if isinstance(data_frame, str):
        data_frame = pd.read_csv(data_frame)
    elif not isinstance(data_frame, pd.DataFrame):
        raise ValueError("The data_frame must be a string or a pandas DataFrame.")
    data_frame = return_constrained_data(data_frame,ot_constrain,bulges_column,mismatch_column)

    if "Index" in data_frame.columns:
        indexes = np.array(data_frame['Index'].values)
    else:
        indexes = np.array(data_frame.index)
    labels = np.array(data_frame[label_column])
    sorted = np.argsort(indexes)
    labels = labels[sorted]
    indexes = indexes[sorted]
    return indexes,labels

def keep_indexes_per_guide(data_frame = None, target_column = None, 
                           ot_constrain = 1, mismatch_column = None, bulges_column = None,
                           by_mismatch = False):
    """
    Returns a dictionary of samples indexes for each guide in the data frame.
    
    Args:
        data_frame (str -path/ panda df): The data frame containing the data.
        target_column (str): The name of the target column.
        ot_constrain (int): The constrain for the data - 
            (1) all off target
            (2) only mismatch
            (3) bulges
        mismatch_column (str): The name of the mismatch column.
        bulges_column (str): The name of the bulges column.
        by_mismatch (bool): If True, return indexes by mismatch.
        
    
    Returns:
        (dict): {guide: indexes}
        if by_mismatch is True: {mismatch: {guide: indexes}} 
    """
    if isinstance(data_frame, str):
        data_frame = pd.read_csv(data_frame)
    elif not isinstance(data_frame, pd.DataFrame):
        raise ValueError("The data_frame must be a string or a pandas DataFrame.")
    data_frame = return_constrained_data(data_frame,ot_constrain,bulges_column,mismatch_column)

    guides = data_frame[target_column].unique() # create a unique list of guides
    if "Index" in data_frame.columns:
        if by_mismatch:
            guides_indexes = {mismatch: {guide: data_frame[(data_frame[target_column] == guide) & (data_frame[mismatch_column] == mismatch)]["Index"].values for guide in guides} for mismatch in data_frame[mismatch_column].unique()}
        else:
            guides_indexes = {guide: data_frame[data_frame[target_column] == guide]["Index"].values for guide in guides}
    else:
        if by_mismatch:
            guides_indexes = {mismatch: {guide: data_frame[(data_frame[target_column] == guide) & (data_frame[mismatch_column] == mismatch)].index for guide in guides} for mismatch in data_frame[mismatch_column].unique()}
        else:
            guides_indexes = {guide: data_frame[data_frame[target_column] == guide].index for guide in guides}
    return guides_indexes
