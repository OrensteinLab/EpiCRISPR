# code for labeling positive (off target chromosomal positions by guideseq).
import pandas as pd
import os
import re
import numpy as np
import pybedtools
import pyBigWig
import time
from file_utilities import validate_path, create_folder, remove_dir_recursivly, get_bed_and_bigwig_files, get_ending

ORDERED_COLUMNS_IDENTIFIED_GS = ['chrom','chromStart','chromEnd','Position','Filename','strand','offtarget_sequence','target','realigned_target','Read_count','missmatches','insertion','deletion','bulges','Label']
NORMAL_ORDERED_COLUMNS = ['chrom','chromStart','chromEnd','offtarget_sequence','target','strand','realigned_target','Read_count','missmatches','insertion','deletion','bulges']
#### Identified guideseq preprocessing functions ####

def process_folder(input_folder):
    """
Function gets a folder with guide-seq identified txt files.
It creates a new output folder (if not exists) with csv files filtered by label identified function
folder name created: _labeled
Args:
1. input_folder - folder with identified txt files
------------
Returns: None
Runs: identified_to_csv function on each txt file in the folder
    """
    label_output_folder = input_folder + '_labeled'
    create_folder(label_output_folder)
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            txt_file_path = os.path.join(input_folder, filename)
            filter_identified_guideseqs_to_csv(txt_file_path, label_output_folder)

def filter_identified_guideseqs_to_csv(input_path,output_path):
    
    """
    Filter GUIDE-seq output file (identified.txt) based on the BED_site_name colum.
    If the BED_site_name colom is not null it represents a detected off-target site.

    Function:
        1. Filter by bed_site_column and up to 6 mismatches.
        2. Get Off target sites with missmatches only/ bulges and missmatches.
        3. Extract file name - expriment name.
        4. Create csv file in the output path named with expirement name + "_label"

    Columns kept are: 
    chrom, chromStart, chromEnd, Position, Filename, strand, offtarget_sequence, target,
    realigned_target, Read_count, missmatches, insertions, deletions, bulges, Label

    Args:
        input_path (str): path to the identified.txt file
        output_path (str): path to the output folder
    ------------
    Returns: None
    Saves: csv file in the output folder with the columns mentioned above.
   """

    identified_file = pd.read_csv(input_path,sep="\t",encoding='latin-1',on_bad_lines='skip')
    # 1. filter by bed_site - accautal off target site
    valid_otss = identified_file.dropna(subset=['BED_Site_Name'])
    # 2. filter by missmatch count <=6
    valid_otss = valid_otss[
        (valid_otss['Site_SubstitutionsOnly.NumSubstitutions']  <= 6) |
        (valid_otss['Site_GapsAllowed.Substitutions'] <= 6)  ]
    mismatch_df = get_ots_missmatch_only(valid_otss) # Get mismatch only
    bulge_df = get_ots_bulges_nd_mismatches(valid_otss) # Get bulges
    merged_df = pd.concat([mismatch_df,bulge_df],axis=0,ignore_index=True) # Merged both pds
    # 3. get expirement name
    exp_name = merged_df.iloc[0]['Filename']
    exp_name = exp_name.replace('.sam','')
    output_filename = exp_name + "_labeled.csv"
    output_path = os.path.join(output_path,output_filename)
    # 5. create csv file
    merged_df.to_csv(output_path, index=False)
    print("created {} file in folder: {}".format(output_filename,output_path))

def get_ots_missmatch_only(guideseq_identified_data_frame):
    """
    This function accepets GUIDE-seq output file (idenetified) data frame
    It extract Off target sites with missmatches only
    Setting insertions, deletions, bulges to 0.
    
    Args:
        guideseq_identified_data_frame (data frame): data frame with guide-seq identified data
    
    Returns: 
        mismatch_df (data frame): data frame with off-target sites with missmatches only
    """
    # Drop rows without ots seq in mismatch
    guideseq_identified_data_frame = guideseq_identified_data_frame.dropna(subset=['Site_SubstitutionsOnly.Sequence'])
    columns = {'WindowChromosome':'chrom' ,'Site_SubstitutionsOnly.Start':'chromStart','Site_SubstitutionsOnly.End' : 'chromEnd',
               'Position':'Position','Filename':'Filename','Site_SubstitutionsOnly.Strand':'strand',
               'Site_SubstitutionsOnly.Sequence':'offtarget_sequence','TargetSequence':'target',
               'RealignedTargetSequence':'realigned_target','bi.sum.mi':'Read_count',
               'Site_SubstitutionsOnly.NumSubstitutions':'missmatches'}
    mismatch_df = guideseq_identified_data_frame[columns.keys()]
    mismatch_df.rename(columns=columns,inplace=True)
    mismatch_df[["insertion","deletion","bulges"]] = 0
    mismatch_df['Label'] = 1
    return mismatch_df

def get_ots_bulges_nd_mismatches(guideseq_identified_data_frame):
    """
    This function accepets GUIDE-seq output file (idenetified) data frame
    It extract Off target sites with bulges and mismatches
    Args:
        guideseq_identified_data_frame (data frame): with guide-seq identified data
    ------------
    Returns: 
        bulge_df (data frame): with off-target sites with bulges and mismatches
    """
    columns = {'WindowChromosome':'chrom' ,'Site_GapsAllowed.Start':'chromStart','Site_GapsAllowed.End':'chromEnd',
               'Position':'Position','Filename':'Filename','Site_GapsAllowed.Strand':'strand',
               'Site_GapsAllowed.Sequence':'offtarget_sequence','TargetSequence':'target',
               'RealignedTargetSequence':'realigned_target','bi.sum.mi':'Read_count',
               'Site_GapsAllowed.Substitutions':'missmatches',
               'Site_GapsAllowed.Insertions':'insertion','Site_GapsAllowed.Deletions': 'deletion'}
    # Drop rows without bulges
    guideseq_identified_data_frame = guideseq_identified_data_frame.dropna(subset=['Site_GapsAllowed.Sequence'])
    bulge_df = guideseq_identified_data_frame[columns.keys()]
    bulge_df.rename(columns=columns,inplace=True)
    bulge_df["bulges"] = bulge_df["insertion"] + bulge_df["deletion"]
    bulge_df['Label'] = 1
    return bulge_df



def merge_positives(folder_path, n_duplicates, file_ending, output_folder_name):
        
    """
    Looks for multiple duplicates of the same expriment and merge their data.
    Utilizes the mergning function.
    NOTE: Mergning will be the summation (aggregation) of the read counts for the same off-target sites!
    NOTE: all files should have the same ending, for example: -D(n)_labeled
    Function gets each file by iterating on the number of duplicates and changing the file ending.
    For each 2 or more duplicates concatanate the data.
    
    Args:
        folder_path (str): path to the folder with the labeled files
        n_duplicates (int): number of duplicates for each expriment
        file_ending (str): ending of the file name
        output_folder_name (str): name of the output folder
     ------------
     Returns: None
     Saves: csv files in the output folder with the merged data
     """
    assert n_duplicates > 1, f"duplicates should be more then 1 per expriment, got: {n_duplicates}"
    # more then 1 duplicate
    file_names = os.listdir(folder_path)
    # create pattern of xxx-D(n) + suffix
    pattern = r"(.+?-D)\d+" + re.escape(file_ending)
    # create one string from all the file names and find the matching pattern.
    file_names = ''.join(file_names)
    mathces = re.findall(pattern,file_names)
    # use a set to create a unique value for n duplicates
    unique = list(set(mathces))
    print('before mergning: ',len(mathces))
    print('after mergning: ',len(unique))
    # get tuple list - df, name
    final_file_list = mergning(unique,n_duplicates,file_ending,folder_path)   
    # create folder for output combined expriment:
    # remove .csv\.txt from ending.
    file_ending = file_ending.rsplit('.', 1)[0]
    # create folder
    output_path = os.path.join(os.path.dirname(folder_path), output_folder_name)
    create_folder(output_path)
    # create csv file from eached grouped df.
    for tuple in final_file_list:
        name = tuple[1] + '.csv'
        temp_output_path = os.path.join(output_path,name)
        tuple[0].to_csv(temp_output_path,index=False)
    
        


def mergning(files, n_duplicates, file_ending, folder_path):
    """
    This function gets a list of files and merge them togther summing the read count.
    It merge n duplicates for each file.
    Args:
    1. files - list of files to merge
    2. n - number of duplicates for each file
    3. file_ending - ending of the file name
    4. folder_path - path to the folder with the files
    ------------
    Returns: final_file_list - list of tuples with the merged data frames and the file name
    """ 
    assert n_duplicates > 1, f"duplicates should be more then 1 per expriment, got: {n_duplicates}"  
    # more then 1 duplicate per file
    final_file_list = []
    grouping_columns = ['chrom', 'chromStart', 'chromEnd','strand', 'offtarget_sequence', 'target', 'realigned_target','missmatches','insertion', 'deletion','bulges' ]
    for file_name in files:
        # create n duplicates file list
        n_file_list =[]
        for count in range(int(n_duplicates)):
            # create string for matching duplicate file
            temp_file_name = file_name + str(count+1) + file_ending 
            input_path = os.path.join(folder_path,f'{temp_file_name}')
            # append df into the list
            n_file_list.append(pd.read_csv(input_path,sep=",",encoding='latin-1',on_bad_lines='skip'))
        # n_file_list have all duplicates in it, merge them:
        merged_df = pd.concat(n_file_list, axis=0, ignore_index=True)
        print ('before grouping: ',len(merged_df))
        # group by position, number of missmatches, and siteseq. sum the bi - reads
        grouped_df = merged_df.groupby(grouping_columns).agg({
    'Position': 'first', 'Filename': 'first', 'Read_count': 'sum', 
    'Label': 'first'  
}).reset_index()
        
        print ('after grouping: ',len(grouped_df))
        grouped_df = grouped_df[ORDERED_COLUMNS_IDENTIFIED_GS]
        # append df to final list
        final_file_list.append((grouped_df,file_name)) 
    return final_file_list

def concat_data_frames(folder_path = None, first_df = None, second_df = None):
    """Function concat data frames vertically.
    If folder path is given it will concat all the csv files in the folder.
    Else it will concat two data frames given in the first and second paths.
    Args:
    1. folder_path - path to the folder with the csv files
    2. first_df - first data frame to concat
    3. second_df - second data frame to concat
    ------------
    Returns: data frame with the concatenated data frames"""
    if folder_path:
        files = os.listdir(folder_path)
        data_frames = []
        for file in files:
            file_path = os.path.join(folder_path,file)
            data_frames.append(pd.read_csv(file_path))
        return pd.concat(data_frames, axis = 0, ignore_index = True)
    elif (not first_df.empty) and (not second_df.empty):
        return pd.concat([first_df,second_df], axis = 0, ignore_index = True)
    else: 
        raise ValueError("No data frames given to concat.")

def preprocess_identified_files(folder_path, idenetified_folder_name, n_duplicates, output_data_name, erase_sub_dir = False):
    '''Function preprocess identified files in a given folder:
    1. Turn idenetified single files into csv files using identified_to_csv function.
    2. Merge the same experiments with multiple duplicates using merge_positives function.
    3. Concat all the data frames into one data frame.
    4. Save the data frame in a csv file.
    5. If erase_sub_dir is True, the function will remove the sub directories.
    Args:
    1. folder_path - path to the folder with the identified files
    2. idenetified_folder_name - name of the folder with the identified files
    3. n_duplicates - number of duplicates for each experiment
    4. output_data_name - name of the output data file
    5. erase_sub_dir - boolean, if True the function will remove the sub directories
    ------------
    Returns: None'''
    if validate_path(folder_path):
        identified_folder_path = os.path.join(folder_path, idenetified_folder_name)
        if validate_path(identified_folder_path):
            process_folder(identified_folder_path)
            labled_folder_path  = identified_folder_path + "_labeled"
            if not validate_path(labled_folder_path): # validate label folder was created
                raise RuntimeError(f"No labeled folder created need to process: {idenetified_folder_name} agian!")
            if n_duplicates > 1: # no duplicates need to be merged
                out_put_merged_folder_name = "merged_experiments"
                merge_positives(labled_folder_path, n_duplicates, '_labeled.csv', out_put_merged_folder_name)
                merged_folder_path = os.path.join(folder_path, out_put_merged_folder_name)
                if not validate_path(merged_folder_path): # validate merged folder was created
                    raise RuntimeError(f"No merged folder created need to process: {idenetified_folder_name} agian!")
            else : 
                merged_folder_path = labled_folder_path
            
            if not output_data_name.endswith('.csv'): # add .csv to output file if needed.
                output_data_name = output_data_name + '.csv'
            merged_df = concat_data_frames(folder_path = merged_folder_path) # concat all the data frames
            merged_df.to_csv(os.path.join(folder_path, output_data_name), index = False) # save merged data frame
            if erase_sub_dir:
                remove_dir_recursivly(labled_folder_path) # remove labeled folder
                if n_duplicates > 1:
                    remove_dir_recursivly(merged_folder_path)
    print(f"Preprocessing of {idenetified_folder_name} is done. Merged data saved in {output_data_name} file.")

#####################################################################      

### Negative Labeling functions ###
'''Example input file (DNA bulge size 2, RNA bulge size 1):
/var/chromosomes/human_hg38
NNNNNNNNNNNNNNNNNNNNNRG 2 1
GGCCGACCTGTCGCTGACGCNNN 5'''
# inputpath for folder containing indentified files.
# extract guide sequence and create intput file out of it.
def create_csofinder_input_by_identified(input_path,output_path):
    # get targetseq = the guide rna used
    identified_file = pd.read_csv(input_path,sep="\t",encoding='latin-1',on_bad_lines='skip')
    seq = identified_file.iloc[0]['TargetSequence']
    # guideseq size 'N' string
    n_string = 'N' * len(seq)
    exp_name = seq + "_" + identified_file.iloc[0]['Filename']
    output_filename = f"{seq}_input.txt"
    output_path = os.path.join(output_path,output_filename)
    if not os.path.exists(output_path):
        with open(output_path, 'w') as txt_file:
            txt_file.write("/home/labhendel/Documents/cas-offinder_linux_x86-64/hg38noalt\n")
            txt_file.write(n_string + "\n")
            txt_file.write(seq + ' 6')

'''function to read a table and get the unique target grna to create an input text file
for casofinder
table - all data
target_colmun - column to get the data
outputname - name of folder to create in the scripts folder'''
def create_cas_ofinder_inputs(table , target_column, output_name, path_for_casofiner):
    table = pd.read_excel(table)
    output_path = os.getcwd() # get working dir path
    output_path = os.path.join(output_path,output_name) # add folder name to it
    create_folder(output_path)
    try:
        guides = set(table[target_column]) # create a set (unquie) guides
    except KeyError as e:
        print(f"no column: {target_column} in data set, make sure you entered the right one.")
        exit(0)
    casofinder_path = get_casofinder_path(path_for_casofiner)
    casofinder_path = casofinder_path + "\n"
    one_file_path = os.path.join(output_path,f"one_input.txt")
    n_string = 'N' * 23
    with open(one_file_path,'w') as file:
        file.write(casofinder_path)
        file.write(n_string + "\n")
    for guide in guides:
        n_string = 'N' * len(guide)
        output_filename = f"{guide}_input.txt"
        temp_path = os.path.join(output_path,output_filename)
        with open(temp_path, 'w') as txt_file:
            txt_file.write(casofinder_path)
            txt_file.write(n_string + "\n")
            txt_file.write(guide + ' 6')
        with open(one_file_path,'a') as txt:
            txt.write(guide + ' 6\n')
    
'''if genome == hg19,hg38 set own path else keep others.'''
def get_casofinder_path(genome):
    path = genome
    if genome == "hg19":   
        path = "/home/labhendel/Documents/cas-offinder_linux_x86-64/hg19"
    elif genome == "hg38":
        path = "/home/labhendel/Documents/cas-offinder_linux_x86-64/hg38noalt"
    else : print(f"no genome file exists for: {genome}")
    return path
''' function to return a dict with the lengths of the guides'''
def guides_langth(guide_set):
    lengths = {}
    for guide in guide_set:
        length_g = len(guide) # length of guide
        if length_g in lengths.keys():
            lengths[length_g] += 1
        else:   lengths[length_g] = 1
    return lengths
     



### Cas-offinder creation and negative labeling functions ###

def transform_casofiner_into_csv(path_to_txt):
    columns = ['target','Chrinfo','chromStart','offtarget_sequence','strand','missmatches']
    output_path = path_to_txt.replace(".txt",".csv")
    try:
        negative_file = pd.read_csv(path_to_txt,sep="\t",encoding='latin-1',on_bad_lines='skip')
    except pd.errors.EmptyDataError as e:
        print(f"{path_to_txt}, is empty")
        exit(0)
    negative_file.columns = columns
    print(negative_file.head(5))
    print(negative_file.info())
    potential_ots = add_info_to_casofinder_file(negative_file)
    potential_ots.to_csv(output_path,sep=',',index=False)

def add_info_to_casofinder_file(data):

    '''add info to casofinder output file:
    1. Extract chrinfo from Chrinfo column.
    2. Set chromend to be chrom start + len of the ots
    3. Set Read_count, insertions, deletions, bulges, Position to 0
    4. Set realigned target to be the same as target
    5. Set Filename to empty string
    6. Upcase offtarget_sequence'''

    data['chrinfo_extracted'] = data['Chrinfo'].str.extract(r'(chr[^\s]+)') # extract chr
    data = data.rename(columns={ 'chrinfo_extracted':'chrom'}) 
    data = data.drop('Chrinfo',axis=1) # drop unwanted chrinfo
    print(data.head(5))
    print(data.info())
    data['chromEnd'] = data['chromStart'] + data['offtarget_sequence'].str.len() 
    print(data.head(5))
    print(data.info())
    data['Read_count'] = data['insertion'] = data['deletion'] = data['bulges'] = data['Position'] = data["Label"] = 0 
    data['realigned_target'] = data['target'] # set realigned target to target
    data['Filename'] = '' # set filename to empty
   
    data = upper_case_series(data,colum_name="offtarget_sequence") # upcase all oftarget in data
    print(data.head(5))
    print(data.info())
    new_data = pd.DataFrame(columns=ORDERED_COLUMNS_IDENTIFIED_GS) # create new data with spesific columns
    for column in data.columns:
        new_data[column] = data[column] # set data in the new df from the old one
    print(new_data.head(5))
    print(new_data.info())
    return new_data



def upper_case_series(data, colum_name):
    '''function to upcase a series in a data frame'''
    values_list = data[colum_name].values
    upper_values = [val.upper() for val in values_list]
    data[colum_name] = upper_values
    return data


### Merge positive data set with negative data set ###

def merge_positive_negative(positives, negatives, output_name, output_path, target_column, remove_bulges):
    '''Function to merge positive data set and negative data set:
    1. Add label column to both data sets with 1 for positive and 0 for negative.
    2. Remove from the negative data all the guides that presnted in the positive data.
    3. Group by columns and aggregate Read_count
    4. Print the amount of data points in the merged data set.
    5. Save the data set'''    
    positives = read_to_df_add_label(positives,1) # set 1 for guide seq
    negatives = read_to_df_add_label(negatives,0,True) # set 0 for changeseq
    negatives = remove_unmatching_guides(positive_data=positives,target_column=target_column,negative_data=negatives)
    grouping_columns = ['chrom', 'chromStart', 'chromEnd','strand', 'offtarget_sequence', 'target','missmatches','insertion', 'deletion','bulges' ]

    if remove_bulges:
        negatives = negatives[negatives['bulges'] == 0]
        positives = positives[positives['bulges'] == 0]
        grouping_columns = ['chrom', 'chromStart', 'chromEnd','strand', 'offtarget_sequence', 'target','missmatches' ]
    
    positives, gs_duplicates = drop_by_colmuns(positives,grouping_columns,"first")
    negatives,cs_duplicates = drop_by_colmuns(negatives,grouping_columns,"first")
    neg_length = len(negatives)
    pos_length = len(positives)
    merged_data = pd.concat([positives,negatives])
    print(f"data points should be: Merged: {len(merged_data)},Pos + Neg: {pos_length+ neg_length}")
    merged_data,md_duplicates = drop_by_colmuns(merged_data,['chrom', 'chromStart', 'chromEnd','strand', 'offtarget_sequence', 'target'],"first")
     
    
    
    
    count_ones = sum(merged_data["Label"] > 0)
    count_zeros = sum(merged_data["Label"]==0)
    print(f"Positives: {pos_length}, By label: {count_ones}")
    print(f"Negatives: {neg_length} - {count_ones} = {neg_length - count_ones + (count_ones-md_duplicates)}, label: {count_zeros} ")
    print(merged_data.columns)
    output_path = f"{os.path.join(output_path,output_name)}.csv"
    merged_data.reset_index(drop=True,inplace=True)
    merged_data["Index"] = merged_data.index
    merged_data.to_csv(output_path,index=False)

def remove_unmatching_guides(positive_data, target_column, negative_data):
    '''Function to remove from the negative data all the guides that presnted in the positive data.
    1. Create a unuiqe set of guides from positive guides and negative guides.
    2. Keep only the guides that presnted in the negative but not positive set.
    3. Remove the guides from the negative set'''
    # create a unuiqe set of guides from positive guides and negative guides
    positive_guides = set(positive_data[target_column])
    negative_guides = set(negative_data[target_column])
    # keep only the guides that presnted in the negative but not positive set
    diffrence_set = negative_guides - positive_guides
    intersect = negative_guides.intersection(positive_guides)
    print(f'intersect: {len(intersect)}, length pos: {len(positive_guides)}, length negative: {len(negative_guides)},\ndifrence: {len(diffrence_set)}')
   
    before = len(negative_data)
    for guide in diffrence_set:
        negative_data = negative_data[negative_data[target_column]!=guide]
        after = len(negative_data)
        guide_seq_amount = len(positive_data[positive_data[target_column]==guide])
        removed_amount = before-after
        print(f'{removed_amount} rows were removed')
        before =  after
    return negative_data

def drop_by_colmuns(data,columns,keep):
    '''drop duplicates by columns return data and amount of duplicates'''
    length_before = len(data)
    data = data.drop_duplicates(subset=columns,keep=keep)
    length_after = len(data)
    return data,length_before-length_after  

def read_to_df_add_label(path , label, if_negative=False):
    '''Add label column to data frame
    Column will be named Label and will be set to the label value
    If negative is True, the function will read the data as negative data and set the label to 0'''
    table = pd.read_csv(path, sep=",",encoding='latin-1',on_bad_lines='skip')
    columns_before = table.columns
    if (not "Label" in table.columns):
        table["Label"] = label
    columns_after = table.columns
    print(f"columns before: {columns_before}\nColumns after: {columns_after}")
    print(table.head(5))
    return table

### Epigenetic assignment ###
def get_bed_columns(bedtool, columns_dict = {4:"score",6:"fold_enrichemnt",7:"logp",8:"logq"}):
    '''This function accepts a bedtool and returns the columns of the bedtool as a list.
    The function add the score, fold_enrichement, logp and logq columns to the list.
    Args:
    1. bedtool - a bedtool object
    2. columns_dict - a dictionary with the columns names {column_number: column_name} - {5:score...} (columns are 0 based)
    defualt is 4-score,6-fold,7logp, 8logq
    ------------
    Returns: a list of columns'''
    # Get the first interval
    first_interval = next(iter(bedtool))
    # Get the number of fields (columns) for the first interval
    num_fields = len(first_interval.fields)
    columns = [i for i in range(1,num_fields+1)]
    if num_fields <= 3:
        # only chr,start,end
        return columns
    elif columns_dict:
        for num, column_name in columns_dict.items():
            if num in range(num_fields):
                columns[num] = column_name
            else:
                break
    
    return columns
def intersect_with_epigentics_BED(whole_data,epigentic_data,if_strand, chrom_column_names=["chrom","chromStart","chromEnd"]):
    """
    Intersects off-target data with epigenetic from BED.
    Utilize pybedtools to intersect the data with -wb param to keep both datas information.
    Args:
        whole_data (pd.DataFrame): data frame with off-target data
        epigentic_data (str): bed file with epigenetic data
        if_strand (bool): if True, the function will intersect by strand
    Returns:
        whole_data (pd.DataFrame): data frame with off-target data
        intersection_df_wa (pd.DataFrame): data frame with intersection data
    """
    whole_data[chrom_column_names[1]] = whole_data[chrom_column_names[1]].astype(int)
    whole_data[chrom_column_names[2]] = whole_data[chrom_column_names[2]].astype(int)

    whole_data_bed = pybedtools.BedTool.from_dataframe(whole_data)
    epigentic_data = pybedtools.BedTool(epigentic_data)
    # set columns to data columns + bed columns (5th of bed is score)
    columns = whole_data.columns.tolist() 
    columns = columns +  get_bed_columns(epigentic_data)
    intersection_wa = whole_data_bed.intersect(epigentic_data,wb=True,s=if_strand) # wb keep both information
    intersection_df_wa = intersection_wa.to_dataframe(header=None,names = columns)
    return whole_data,intersection_df_wa
    
def intersect_and_assign_bigwig_data(off_target_data, bigwig_path, file_name, chrom_column_names=["chrom","chromStart","chromEnd"]):
    """
    Compute average signal per region in a DataFrame using a BigWig file.
    The result Series matches the original DataFrame's index for direct assignment.
    Assigns the average series to the original dataframe and returns it.

    Args:
        off_target_data (pd.DataFrame): Must contain 'chrom', 'chromStart', 'chromEnd'.
        bigwig_path (str): Path to the BigWig file.
        file_name (str): String of the epigenetic file
        chrom_column_names (list): List of three strings, chr, start, end

    Returns:
        pd.DataFrame: DataFrame assigned with average bigwig values.
    """
    # Preserve original index
    original_index = off_target_data.Index
    # Sort for efficient BigWig access
    off_target_data = off_target_data.rename(columns={
    chrom_column_names[0]: "chrom",
    chrom_column_names[1]: "chromStart",
    chrom_column_names[2]: "chromEnd"
})
    
    df_sorted = off_target_data.sort_values(by=["chrom", "chromStart"]).reset_index()
    df_sorted["chromStart"] = df_sorted["chromStart"].astype(int)
    df_sorted["chromEnd"] = df_sorted["chromEnd"].astype(int)
    # Open BigWig
    bw = pyBigWig.open(bigwig_path)
    averages = []
    for row in df_sorted.itertuples(index=False):
        idx, chromosome, start, end = row.Index, row.chrom, (row.chromStart), (row.chromEnd)
        if chromosome in bw.chroms():
            avg = bw.stats(chromosome, start, end, type="mean")[0]
            avg = 0.0 if avg is None else avg
        else:
            avg = 0.0
        averages.append((idx, avg))

    bw.close()
    # Convert to Series with original index
    avg_series = pd.Series({idx: avg for idx, avg in averages})
    off_target_data[file_name] = off_target_data['Index'].map(avg_series)
    return off_target_data

    
def assign_epigenetics(off_target_data,intersection,file_ending,score_type_dict={"binary":True}):
    '''
    This function assign epigenetic data to the off-target data frame.
    It extracts the epigenetic type - i.e. chromatin accesesibility, histone modification, etc.
    To that it adds the epigenetic mark itself - i.e. H3K4me3, H3K27ac, etc.
    To each combination of type and mark it assigns a binary column by defualt.
    If other score types are given it will assign them as well.
    An example to set binary value and the fold enrichement value:
    score_type_dict = {"binary":True,"score":False,"fold_enrichemnt":True,"log_fold_enrichemnt":False,"logp":False,"logq":False}
    Args:
    1. off_target_data - data frame with off-target data
    2. intersection - data frame with intersection data
    3. file_ending - the ending of the bed file - i.e. H3K4me3, H3K27ac, etc.
    4. chrom_type - the type of the epigenetic data - i.e. chromatin accesesibility, histone modification, etc. 
    5. score_type_dict - a dictionary with the score types and the values to assign to the columns
    ------------
    Returns: off_target_data - data frame with the epigenetic data assigned.
    '''
    
    columns_dict = {key: f'{file_ending}_{key}' for key in score_type_dict.keys()} # set columns names
    # add columns to the off-target data
    for column_name in columns_dict.values():
        off_target_data[column_name] = 0
    # Set a dictionary with the columns names and the intersect values
    values_dict = {key: None for key in score_type_dict.keys()} # Intersect columns
    values_dict.pop("binary") # remove binary from the dict
    log_gold_flag = False
    for key in values_dict.keys():
        if key == "log_fold_enrichemnt": # if log fold enrichemnt need to be set set the flag.
            log_gold_flag = True
            continue
        values_dict[key] = intersection[key].tolist()
    # set log fold enrichemnt
    if log_gold_flag and "fold_enrichemnt" in values_dict.keys():
        log_fold_vals = np.array(values_dict["fold_enrichemnt"])
        log_fold_vals = np.log(log_fold_vals)
        values_dict["log_fold_enrichemnt"] = log_fold_vals.tolist()
    
    if not intersection.empty:
        try:
            print(f"Assigning the next epigenetic values: {columns_dict.keys()}")
            print("OT data before assignment:\n",off_target_data.head(5))
            time.sleep(1)
            # Assign intersection indexes in the off-target data with 1 for binary column and values to other columns
            for key in score_type_dict.keys():
                if key == "binary":
                    off_target_data.loc[off_target_data["Index"].isin(intersection["Index"]), columns_dict["binary"]] = 1
                else :
                    off_target_data.loc[off_target_data["Index"].isin(intersection["Index"]), columns_dict[key]] = values_dict[key]
            print("OT data after assignment:\n",off_target_data.head(5))
        except KeyError as e:
              print(off_target_data,': file has no intersections output will be with 0')
    ## Print statistics   
    labeled_epig_1 = sum(off_target_data[columns_dict["binary"]]==1)
    labeled_epig_0 =  sum(off_target_data[columns_dict["binary"]]==0)
    if (labeled_epig_1 + labeled_epig_0) != len(off_target_data):
        raise RuntimeError("The amount of labeled epigenetics is not equal to the amount of data")
    print(f"length of intersect: {len(intersection)}, amount of labled epigenetics: {labeled_epig_1}")
    active_labeled = sum((off_target_data[columns_dict['binary']]==1) & (off_target_data['Label']>0))
    total_actives = sum(off_target_data['Label'] > 0)
    print(f'total actives data points: {total_actives}, out of them: {active_labeled} marked with {file_ending}')
    print(f'length of data: {len(off_target_data)}, 0: {labeled_epig_0}, 1+0: {labeled_epig_1 + labeled_epig_0}')
    return off_target_data

def run_intersection(merged_data_path,epigenetic_folder,if_update, chrom_column_names=["chrom","chromStart","chromEnd"]):
    """
    Intersect off-target data with epigenetic data given in bed files or bigwig files.

    It will intersect the data with each bed file in the folder and assign the epigenetic data to the off-target data.
    If `if_update` is True, the function will update the existing data with the new epigenetic data, i.e, it will find
    epigenetic files that not exists in the data and will intersect only their data.

    Bed files assignment will be binary.
    BigWig assignment will be average nucleotide values over the off-target site.

    Args:
        merged_data_path (str): Path to the merged off-target data.
        epigenetic_folder (str): Path to the folder containing the epigenetic data in BED or Bigwig format.
        if_update (bool): If True, the function will update the existing data with the new epigenetic data.
        chrom_column_names (list): List of column names for chromosome, start, and end positions. Default is ["chrom","chromStart","chromEnd"].

    Returns:
        None

    Description:
        This function saves the new data frame with the epigenetic data in the same path as the merged data, 
        with the file ending "_withEpigenetic.csv".

    """
    '''
    example:
    run_intersection("Off-Target-data-proccessing/Data/TrueOT/Refined_TrueOT_shapiro_park.csv",
                     "Off-Target-data-proccessing/Epigenetics/HSPC",False)
                     '''
    data = pd.read_csv(merged_data_path)
    data = order_data_column_for_intersection(data,chrom_column_names)
    if not "Index" in data.columns:
        data["Index"] = data.index
    bed_paths,bigwig_paths = get_bed_and_bigwig_files(epigenetic_folder)
    
    if "withEpigenetic" not in merged_data_path:
        new_data_name = merged_data_path.replace(".csv","")
        new_data_name += "_withEpigenetic.csv"
    else: new_data_name = merged_data_path
    
    if if_update:
        bed_paths = remove_exsiting_epigenetics(data,bed_paths,True) # remove exsiting epigenetics
        bigwig_paths = remove_exsiting_epigenetics(data,bigwig_paths,True)
        new_data_name = merged_data_path # update exsiting data
    print('Assigning epigenetic data in BED format')
    for bed_path in bed_paths:
        data,intersect = intersect_with_epigentics_BED(data,epigentic_data=bed_path,if_strand=False, chrom_column_names=chrom_column_names)
        data = assign_epigenetics(off_target_data=data,intersection=intersect,file_ending=get_ending(bed_path))
    print('Assigning epigenetic data in BigWig format')
    for bigwig_path in bigwig_paths:
        data = intersect_and_assign_bigwig_data(off_target_data=data, bigwig_path= bigwig_path, 
                                                file_name=get_ending(bigwig_path), chrom_column_names=chrom_column_names)
    data.to_csv(new_data_name,index=False)

def remove_exsiting_epigenetics(data,bed_type_nd_paths,full_match=False):
    '''This function accpets data frame and list of tuples where
    First element is epigenetic type - Chrom, methylation, etc
    Second element is the epigeneitc mark itself - h3k4me3...
    Removes from the tuple list any epigenetic mark that already exists in the data frame.
    if full_match is True, the function will remove only the epigenetic marks that are fully matched in the data frame.
    if full_match is False, the function will remove any epigenetic mark that is partially matched in the data frame.
    '''
    new_chrom_information = [] # assign new list to keep only new data
    for bed_paths in bed_type_nd_paths:
        paths_list = [] 
        for bed_path in bed_paths:
            file_ending = get_ending(bed_path) 
            if full_match:
                column_to_check = f'^{file_ending}$'
            else : column_to_check = f'^{file_ending}'
            if any(re.match(column_to_check, column) for column in data.columns):
                continue
            else : paths_list.append(bed_path)
            
        new_chrom_information.append((paths_list))
    return new_chrom_information
def order_data_column_for_intersection(data,chrom_columns):
    '''
    This function orders the columns of the data frame for the intersection.
    it sets the chrom, chromstart and chromend columns first and the rest of the columns after them.
    Args:
    1. data - data frame with the data
    2. chrom_columns - list of the chrom columns - chrom,start,end
    ------------
    Returns: data frame with the ordered columns
    '''
    data_columns = data.columns.tolist()
    data_columns = chrom_columns + [column for column in data_columns if column not in chrom_columns] # set the chrom columns first
    return data[data_columns]
### Multiple sources data merging ###
def combine_data_from_diffrenet_studies(studies_list, target_column, with_intersecting = False):
    '''This function combines data from different studies.
    It assums the data is already divided to positives and negatives or read count is assigned.
    Further more: NOTE BOTH DATA FRAMES SHOULD HAVE CORRESPONDING COLUMNS
    Args:
    1. studies_list - list of data frames with the data from different studies.
    2. target_column - column with the target sequence.
    3. with_intersecting - boolean, if True the function will merge the data frame and keep intersecting guides.
    if False the function will only concat the data frames by removing the intersecting guides from both data frames.
    ------------
    Returns: data frame with the combined data.'''
    if not studies_list:
        raise ValueError("No data frames given to combine.")
    guides_set_list = []
    for data in studies_list:
        # if not target_column in any(data.columns):
        #     raise ValueError(f"No target column in the data: {data.info()}")
        guides_set_list.append(set(data[target_column]))
    intersecting_guides = set.intersection(*guides_set_list)
    all_guides = set.union(*guides_set_list)
    diffrence_guides = all_guides - intersecting_guides
    # remove intersecting guides from data frames
    new_data_list = []
    for data in studies_list:
        data = data[~data[target_column].isin(intersecting_guides)]
        new_data_list.append(data)
    # validate all guides is equal to the guides of the data frames
    merged_data = pd.concat(new_data_list, axis = 0, ignore_index = True)
    merged_data_guides = set(merged_data[target_column])
    if merged_data_guides != diffrence_guides:
        raise ValueError("Not all guides are in the data frames.")
    return merged_data



#############################################
## Remove unwanted examples ##
def remove_unwanted_samples(dataframe,column,value,if_treshold =False,treshold_sign=None):
    """
    This function removews unwanted samples from the data frame
    if_treshold is True, the function will remove samples with value above or below the treshold given the threshold sign
    if no treshold if the value exists in the column it will be removed
    Args:
    dataframe (pd.DataFrame): data frame with the data
    column(str): column name to remove the samples from
    value (int,str) - value to remove
    if_treshold (bool): if True the function will remove samples above or below the treshold
    treshold_sign (str): sign of the treshold - <, >, <=, >=, ==
    ------------
    Returns: data frame with the removed samples.
    """
    if if_treshold:
        if not treshold_sign:
            raise ValueError("No treshold sign given.")
        if not treshold_sign in ["<",">","<=",">=","=="]:
            raise ValueError(f"Invalid treshold sign: {treshold_sign}")
        else :
            filtered_df =  dataframe[dataframe[column].apply(lambda x: eval(f'{x} {treshold_sign} {value}'))]
            print(f"Filtered {len(dataframe)-len(filtered_df)} samples with {column} {treshold_sign} {value}")
            return filtered_df
    else:
        filtered_df = dataframe[~dataframe[column].str.contains(f"[{value}]", regex=True)]
        print(f"Filtered {len(dataframe)-len(filtered_df)} samples with {column} {value}")
        return filtered_df
 
def return_constrained_data(data_frame, off_target_constraints, bulge_column = None, mismatch_column = None):
    '''
    This function takes an OT data frame and returns a data frame after applying given constraint
    Args:
    data_frame: data frame with the data
    off_target_constraints (int): the constraint to apply. 1- no constrains, 2- only mismatches, 3- only bulges
    bulge_column(str): column with the bulges
    mismatch_column(str): column with the mismatches
    
    Returns: data frame with the constrained data.
    '''
    if off_target_constraints == 1: # No constraints
        return data_frame
    elif off_target_constraints == 2: # Mismatch only remove all OTS with bulges - bulge < 1
        if bulge_column:
            return remove_unwanted_samples(data_frame,bulge_column,0,True, "==")
        else:
            raise ValueError("No bulge column given.")
    elif off_target_constraints == 3: # Bulge only - get all OTS with bulges - bulge >0
        if bulge_column is None or mismatch_column is None:
            raise ValueError("No bulge or mismatch column given.")
        with_bulges = remove_unwanted_samples(data_frame,bulge_column,0,True, ">")
        return remove_unwanted_samples(with_bulges,mismatch_column,0,True, "==")


# def split_all_guide_seq_to(data_path, output_path, output_name):
#     data = pd.read_csv(data_path)
    
#     data.rename(columns={'Align.sgRNA': 'realigned_target',
#        'distance':'missmatches', 'Align.#Bulges':'bulges', 'Label':'Read_count'}, inplace=True)
    
#     print(f'len before: {len(data)}')
#     data.drop_duplicates(subset=['offtarget_sequence','target','chromStart','chromEnd'],keep='first',inplace=True)
#     print(f'len after: {len(data)}')
#     data.to_csv(os.path.join(output_path,f'{output_name}.csv'),index=False)
def ot_data_frame_to_convention_columns(data_frame=None, target_column=None, off_target_column=None, chrom=None, realigned_target=None,
                                             chrom_start=None, chrom_end=None, strand=None, bulges=None, insertions=None,
                                               deletions=None,  label=None, read_count=None, mismatches=None):
    '''
    This function takes a data frame with OT data and transform it columns to the convention columns.
    So the new data frame will have the same columns like all data frames.
    true_ot = pd.read_csv("Off-Target-data-proccessing/Data/TrueOT/Refined_TrueOT.csv")
    true_ot = ot_data_frame_to_convention_columns(data_frame=true_ot, target_column= 'sgRNA',off_target_column= 'Align.off-target',
                                                  chrom= 'chrom', realigned_target= 'Align.sgRNA', chrom_start='Align.chromStart',
                                                   chrom_end= 'Align.chromEnd',strand= 'Align.strand',bulges= 'Align.#Bulges',insertions=None,
                                                   deletions=None, label='label',read_count=None,mismatches='Align.#Mismatches')
    
    true_ot.to_csv("/Off-Target-data-proccessing/Data/TrueOT/Refined_TrueOT.csv",index=False)
    print(true_ot.head(5))  
                              
    
    
    '''
    if not isinstance(data_frame, pd.DataFrame):
        if os.path.exists(data_frame):
            data_frame = pd.read_csv(data_frame)
        else:
            raise ValueError("No data frame given.")
    if not (target_column and off_target_column and chrom and chrom_start and chrom_end and (label or read_count)):
        raise ValueError("No essential columns given.")
    renamed_columns = {target_column: 'target', off_target_column: 'offtarget_sequence', chrom: 'chrom',
                       chrom_start: 'chromStart', chrom_end: 'chromEnd'}
    data_frame.rename(columns=renamed_columns, inplace=True)
    # Check if label is continuous or binary
    if read_count:
        if check_continous_values(data_frame[read_count].values): # read_count is regression
            data_frame.rename(columns={read_count: 'Read_count'}, inplace=True)
        if label:
            data_frame.rename(columns={label: 'Label'}, inplace=True)
        else : # Add label to the data frame
            data_frame["Label"] = 0
            add_label_by_countiniuos_values(data_frame, 'Read_count', 'Label')
    else: # Only binary label
        data_frame.rename(columns={label: 'Label'}, inplace=True)
        data_frame['Read_count'] = 0
    if strand:
        renamed_columns[strand] = 'strand'
    if realigned_target:
        renamed_columns[realigned_target] = 'realigned_target'
    else:
        data_frame['realigned_target'] = data_frame['target']
    data_frame.rename(columns=renamed_columns, inplace=True)
    if bulges:
        if insertions:
            renamed_columns[insertions] = 'insertion'
        else: data_frame['insertion'] = 0
        if deletions:
            renamed_columns[deletions] = 'deletion'
        else: data_frame['deletion'] = 0
        data_frame = add_inserertion_deletion(data_frame, 'realigned_target', 'offtarget_sequence', bulges)
        renamed_columns[bulges] = 'bulges'
        
    if mismatches:
        renamed_columns[mismatches] = 'missmatches'
    data_frame.rename(columns=renamed_columns, inplace=True)
    
    if not "Index" in data_frame.columns:
        data_frame["Index"] = data_frame.index
    return data_frame
def check_continous_values(values):
    if values.value_counts() > 2:
        return True
    elif values.value_counts() == 2:
        return False
def add_label_by_countiniuos_values(data, continous_column, label_column):
    data[label_column] = data[data[continous_column] > 0].astype(int)
    return data
def add_inserertion_deletion(data, align_target_column, off_target_column, bulges_column = None):
    '''
    This function extract from a given sgRNA,OT pair the amount of deletions and insertions if any.
    It assigns the corresponding value to the data frame.
    '''
    # Keep only bulges
    if bulges_column:
        only_bulges = data[data[bulges_column] > 0]
    else: only_bulges = data.copy()
    # Inseretions - DNA BULGE
    only_bulges['insertion'] = only_bulges[align_target_column].apply(lambda x: x.count('-'))
    # Deletions - RNA BULGE
    only_bulges['deletion'] = only_bulges[off_target_column].apply(lambda x: x.count('-'))
    data['insertion'] = data['deletion'] = 0
    data.loc[only_bulges.index,'insertion'] = only_bulges['insertion']
    data.loc[only_bulges.index,'deletion'] = only_bulges['deletion']
    if not bulges_column:
        data["bulges"] = 0
        data.loc[only_bulges.index,'bulges'] = only_bulges['insertion'] + only_bulges['deletion']
    bulges_count = len(data[data[bulges_column] > 0])
    insertion_deletion_count = len(data[(data['insertion'] > 0) | (data['deletion'] > 0)])
    print(f"Number of rows where bulges > 0: {bulges_count}")
    print(f"Number of rows where insertion or deletion > 0: {insertion_deletion_count}")
    return data
def replace_N_for_ot(grna, ot):
    '''This function replaces the N in the grna sequence with the corresponding base from the offtarget sequence.
    Args:
    1. grna - guide sequence
    2. ot - off-target sequence
    ------------
    Returns: guide sequence with the N replaced.'''
    return grna.replace('N',ot)



def split_data_by_guides(whole_data = None, guides_list=None, guide_column=None, 
                         output_suffix=None, output_prefix=None, output_path=""):
    '''
    This function splits a data frame by the given guides.
    It create a new data frame with the data for the given guides.
    Args:
    1. whole_data - (data_frame/str) of the data
    2. guides_list - (list) of guides to split the data by
    3. target_column - (str) column for the guide sequence
    4. output_suffix - (str) suffix for the output file
    5. output_prefix - (str) prefix for the output file
    6. output_path - (str) path for the output file
    ------------
    Returns: None
    Saves the new data frame in the output path.
    '''
    if isinstance(whole_data, str):
        if not output_prefix:
            output_prefix = get_ending(whole_data)
        whole_data = pd.read_csv(whole_data)
    if whole_data.empty:
        raise ValueError("No data frame given.")
    
    if not guides_list:
        raise ValueError("No guides list given.")
    if not output_suffix:
        raise ValueError("No output suffix given.")
    
    current_guides = set(whole_data[guide_column])
    guides_list = set(guides_list)
    intersect = current_guides.intersection(guides_list)
    print(f"Current guides: {len(current_guides)}, Guides to keep: {len(guides_list)}, Intersect: {len(intersect)}")
    filtered_data = whole_data[whole_data[guide_column].isin(guides_list)]
    filtered_guides = set(filtered_data[guide_column])
    if filtered_guides != intersect:
        raise ValueError("Not all guides were filtered.")
    filtered_data.to_csv(os.path.join(output_path,f"{output_prefix}_{output_suffix}.csv"),index=False)
    
def split_data_by_name(data=None, name=None, name_column = None, if_by_guides = False, guide_column=None,
                       output_suffix=None, output_prefix=None, output_path=""):
    '''
    This function splits the data by a given name.
    It creates a new data frame with the data for the given name.
    Args:
    1. data - (data_frame/str) of the data
    2. name - (str) name to split the data by, (list) of names.
    3. name_column - (str) column for the name
    4. if_by_guides - (bool) if True the function will split the data by the guides of the given name
    5. guide_column - (str) column for the guide sequence
    6. output_suffix - (str) suffix for the output file
    7. output_prefix - (str) prefix for the output file
    8. output_path - (str) path for the output file
    ------------
    Returns: None
    Saves the new data frame in the output path.

    example:
    split_data_by_name("Off-Target-data-proccessing/Data/TrueOT/Refined_TrueOT.csv",
                       ["2020_Shapiro","2019_Park"], "Dataset", True, 'target', 'shapiro_park')
    '''
    if not data:
        raise ValueError("No data frame given.")
    if isinstance(data, str):
        if not output_prefix:
            output_prefix = get_ending(data) # get the name of the data
        data = pd.read_csv(data)
    if data.empty:
        raise ValueError("Data frame is empty.")
    if if_by_guides:
        # Get the guide of the given name and split by guides
        if isinstance(name, str):
            name = [name]
        guides = data[data[name_column].isin(name)][guide_column].tolist()
        split_data_by_guides(whole_data=data, guides_list=guides, 
                             guide_column=guide_column, output_suffix=output_suffix,
                              output_prefix=output_prefix, output_path=output_path)
    else:
        filtered_data = data[data[name_column] == name]
        filtered_data.to_csv(os.path.join(output_path,f"{output_suffix}.csv"),index=False)

def calculate_epigenetic_disterbution(folder_path, output_path, epigenetic_file_lists=None):
    '''
    Calculate the epigenetic disterbution (abundance) over the genome.
    Saves the output results in a file in the output path.
    Args:
        folder_path (str): path to the folder with the epigenetic data
        output_path (str): path to save the output
        epigenetic_file_lists (list, optional): a list of the epigenetic files, if given this list will be used instead of the files in the folder
    '''
    '''
    example:
      calculate_epigenetic_disterbution('Off-Target-data-proccessing/Epigenetics/Change-seq/Bed',
                                      'Epigenetics/Change-seq',epigenetic_file_lists=None)  '''
    if epigenetic_file_lists:
        bed_files = epigenetic_file_lists
    else:
        bed_files = get_bed_files(folder_path)
    if len(bed_files) == 0:
        raise ValueError("No bed files found.")
    epigenetic_dict_values = {} # dictionary to keep the values {mark : value}
    genome_base_num = 3e9
    for bed_file in bed_files:
        bed = pybedtools.BedTool(bed_file)
        total_bases = sum(interval.length for interval in bed) # Total number of coverage bases
        mark = get_ending(bed_file)
        mark_disterbution = total_bases/genome_base_num
        print(f'Mark: {mark}, Total bases: {total_bases}, Disterbution: {mark_disterbution}')
        epigenetic_dict_values[mark] = mark_disterbution
    output_file = os.path.join(output_path,"Epigenetic_disterbution.csv")
    pd.DataFrame([epigenetic_dict_values]).to_csv(output_file,index=False)
    print(f"Output file saved in: {output_file}")


def nucleotide_distribution(sequences):
    """
    Calculate the per-position frequency distribution of A, C, G, and T across a list or Series of equal-length DNA sequences.

    Parameters:
    -----------
    sequences : list or pandas.Series of str
        A collection of DNA sequences. All sequences must be the same length.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame where rows correspond to nucleotides ('A', 'C', 'G', 'T'),
        columns correspond to sequence positions (0-based), and values are the
        relative frequency (0–1) of each nucleotide at each position.
    
    Notes:
    ------
    - Sequences are automatically uppercased.
    - Characters other than A/C/G/T are ignored in the output.
    - Assumes all sequences are the same length; no padding or trimming is performed.
    """
    # Convert to uppercase numpy array of shape (n_sequences, sequence_length)
    arr = np.array([list(seq.upper()) for seq in sequences])
    
    # Get sequence length
    seq_len = arr.shape[1]
    
    # Possible nucleotides
    nucleotides = ['A', 'C', 'G', 'T']
    
    # Prepare result: rows = nucleotides, columns = positions
    result = pd.DataFrame(0.0, index=nucleotides, columns=range(seq_len))
    
    for i in range(seq_len):
        col = arr[:, i]
        counts = pd.Series(col).value_counts(normalize=True)
        for nuc in counts.index:
            if nuc in nucleotides:
                result.at[nuc, i] = counts[nuc]
    
    return result


if __name__ == '__main__':
    ### assign epigenetic
    run_intersection(merged_data_path="EpiCRISPROff_/Data_sets/Refined_TrueOT_Lazzarotto_withEpigenetic.csv",
                     epigenetic_folder="EpiCRISPROff_/Epigenetic_data/Epigenetic_data/CHANGE-seq",if_update=False)
    
    
    






  

