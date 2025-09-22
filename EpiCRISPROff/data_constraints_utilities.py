'''
This module contains setting parameters and validating them.
'''

def with_bulges(constraints):
    '''
    Return True if the constraints allows for bulges
    '''
    if constraints == 1: # No constraints
        return True 
    elif constraints == 2: # Mismatch only
        return False
    elif constraints == 3: # Bulges only
        return True
    else:
        raise ValueError("Invalid constraints value")

def get_ot_constraint_name(ot_constraint):
    '''
    Get the name of the off-target constraint.
    '''
    if ot_constraint == 1:
        return "No constraints"
    elif ot_constraint == 2:
        return "Mismatch only"
    elif ot_constraint == 3:
        return "Bulges only"
    else:
        raise ValueError("Invalid ot_constraint value")

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
