'''
Module to interpret data and models
'''
import numpy as np
import pickle
import os
import shap
from file_utilities import create_folder
from features_and_model_utilities import get_feature_name
#from plotting import plot_subplots, sub_plot_shap_beeswarn, sub_plot_shap_bar_plot
from train_and_test_utilities import keep_intersect_guides_indices
from interpertation_utilities import *
from features_engineering import extract_features
#from plotting_utilities import return_colormap
import seaborn as sns
import signal
import tensorflow as tf
tf.experimental.numpy.experimental_enable_numpy_behavior()
def tf_clean_up():
    tf.keras.backend.clear_session()
    print("GPU memory cleared.")
def set_signlas_clean_up():
    signal.signal(signal.SIGINT, tf_clean_up)
    signal.signal(signal.SIGSTOP, tf_clean_up)
##################### MODEL INTERPERTABILITY #####################

##################### SHAP #####################
def run_shap_by_epi_partition(model_path, background_data_path, output_path, explain_data_path = None,
             set_background_from_explain = False, num_of_points=None, specific_indices=None, specific_guides=None):
    """
    Run SHAP analysis by clustering epigenetic features.
    
    Cluster epigenetic features by correlation and use the clusters as background data for SHAP analysis.
    Plot beeswarn and bar plots for each guide.
    
    Args:
        model_path (str): Path to the model/folder of models.
        background_data_path (str): Path to the background data - usually training.
        output_path (str): Path to save the plots and SHAP values.
        explain_data_path (str, optional): Data to explain. If not provided, uses background data.
        set_background_from_explain (bool, optional): If True, sets background data from explain data.
        num_of_points (int, optional): Number of samples to explain.
        specific_indices (list, optional): Specific indices to extract from x_background.
        specific_guides (list, optional): Specific guides to extract from x_background.

    Returns:
        None
    """
    model,model_path = get_model(model_path,"deep") # get model
    #model = model[0]
    features = ["H3K27me3_peaks_binary", "H3K27ac_peaks_binary", "H3K9ac_peaks_binary", "H3K9me3_peaks_binary", "H3K36me3_peaks_binary", "ATAC-seq_peaks_binary", "H3K4me3_peaks_binary", "H3K4me1_peaks_binary"]

    # set epigenetic background data and mask via clustering partition 
    x_background = get_data(background_data_path,features)[0]
    x_background = np.concatenate(x_background)
    epigenetic_data = x_background[:, 600:608]
    corr = np.corrcoef(epigenetic_data, rowvar=False)
    cg = sns.clustermap(corr, method="complete", cmap="RdBu", annot=True)
    linkage = cg.dendrogram_col.linkage  # or row.linkage; they're symmetric here
    masker = shap.maskers.Partition(epigenetic_data, clustering=linkage,max_samples=1000)
    
    # init masker and explainer
    m_basline = None # local sequence changes through inner loop
    def model_wrapper(X_masked):
        batch_size = X_masked.shape[0]
        # unique_rows = np.unique(X_masked, axis=0)
        # num_unique_rows = len(unique_rows)
        # print(f"Number of unique rows: {num_unique_rows}")
        m_input = np.tile(m_basline, (batch_size, 1))
        input = [m_input,X_masked]
        if isinstance(model,list):
            predictions = np.zeros((len(model),batch_size))
            for index,model_ in enumerate(model):
                predictions[index] = model_.predict(input).ravel()
            predictions = predictions.mean(axis=0)
            return predictions 
        return model.predict(input)
    
    
    # set explanation data
    x_explain,y,guides,otss_dict = get_data(explain_data_path,features)
    features = [get_feature_name(feature) for feature in features] # update feature name
    shap_vals = []
    output_path = create_folder(output_path,'SHAP_values_clustering')

    for idx,guide in enumerate(guides): # iterate guides and their off targets
        guide_shap_values_list = []
        sg_x_background = x_explain[idx]
        sg_y = y[idx]
        sg_otss = otss_dict[guide]
        sg_x_selected, sgrna_otss = filter_data_for_interpertation(sg_x_background, sg_y, sg_otss, number_of_points=num_of_points)
        seq_data,epi_data = extract_features(sg_x_selected,600)
        points = len(sg_x_selected)
        for i_data,(m_basline, epi) in enumerate(zip(seq_data,epi_data)): # update m_baseline in each iteration
            print(f'iteration: {i_data+1}/{points}')
            explainer = shap.PermutationExplainer(model_wrapper, masker=masker, max_evals = 17) # 8 *2 +1
            epi = epi.reshape(1,-1)
            shap_values = explainer(epi)
            guide_shap_values_list.append(shap_values)
        guide_shap_values = combine_multiple_epi_shap_to_one(guide_shap_values_list,features)
        np.save(os.path.join(output_path,f'{guide}.npy'), guide_shap_values.values)
        shap_vals.append(guide_shap_values)

    # sub_plot_shap_beeswarn(shap_vals, guides, output_path, suffix='clustered')
    # sub_plot_shap_bar_plot(shap_vals, guides, output_path, linkage)





def get_shaply_values(model, x_background, explainer_type, x_selected = None, only_seq = False):
    """
    Computes SHAP values for the given model using the specified explainer type.
    
    Args:
        model: Trained model to explain.
        x_background (numpy.ndarray or pandas.DataFrame): Background dataset for SHAP.
        explainer_type (str): Type of SHAP explainer to use ('deep', 'gradient', 'kernel').
        x_selected (numpy.ndarray or pandas.DataFrame, optional): Selected dataset to explain.
    
    Returns:
        shap_values (list of numpy arrays): Computed SHAP values for each output class (or regression target).
    """
    
    class SHAPModelWrapper(tf.keras.Model):
        def __init__(self, model, encoded_length=600):
            super().__init__()
            self.model = model
            self.encoded_length = encoded_length
            self.inputs = model.inputs
            self.outputs = model.outputs
        def call(self, X):
            if isinstance(X, list) and len(X) == 1:
                X = X[0]
            if not only_seq:
                X = extract_features(X, encoded_length=self.encoded_length)

            return self.model(X)  # Use this, NOT .predict()

        def predict(self, X, **kwargs):
            return self.call(X).numpy()
    def model_wrapper(X):
        num_of_points = len(X)
        if not only_seq:
                
            X = extract_features(X, encoded_length= 600)
        if isinstance(model,list):
            predictions = np.zeros((len(model),num_of_points))
            for index,model_ in enumerate(model):
                predictions[index] = model_.predict(X).ravel()
            predictions = predictions.mean(axis=0)
            return predictions 
        return model.predict(X)
    if explainer_type == 'deep': # doesnt work
        deep_shap = SHAPModelWrapper(model=model,encoded_length=600)
        explainer = shap.DeepExplainer(deep_shap, x_background)
    elif explainer_type == 'gradient':
        explainer = shap.GradientExplainer(model, x_background)
    elif explainer_type == 'kernel':
        x_background = np.random.permutation(x_background)
        x_background = x_background[:10000]
        explainer = shap.KernelExplainer(model_wrapper, x_background)
    else:
        print('using permutation explainer')
        explainer = shap.PermutationExplainer(model_wrapper, x_background, max_evals = 1217,max_samples=1000) # 608 *2 +1
    if x_selected is None:
        x_selected = x_background
    shap_values = explainer(x_selected)
    
    return shap_values

def transform_to_heatmap(shap_values, seqeunce_length, bits_per_base, additional_features_length = 0):
    """
    Transforms SHAP values into a 2D matrix for heatmap representation.
    """
    if isinstance(shap_values,shap.Explanation):
        shap_values = shap_values.values
    min_shap = shap_values.min()
    max_shap = shap_values.max()
    epigenetics_values = None
    if additional_features_length > 0: # split the shap values to sequence and epigenetics
        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)
        sequence_values = shap_values[:,:seqeunce_length * bits_per_base]
        epigenetics_values = shap_values[:,seqeunce_length * bits_per_base:]
    else: sequence_values = shap_values
    if sequence_values.ndim == 1:
        sequence_values = sequence_values.reshape(1,seqeunce_length , bits_per_base)
    elif sequence_values.ndim == 2:
        sequence_values = sequence_values.reshape(sequence_values.shape[0],seqeunce_length , bits_per_base)
    else:
        raise ValueError("SHAP values should be 1D or 2D")
    return sequence_values, epigenetics_values, min_shap, max_shap

def run_shap(model_path, background_data_path, explainer_type, output_path, explain_data_path = None,
             set_background_from_explain = False, num_of_points=None, specific_indices=None, specific_guides=None,
             only_seq=False,  shuffle_background = False):
    '''
    Runs on the data and model given and extract shap values
    Plots the bars, beeswarm and waterfall plots
    
    Args:
        model_path (str): path to the model/folder of models
        background_data_path (str): path to the background data - usualy training.
        explainer_type (str): type of the explainer to use
        output_path (str): path to save the plots and shapley values.
        
        explain_data_path (str): data to explain.
        set_background_from_explain (str): if to set the background data from the explain data.
        By defualt if explain data not given the explaination is from the background data.

        num_of_points (int, optional): Number of first indices to use from x_background.
        specific_indices (list, optional): Specific indices to extract from x_background.
        specific_guides (list, optional): Specific guides to extract from x_background.
        only_seq (bool, optional) defualt False: If True, only the sequence features will be used otherwise split to sequence and epigenetics.
        shuffle_background (bool, optional) defualt False: if shuffle the background data for each guide. 
    '''
    features = ["H3K27me3_peaks_binary", "H3K27ac_peaks_binary", "H3K9ac_peaks_binary", "H3K9me3_peaks_binary", "H3K36me3_peaks_binary", "ATAC-seq_peaks_binary", "H3K4me3_peaks_binary", "H3K4me1_peaks_binary"]
    
    
    models,model_path = get_model(model_path,"deep")
    model = models[0]
    
    x_background,y,guides,otss_dict = get_data(background_data_path,features)
    
    if explain_data_path:
        x_explain,y,guides,otss_dict = get_data(explain_data_path,features)
        if set_background_from_explain:
            x_background = x_explain
    else: x_explain = x_background
    if specific_guides is None:
        specific_guides = guides
        guide_idx = keep_intersect_guides_indices(guides,specific_guides)
    
    whole_background = np.concatenate(x_background)
    whole_selected = []
    # additional_features = whole_background.shape[1] - 600 if not only_seq else 0
    shap_vals = []
    if shuffle_background:
        output_path = os.path.join(output_path,'Shuffled_background')
    output_path = create_folder(output_path,'SHAP_values')

    for idx in guide_idx:
        sgrna = specific_guides[idx]
        sg_x_background = x_explain[idx]
        sg_y = y[idx]
        sg_otss = otss_dict[sgrna]
        sg_x_selected, sgrna_otss = filter_data_for_interpertation(sg_x_background, sg_y, sg_otss, number_of_points=num_of_points)
        whole_selected.append(sg_x_selected)
        # NOTE: Background set to the whole data
        print(f'shap vals for {sgrna}')
        if shuffle_background:
            np.random.shuffle(whole_background)
        # shap_values = get_shaply_values(models, whole_background, explainer_type, sg_x_selected, only_seq=only_seq)
        # shap_vals.append(shap_values)
        # with open(os.path.join(output_path,f'{sgrna}.pkl'), 'wb') as f:
        #     pickle.dump(shap_values,f)
        #np.save(os.path.join(output_path,f'{sgrna}.npy'), shap_values.values)
            
    # all explaination togther:
    whole_selected = [x[:100] for x in whole_selected] # get first 100 samples
    whole_selected = np.concatenate(whole_selected)
    if shuffle_background:
            np.random.shuffle(whole_background)
    shap_values = get_shaply_values(models, whole_background, explainer_type, whole_selected, only_seq=only_seq)
    shap_vals.append(shap_values)
    with open(os.path.join(output_path,'all_guides.pkl'), 'wb') as f:
        pickle.dump(shap_values,f)
    #np.save(os.path.join(output_path,f'all_guides.npy'), shap_values.values)
    guides.append('All_guides')
    features = [get_feature_name(feature) for feature in features]
    #return shap_vals, features, output_path, guides
    # plot_shap_only_epigenetics(shap_vals, output_path = output_path, 
    #                            feature_names = features, sgrna_otss= guides)

def plot_shap_only_epigenetics(shap_values, output_path, feature_names, sgrna_otss = None):
    '''
    Given a list of shap.explantions, extract the shap values for the epigenetic features and plot them.
    '''
    if isinstance(shap_values,list):
        shap_values = [convert_shap_to_shap_epi(shap_vals,feature_names=feature_names)for shap_vals in shap_values]
        if not sgrna_otss:
            sgrna_otss = [f'Guide {i+1}'for i in range(len(shap_values))]
        # sub_plot_shap_beeswarn(shap_values,sgrna_otss,output_path)
    

def convert_shap_to_shap_epi(shap_values, feature_names):
    """
    Create a shap explanantion object for the epigenetic features from a given shap explantation object.
    The feature_names should match the order of the feature in the shap object.

    Args:

    Returns:

"""
    feature_number = len(feature_names)
    if feature_number > shap_values.values.shape[1]:
        raise RuntimeError("number of features is bigger than shap values")
    subset_shap = shap.Explanation(
    values=shap_values.values[:, -feature_number:],
    base_values=shap_values.base_values,
    data=shap_values.data[:, -feature_number:],
    feature_names=feature_names  # custom names
)
    return subset_shap

def combine_multiple_epi_shap_to_one(shap_values_list, feature_names):
    """
    
    """
    combined_values = np.concatenate([e.values for e in shap_values_list], axis=0)
    combined_data = np.concatenate([e.data for e in shap_values_list], axis=0)
    combined_base_values = np.concatenate([e.base_values for e in shap_values_list], axis=0)
    
    combined_shap = shap.Explanation(
        values=combined_values,
        data=combined_data,
        base_values=combined_base_values,
        feature_names=feature_names
    )
    return combined_shap

    

def main_shap():
    by_clustering = False
    epi_model_path = "Models/Exclude_Refined_TrueOT/GRU-EMB/Ensemble/With_features_by_columns/10_ensembels/50_models/Binary_epigenetics/All-epigenetics/ensemble_1"
    explain_data_path = "Data_sets/Refined_TrueOT_Lazzarotto_withEpigenetic.csv"
    background_data = "Data_sets/vivo-silico-78-Guides_withEpigenetic.csv"
    explainer_type = ""
    output_path = 'Plots/Interpertability'
    specific_guides = None
    number_of_points = 200
    if by_clustering:
        run_shap_by_epi_partition(model_path = epi_model_path, background_data_path = background_data, output_path = output_path,
                                  explain_data_path = explain_data_path, num_of_points = number_of_points)
    else:
        run_shap(model_path=epi_model_path,background_data_path = background_data,explain_data_path=explain_data_path,explainer_type=explainer_type,
                output_path=output_path,num_of_points=number_of_points,specific_guides=specific_guides,only_seq=False)
    

##################### Epigenetics #####################   

if __name__ == "__main__":
    main_shap()
    
