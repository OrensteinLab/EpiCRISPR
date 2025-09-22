import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
key_renaming = {
    "Only-seq": "Sequence only",
    "Only-sequence": "Sequence only",
    "All-epigenetics": "All epigenetic marks",
    "ATAC-seq": "Chromatin accessibility"
}


# Replace these with your actual data
feature_color_map = {
    "Sequence only": "cornflowerblue",
    "H3K9me3": "lavender",
    "H3K27me3": "sandybrown",
    "H3K36me3": "peachpuff",
    "CRISPRon": "mediumseagreen",
    "CTCF binding": "lightgreen",
    "H3K4me1": "lightcoral",
    "DNA methylation": "pink",
    "H3K9ac": "mediumpurple",
    "Chromatin accessibility": "thistle",
    "H3K27ac": "tan",
    "H3K4me3": "linen",
    "All epigenetic marks": "plum",
    "All epigenetic marks+CRISPRon": "lightpink"
}

# Filter keys without "_" and apply replacements
def rename_and_filter_keys(d, transform_to_all=False):
    new_d = {}
    for key, value in d.items():
        
        if "_" not in key:
            for old, new in key_renaming.items():
                key = key.replace(old, new)
            new_d[key] = value
        elif transform_to_all:
            new_d["All epigenetic marks"] = value
    return new_d


def plot_bar_plot(metric_values,std_values,p_values,feature_names, metric_name, plot_name,  fmt='.4f',
                  additional_info = None, key_renaming_bool=False):
    if key_renaming_bool:
        feature_names = [key_renaming.get(name, name) for name in feature_names]
    sorted_indices = np.argsort(metric_values)
    metric_values = [metric_values[i] for i in sorted_indices]
    std_values = [std_values[i] for i in sorted_indices]
    feature_names = [feature_names[i] for i in sorted_indices]
    matching_colors = [feature_color_map[feature] for feature in feature_names ]

    formatted_feature_names = [
    " ".join(name.split()[:-1]) + "\n " + name.split()[-1] if " " in name else name
    for name in feature_names
]    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    bars = ax.bar(formatted_feature_names, metric_values, yerr=std_values, capsize=5, color=matching_colors)
    min_y = max(min(metric_values)-3*max(std_values),0)
    max_metric_std = max([m+s for m,s in zip(metric_values,std_values)]) + min(std_values)/1.5
    max_y = min(max_metric_std,1)
    # Add value labels
    min_loc = (min([bar.get_height() for bar in bars]) - min_y)/2.0
    for bar, mean,feature_name,std_val in zip(bars, metric_values,feature_names,std_values):
       # gap = (bar.get_height() - min_y)/2.0
        ax.text(bar.get_x() + bar.get_width()/2.0, min_y + 0.6*min_loc , f"{mean:{fmt}}",
                ha='center', va='bottom', fontweight='bold',rotation=90,fontsize=12)
        if feature_name!='Sequence only':
                
            p_val = p_values[feature_name]
            annotation = p_val_annotation(p_val)
            max_height = min(max_y, bar.get_height() + std_val)
            ax.text(bar.get_x() + bar.get_width()/2.0 ,max_height , annotation, va='bottom', ha='center', fontsize=12)
    # Aesthetics
    ax.set_ylabel(metric_name,fontsize=13)
    
    ax.set_ylim(min_y, max_y)
    
    ax.tick_params(axis='x', labelrotation=90, labelsize=13)
    ax.tick_params(axis='y' ,labelsize=12)
    if additional_info:
        for key,info_ in additional_info.items():
            
            if key == 'Cell type':
                ax.plot([], label=info_, color='none')
                continue
            ax.plot([], label=f'{key}: {info_}', color='none')
        ax.legend(loc='upper left',fontsize = 11,borderaxespad=0.05,ncol=1,frameon=False)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    plt.tight_layout()
    path = os.path.join('Figures',f'{plot_name}.png')
    plt.savefig(path,dpi=300)

def p_val_annotation(p_val):
    '''Function returns annotation for a given p-value.'''
    if p_val < 0.001:
        annotation = "***"
    elif p_val < 0.01:
        annotation = "**"
    elif p_val < 0.05:
        annotation = "*"
    else:
        annotation = ""
    return annotation

def create_figure(mean_std_path, p_vals_path, partition_information, suffix, transform_underscore_to_all=False):
    with open(mean_std_path, 'rb') as f:
        mean_std = pickle.load(f)
    with open(p_vals_path, 'rb') as f:
        p_vals = pickle.load(f)
    # Apply to both dictionaries
    mean_std = rename_and_filter_keys(mean_std,transform_underscore_to_all)
    p_vals = rename_and_filter_keys(p_vals,transform_underscore_to_all)
    x_values = list(mean_std.keys())
        
    aurocs_results = [results[0][0] for results in mean_std.values()]
    aurocs_stds = [results[1][0] for results in mean_std.values()]
    roc_pvals = {key: p_vals[key][0] for key in p_vals.keys()}
    plot_bar_plot(aurocs_results,aurocs_stds,roc_pvals,x_values,"Average AUROC",plot_name=f"AUROC_{suffix}")
    auprcs_results = [ results[0][1] for results in mean_std.values()]
    auprcs_stds = [results[1][1] for results in mean_std.values()]
    prc_pvals = {key: p_vals[key][1] for key in p_vals.keys()}
    plot_bar_plot(auprcs_results,auprcs_stds,prc_pvals,x_values,"Average AUPRC",fmt='.3f',plot_name=f"AUPRC_{suffix}",additional_info=partition_information)


if __name__ == "__main__":
        
    # Plot T-cell evaluation
    partition_information = {'Cell type' : 'T cells', 'Verified OTSs': 73, 'Potential OTSs': 236583,'sgRNAs':6}
    mean_std = 'Plots/Exclude_Refined_TrueOT/on_Refined_TrueOT_Lazzarroto/GRU-EMB/Ensemble/All_guides/10_ensembles/mean_std.pkl'
    p_vals = 'Plots/Exclude_Refined_TrueOT/on_Refined_TrueOT_Lazzarroto/GRU-EMB/Ensemble/All_guides/10_ensembles/p_vals.pkl'
    create_figure(mean_std,p_vals,partition_information,'T_cells')

    #Plot HSPC generizability
    partition_information = {'Cell type' : 'CD34+ HSPC', 'Verified OTSs': 25, 'Potential OTSs': 81030,'sgRNAs':3}
    mean_std = 'Plots/Exclude_Refined_TrueOT/on_Refined_TrueOT_shapiro_park/GRU-EMB/Ensemble/All_guides/10_ensembles/mean_std.pkl'
    p_vals = 'Plots/Exclude_Refined_TrueOT/on_Refined_TrueOT_shapiro_park/GRU-EMB/Ensemble/All_guides/10_ensembles/p_vals.pkl'
    create_figure(mean_std,p_vals,partition_information,'HSPC',True)