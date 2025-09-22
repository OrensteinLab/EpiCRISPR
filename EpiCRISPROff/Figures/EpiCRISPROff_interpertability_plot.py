
import shap
import os
import matplotlib.pyplot as plt
import pickle



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

# 1. Shapley values for the EpiCRISPR Off-target prediction model

shaply_vals = 'Plots/Interpertability/SHAP_values/all_guides.pkl'
feature_names = ["H3K27me3", "H3K27ac", "H3K9ac", "H3K9me3", "H3K36me3", "ATAC-seq", "H3K4me3", "H3K4me1"]
with open(shaply_vals, 'rb') as f:
    shaply_vals = pickle.load(f)
shaply_vals = convert_shap_to_shap_epi(shaply_vals, feature_names)

path = 'Figures'

shap.plots.beeswarm(shaply_vals, show=False, color_bar_label="Feature value", color_bar=True)

# Get the current figure and axes created by SHAP
fig = plt.gcf()
ax = plt.gca()

# Now resize the figure
fig.set_size_inches(6.2, 4.5)

# Adjust font size for y-axis tick labels
for tick in ax.get_yticklabels():
    tick.set_fontsize(14)

# Adjust colorbar ticks (colorbar is usually the last axis)
cbar = fig.axes[-1]
cbar.set_yticks([cbar.get_ylim()[0], cbar.get_ylim()[1]])
cbar.set_yticklabels(['0', '1'])

plt.tight_layout()

shap_path = os.path.join(path, 'EpiCRISPROff_shap_plot.png')
fig.savefig(shap_path, dpi=300)
plt.close()

