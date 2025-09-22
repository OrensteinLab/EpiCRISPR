# EpiCRISPROff

We present EpiCRISPROff, a tool designed to predict CRISPR/Cas9 off-target cleavage probabilty by incorporating epigenetic  marks. EpiCRISPROff is based on the CRISPR-Bulge model: a GRU-embedded sequence processing followed by binary vector of epigenetic marks and multi-layered perceptron consisting of fully connected layers with 128 and 64 neurons, terminating in a final neuron that outputs off-target probabilty based on the CRISPR-Bulge model. The input includes a pair of sgRNA sequence and off-target sequence of 23/24nt length and eight possible epigenetic markers (chromatin accessibility and seven histone modifications: H3K4me1, H3K4me3, H3K9me3, H3K27me3, H3K36me3, H3K9ac, and H3K27ac). 

CRISPR-Bulge paper: Ofir Yaish, Yaron Orenstein, Generating, modeling and evaluating a large-scale set of CRISPR/Cas9 off-target sites with bulges, Nucleic Acids Research, Volume 52, Issue 12, 8 July 2024, Pages 6777–6790, https://doi.org/10.1093/nar/gkae428

CRISPR-Bulge repo: https://github.com/OrensteinLab/CRISPR-Bulge

We utilized the 78 GUIDE-seq experiments reproccesed in the CRISPR-Bulge study dataset to evaluate the contributions of epigenetic marks to off-target cleavage probabilty prediction. Our results indicate that epigenetic markers improve prediction preformance in predicting CRISPR/Cas9 off-target cleavage probabilty. Moreover, we demonstrated the generalizability of our trained model by successfully predicting off-target activity in other cell types.


## Requirements
The required libraries are listed in `requirements.txt`

To download and install through pip just run `pip install -r requirements.txt`

## Zipped files and GIT LFS
This repository uses [Git Large File Storage (LFS)](https://git-lfs.github.com/) to manage large files such as `.zip` archives in the following folders:

- `ML_results/`
- `Data_sets/`
- `Epigenetic_data/`

### How to set up Git LFS

   ```bash
   git lfs install
   git clone repo
   cd repo
   git lfs pull
   ```
   After that, the .zip files in the folders above will be available as expected. 
   **Unzip these folders**


## Downloading the trained ensembles
EpiCRISPROff is an ensemble of randomly iniziatled models. There are 10 ensemble each with 50 models for each feature. Thus, due to storage capicity we uploaded the trained ensembles to external repository in `zonedo.org` with a total of ~14Gb models.

To download the models go to the following DOI:
`https://zenodo.org/records/15826405`

Download all the zip files into the **folder `Downloaded_models`**

To unzip all files quickly in the right place:
1. Go to the working repo i.e, `cd EpiCRISPROff_`
2. Run `python Extract_trained_models.py`
This will move all the zipped files into their location and will unzip them
* By defualt the zip folders are not deleted to avoid re downloading. If storage dont allow to keep both zip and unzipped files change `remove_zip` arg to True in `extract_zip` Function in `Extract_trained_models.py`.
* To remove all zipp files at once run: `python Remove_zipped_folders.py`


There are 11 folders, one for each feature, inside each folder there are 10 ensembles yielding a total of 500 models.

## Data

### Off-target data
The off-target sites (OTSs) the model will be trained on should have the following columns:
BULGES_COLUMN, MISMATCH_COLUMN, TARGET_COLUMN (sgRNA sequence), REALIGNED_COLUMN (sgRNA with bulges if), 
OFFTARGET_COLUMN (OTS sequence), CHROM_COLUMN (chromosome of the OTS), START_COLUMN (start position of the OTS)
END_COLUMN (end position of the OTS), BINARY_LABEL_COLUMN (1/0 label for the OTS), REGRESSION_LABEL_COLUMN.
- An off-target dataset example is located in `Data_sets/Sample_data.csv`

**To set these spesific columns values one should set the ```Columns_dict``` values in Jsons\Data_columns_and_paths.json**
- if the sgRNA allignment has no alternations from the original sgRNA one can set the values in REALIGNED_COLUMN to TARGET_COLUMN.

### Epigenetic data
When training a model with sequence and epigenetic data the script assumes the epigenetic data is given in the off-target dataset csv file,
i.e, each epigenetic feature has a column with values for that OTS. 
Further explantion of how to train each type of model is below.

#### BigWig processing
To combine multiple bigwig files and average their per-base value run:
`python bigwig_averaging.py file1.bigwig file2.bigwig.. --output_path /output_directory --output_name file_name`
That will create in `output_directory` a file with `file_name` with average base pair values across the files given in the `[file1, file2...]` list.

**To assign new epigenetic values to an OT dataset one should follow these steps:**
1. In a folder have all the BED/BigWig files you want to assign their data.
2. Make sure a valid bed formated epigenetic data: chromosome \t start \t end (at least).
3. Run the ```run_intersection``` function in the ```Data_labeling_and_processing.py``` script with paths to the off-target data, folder containing the wanted epigenetic data and list of the ["chrom","chrom_start","chrom_end"] columns in the OT data. The function will output a new off-target data: ```_withEpigenetic.csv``` with the intersection values.
* BigWig data assignment is the average values across the off-target-site while BED assignment is a binary values if there is an intersection.
---
## Quick Start

```bash
cd EpiCRISPROff
python main.py --argfile "Args.txt"
```

---

## Model training, testing and evaluating:

## The `Args.txt` configuration file

This file can be altered by the user to train/test/evaluate different model architectures and inputs.

### Arguments:

* `--model (int)`
  Values: `1-6`
  Model types. `6` is the GRU-Embedding model.

* `--cross_val (int)`
  Values:
  `2` – k-cross validation
  `3` – Ensemble

* `--features_method (int)`
  Values:
  `1` – Only sequence
  `2` – With features by columns

* `--features_columns (str)`
  Path to a JSON file with keys as feature descriptions and values as lists of features.

* `--job (str)`
  Options: `train`, `test`, `evaluation`

* `--exclude_guides (list)`
  Format: `[path, str]`
  Exclude sgRNAs from training.

  * `path`: CSV file containing guide sequences
  * `str`: Column name of the guides to exclude

* `--test_on_other_data (list)`
  Format: `[path, str]`

  * `path`: JSON file mapping data names to paths
  * `str`: Name of dataset to test on (must match a key in the JSON)

📁 *Examples can be found in the `Args_examples` folder.*

---

## Models
---
### Only sequence model
Set `features_method = 1` in `Args.txt`
### Sequence + features
Set `features_method = 2`

Set `--features_columns` to a valid path, e.g.:

  * `Jsons/feature_columns_dict_change_seq.json`
  * For HSPC testing: `Jsons/feature_columns_dict_hspc.json`

> ⚠️ Make sure feature names in the JSON file match the column names in corresponding off-target dataset.

---
### Training

By defualt training a new model will train on the 78 GUIDE-seq experiments from the CHANGE-seq study.

**To change the training data change the `"Vivo-silico"` dataset path in Jsons/Data_columns_and_paths.json**
- If you changed the training data make sure it is formated as described in the Off-target data section

**To exclude spesific sgRNAs and their OTSs from the training data: set the `--exclude_guides` Arg in the 'Args.txt' file.**
- To train a model make sure to set the `--job` arg to `"train"`

#### Ensemble training - train n ensemble with m models
To train an esnemble set `--cross_val` to `3`
* Defaults: 10 ensembles × 50 models
  
**To change the number of ensembles and models, one should change the defualts in the `parsing.py` module**

#### K-Fold training - train a model on each partition
To train k-fold models set `--cross_val` to `2`
* Do **not** provide `--exclude_guides` or `--test_on_other_data`
  
**The partitions are listed in Data_sets/Train_guides and Data_sets/Test_guides. Each Train_k.txt is a list of SUBSET sgRNAs from the CHANGE-seq sgRNAs to train on.
The script will train a model on these sgRNAs and their OTSs**

#### Saving trained models
By defualt the trained models are saved for further used, the path constructed in the following way:

`Models/<Exclude_guides>/Model_name/Cross-validation/Feature_type/<cross_val_params>/<Feature_description>/model.keras`

* **K-Cross example:**
  `Models/GRU-EMB/K_cross/Only_sequence/(k').keras`

* **Ensemble example:**
  `Models/Exclude_Refined_TrueOT/GRU-EMB/Ensemble/With_features_by_columns/10_ensembels/50_models/Binary_epigenetics/H3K27me3/ensemble_(n)/model_(m).keras`

> ⚠️ ENSEMBLE training takes a lot of storage and running time:
> 8 individual features + 2 combinations + only sequence = 11 models
> 11 × 10 ensembles × 50 models = **5500 models** ~14GB storage.
---

### Model prediction
To predict a dataset with a trained model set `--job` arg to  `"test"`.
By defualt the predictions are saved for further evaluation.

#### Ensemble Testing
set `--cross_val` to `3`

**To test the trained ensemble on another dataset the `--test_on_other_data` arg must be provided, if not given, the ensemble is evaluated on the training data in the `"vivo_silico"` path in Jsons/Data_columns_and_paths.json.**

* Saves: `ensemble_(n).pkl` containing average predictions of all m models for that ensemble.
  (Originally saved all 50 model predictions, now only average due to storage)

#### K-Cross Testing
Set `--cross_val` to `2`
- Do **not** provide `--exclude_guides` or `--test_on_other_data`

***Each saved model model_{partitionK}.keras will predict the matching test_guides_k_partition.txt in the Data_sets/Test_guides folder**


#### Predictions output paths 
By defualt the predictions are saved for further used, the path constructed in the following way:
`ML_results/<exclude_guides>/<on_dataset>/Model_name/Cross-validation/Feature_type/<cross_val_params>/<Feature_description>/<model_number>.pkl`

* **K-Cross example:**
  `ML_results/GRU-EMB/K_cross/With_features_by_columns/Feature_name/raw_scores.pkl`

* **Ensemble example:**
  `ML_results/Exclude_Refined_TrueOT/on_Refined_TrueOT_shapiro_park/GRU-EMB/Ensemble/With_features_by_columns/10_ensembels/50_models/Binary_epigenetics/H3K27me3/Scores/ensemble_m.pkl`

---

### Evaluation
Evaluates AUROC, AUPRC, and other metrics based on the scores.pkl prediction files.

Each evlaution will have a .pkl file saved for further ploting and .csv file for human readable inspection.

Set `--job` arg to `evaluation`

#### Ensemble evaluation
Set `--cross_val` to `3`
* Evaluates each `ensemble_(n).pkl` score file
* Saves:
  
  * `all_features.pkl`: Dictionary with feature → \[ensemble\_n, metric values]
  * `mean_std.pkl`, `mean_std.csv`
  * `p_val.pkl`, `p_val.csv` (used in `Figures/ROC_PR_figs.py`)

#### K-Cross evaluation
Set `--cross_val` to `2`
* Saves:

  * `results_summary.xlsx`: Each sheet for different metric (in `Plots/GRU-EMB/K_cross/All_partitions`)
  * `averaged_results.csv`, `p_vals.csv`: Averaged metrics + significance test
  * AUROC/AUPRC plots (in `Figures/`)

>  *P-values from Wilcoxon rank-sum test comparing "only sequence" to features*

* To plot the evaluation results run `python ROC_PR_figs.py` in the Figures folder

---


## Interpretation

To interpret the **All-epigenetic model** trained on the 72 GUIDE-seq experiments:

```bash
python interpertation.py
```

* Evaluates the **first ensemble** trained on all epigenetic features

* Saves SHAP object to:
  `Plots/Interpertability/SHAP_values/all_guides.pkl`

* To interpret a different ensemble:

  * Change the path in `interpertation.py`
  * Modify `run_shap()` feature list to match the trained features

You can run:

```bash
python Figures/EpiCRISPROff_interpertability_plot.py
```

To create a **beeswarm plot** of epigenetic feature importances.


---

## Other methods
To run CRISPR-SGRU, CRISPR-MFH go to Other_methods folder. In each folder there is a corresponding readme.md.

## Software
We trained the model on NVIDIA A100 80GB GPU. 
