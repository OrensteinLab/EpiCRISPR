# EpiCRISPROn
We present EpiCRISPROn, a tool designed to predict CRISPR/Cas9 on-target efficiency by incorporating epigenetic and high-throughput data. EpiCRISPROn is a multi-layered perceptron consisting of fully connected layers with 64 and 8 neurons, terminating in a final neuron that outputs on-target efficiency. The input includes a fixed RNA sequence of 32nt length, ten possible epigenetic markers (chromatin accessibility, CTCF binding, DNA methylation, and seven histone modifications: H3K4me1, H3K4me3, H3K9me3, H3K27me3, H3K36me3, H3K9ac, and H3K27ac), and information from a high-throughput data-based model (we used CRISPROn's score predictions).

We utilized the Leenay et al. dataset to evaluate the contributions of flanking sequences, epigenetic marks, and high-throughput data to on-target efficiency prediction. Our results indicate that epigenetic markers play a crucial role in predicting CRISPR/Cas9 on-target efficiency. Moreover, we demonstrated the generalizability of our trained model by successfully predicting on-target efficiency in other cell types.

# Requirements
The model is implemented with Keras 2.8.4 using Tensorflow backend and numpy 1.24.3.

# Run
To run EpiCRISPROn follow the instruction bellow:

### Model Training and evaluation:
For training the model and evaluating it with specific epigenetic markers and high-throughput data information using the Leenay et al. dataset:
1. The first input should be 1.
2. For adding CRISPROn's score predictions, the second input should be True; otherwise, False.
3. Specify the epigenetic markers you would like to use: chromatin, CTCF, methylation, H3K4me1, H3K4me3, H3K9me3, H3K27me3, H3K36me3, H3K9ac, and H3K27ac. If you wish to use all epigenetic markers, the third input should be all.

For training and evaluating the model on the Leenay et al. dataset without CRISPROn scores and with chromatin accessibility marker:
```python
python run.py 1 False chromatin
```
For training and evaluating the model on the Leenay et al. dataset with CRISPROn scores and with CTCF binding and DNA methylation markers:
```python
python run.py 1 True CTCF methylation
```
For training and evaluating the model on the Leenay et al. dataset with CRISPROn score and all epigenetic markers:
```python
python run.py 1 True all
```
### Model evaluation on other human cell types:
For evaluating the model on other human cell types (HEK293, HCT116, K562, and T cell):

1. The first input should be 2.
2. The second input should be the desired cell type.
3. For adding CRISPROn's score predictions, the second input should be True; otherwise, False.
4. Specify the epigenetic markers you would like to use. If you wish to use all available epigenetic markers, the fourth input should be all. For evaluation on other human cell types, the following epigenetic markers are available: chromatin, CTCF, H3K4me1, H3K27ac, and H3K36me3.

For training the model on the Leenay et al. dataset and evaluating on K562 without CRISPROn scores and with all epigenetic markers:
```python
python run.py 2 K562 False all
```
For training the model on the Leenay et al. dataset and evaluating on T-cell with CRISPROn scores and with H3K9me3 chromatin accessibility and CTCF binding markers:
```python
python run.py 2 T True H3K9me3 chromatin CTCF
```
### Model for further use:
For further use with the trained model on the entire Leenay et al. dataset with all epigenetic markers and CRISPROn scores use the following code:
```python
reconstructed_model = keras.models.load_model("EpiCRISPR_model.keras")
```

### Doench et al. 2014
Based on the paper:
Doench, J., Hartenian, E., Graham, D. et al. Rational design of highly active sgRNAs for CRISPR-Cas9–mediated gene inactivation. Nat Biotechnol 32, 1262–1267 (2014). https://doi.org/10.1038/nbt.3026
We extracted the model wieghts in supp table 9 into doecn14_wieghts.csv.

To predict sequence of 30mers score using the doench model run the following:

`python run_doench14.py data_path` where data path is a path to csv file containing a column named `sequence`.

The script will resave that file with another column named: `doench14` with the predicted scores.
