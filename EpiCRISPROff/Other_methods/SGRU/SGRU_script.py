import numpy as np
import os
import pandas as pd
from Encoder_sgRNA_off import Encoder
from MODEL import Crispr_SGRU
#from sklearn.metrics import roc_curve,precision_recall_curve,auc
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys


def encordingXtest(Xtest):
    final_code = []
    for idx, row in Xtest.iterrows():
        on_seq = row[0]
        off_seq = row[1]
        en = Encoder(on_seq=on_seq, off_seq=off_seq, with_category=True, label=None)
        final_code.append(en.on_off_code)
    return np.array(final_code)



def main():
    data_path = sys.argv[1]
    Xtest = pd.read_csv(data_path,header=None)
    #Xtest = Xtest.iloc[:100]
    labels = Xtest[2]
    Xtest= encordingXtest(Xtest)
    Xtest = np.expand_dims(Xtest, axis=1)

    models = [os.path.join('CHANGEseq',model) for model in os.listdir('CHANGEseq')]
    probs =[]
    for weighs_path in models:
        model=Crispr_SGRU()
        model.load_weights(weighs_path)
        y_pred=model.predict(Xtest)
        y_prob = y_pred[:, 1]
        y_prob = np.array(y_prob)
        probs.append(y_prob)
    probs = np.array(probs)
    probs = probs.mean(axis=0)
    # auprcs, aurocs =[],[]
    # for prob in probs:
    #     fpr, tpr, au_thres = roc_curve(labels, prob)
    #     precision, recall, pr_thres = precision_recall_curve(labels, prob)
    #     aurocs.append(auc(fpr,tpr))
    #     auprcs.append(auc(recall, precision))
    # df = pd.DataFrame({
    #     'auprc': auprcs,
    #     'aurocs': aurocs
    # })
    # df.to_csv('change_seq_CRISPR-SGRU.csv',index=False)
    # print('auprc',auprcs)
    # print('aurocs',aurocs)
    df = pd.DataFrame({'CRISPR-SGRU':probs})
    name = data_path.split(".")[-1]
    output_path = f'CRISPR-SGRU_{name}.csv'
    df.to_csv(output_path,index=False)

if __name__ == "__main__":
    main()