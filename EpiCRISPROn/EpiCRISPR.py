from keras import Sequential
from keras.src.layers import Dense
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr
from sklearn.utils import shuffle
from Epigenetics import *

SIZE = 4  # size of matrix input
UP = 0  # optimal permutation
DOWN = 9


def oneHot(string):
    trantab = str.maketrans('ACGT', '0123')
    string = str(string)
    data = [int(x) for x in list(string.translate(trantab))]
    ret = np.eye(SIZE)[data]
    return ret


def leenay_Model():
    model1 = Sequential()
    model1.add(Dense(64, activation='linear'))
    model1.add(Dense(8, activation='linear'))
    model1.add(Dense(1, activation='sigmoid'))
    model1.compile(optimizer='adam', loss='mse')
    return model1


# Creates the data for training and evaluating the model
def createTrainSet(up, down, seqs_protospacer, seqs_down, seqs_up, seqs_pam, crispron, epigeneticDic):
    data_protospacer = np.array(list(map(oneHot, seqs_protospacer)))
    downSeqs = []
    upSeqs = []
    for seq in seqs_down:
        downSeqs.append(seq[:down])
    for seq in seqs_up:
        upSeqs.append(seq[:up])
    data_pam = np.array(list(map(oneHot, seqs_pam)))
    data_down = np.array(list(map(oneHot, np.array(downSeqs))))
    data_up = np.array(list(map(oneHot, np.array(upSeqs))))
    data = np.append(data_up, data_protospacer, axis=1)
    data = np.append(data, data_pam, axis=1)
    data = np.append(data, data_down, axis=1)
    epigeneticInfo = EpiVector(epigeneticDic, crispron, "Leenay_crispron.csv")
    data = data.reshape(data.shape[0], -1)
    data = np.concatenate((data, epigeneticInfo), axis=1)
    return data


# Creates the test set of other cell types for evaluation of the model
def humanTestCell(cell_type, crispron, epigeneticDic):
    if cell_type == 'T':
        a = pd.read_csv('T_data.csv')
    else:
        a = pd.read_csv('hek293+k562+hct116_efficiency.csv')
    labels_test = a[f'efficiency_{cell_type}'].to_numpy()
    seqs = a["sequence"].tolist()
    gRNAs = []
    for elem in seqs:
        if cell_type == 'T':
            gRNAs.append(elem[13 - UP:36 + DOWN])
        else:
            gRNAs.append(elem[30 - UP:53 + DOWN])
    data = np.array(list(map(oneHot, gRNAs)))
    # adding epigenetic information to sequence input
    humanEpiDic = epigeneticDic
    # for key in epigeneticDic.keys():
    #     if cell_type == 'T':
    #         b = pd.read_csv(f"epigenetics/T/S1/epigenetics_S1_{key}.csv")
    #     else:
    #         b = pd.read_csv(f"epigenetics/{cell_type}/epigenetics_{cell_type}_{key}.csv")
    #     epi = b['epigenetics'].to_numpy()
    #     humanEpiDic[key] = epi
    if cell_type == 'T':
        epigeneticInfo = EpiVector(humanEpiDic, crispron, "T_crispron.csv")
    else:
        epigeneticInfo = EpiVector(humanEpiDic, crispron, "hek293+k562+hct116_crispron.csv")
    data = data.reshape(data.shape[0], -1)
    data = np.concatenate((data, epigeneticInfo), axis=1)
    return data, labels_test


# 5-Cross validation for training and evaluation
def lennaysRun(protospacer, pam, up, down, weights, labels, crispron, epigenetics):
    data = createTrainSet(UP, DOWN, protospacer, down, up, pam, crispron, epigenetics)
    k = 5
    fold = int(len(data) / k) + 1
    sumpearson = 0
    sumspearman = 0
    for i in range(1, len(data) - 1, fold):
        data_test = data[i:i + fold]
        labels_test = labels[i:i + fold]
        data_train = np.append(data[1:i], data[1 + i + fold:], axis=0)
        weights_train = np.append(weights[1:i], weights[1 + i + fold:], axis=0)
        labels_train = np.append(labels[1:i], labels[1 + i + fold:], axis=0)
        data_train, labels_train = shuffle(data_train, labels_train)
        model1 = leenay_Model()
        model1.fit(data_train, labels_train, epochs=20, batch_size=16, verbose=1, sample_weight=weights_train)
        pred_test1 = model1.predict(data_test)
        pred_test = pred_test1
        sumpearson = sumpearson + pearsonr(labels_test, pred_test.reshape(len(pred_test)))[0]
        sumspearman = sumspearman + spearmanr(labels_test, pred_test.reshape(len(pred_test)))[0]
        print(pearsonr(labels_test, pred_test.reshape(len(pred_test))))
    print(sumspearman / k)
    return sumspearman / k


# Used for evaluation of the model on different cell types. works on k562+hek293+hct116+H1
def lennayPredicionOnHumanCells(protospacer, pam, up, down, labels, cell_type, weights, crispron, leenay_epigeneticDic,
                                other_epigenetic_dic):
    data = createTrainSet(UP, DOWN, protospacer, down, up, pam, crispron, leenay_epigeneticDic)
    data_train, labels_train = shuffle(data, labels)
    data_test, labels_test = humanTestCell(cell_type, crispron, other_epigenetic_dic)
    if cell_type == 'T':
        labels_test = np.divide(labels_test, 100.0)
    model1 = leenay_Model()
    model1.fit(data_train, labels_train, epochs=25, batch_size=16, verbose=1, sample_weight=weights)
    pred_test = model1.predict(data_test)
    print(spearmanr(labels_test, pred_test.reshape(len(pred_test))))
    return spearmanr(labels_test, pred_test.reshape(len(pred_test)))
