import math
import numpy as np
import pandas as pd

"This files adds the features (epigenetic information and high-throughput data) to the sequence input"


# Extract the CRISPROn scores
def readCRISPRON(crispron_file):
    a = pd.read_csv(crispron_file)
    names = a['ID'].to_numpy()
    names = np.array(names, dtype=str)
    indexes = np.char.find(names, 'p_5')
    result_indexes = np.where(indexes != -1)[0]
    CRISPRon = a['CRISPRon'].to_numpy()
    scores = CRISPRon[result_indexes]
    return scores.reshape(-1, 1)


# Adds the features to the input
def EpiVector(epigeneticDic, crispron, file):
    epiMat = []
    if len(epigeneticDic.keys()) != 0:
        first = list(epigeneticDic.keys())[0]
        for i in range(len(epigeneticDic[first])):
            vec = []
            for key in epigeneticDic.keys():
                if key != 'methylation':
                    if '1' in str(epigeneticDic[key][i]):
                        vec.append(1)
                    else:
                        vec.append(0)
                else:
                    epi = epigeneticDic[key][i].split(',')
                    s = math.fsum(list(map(float, epi)))
                    vec.append(s / 32)
            epiMat.append(vec)
    data = np.array(epiMat)
    if crispron == 'True':  # adding high-throughput data information
        scores = readCRISPRON(file)
        if epiMat:  # if epigenetic information was requested
            data = np.column_stack((data, scores))
        else:
            return scores
    return data
