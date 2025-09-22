import pandas as pd
from scipy.special import expit
import sys
df = pd.read_csv("doenc14_wieghts.csv")  
WIEGHTS = dict(zip(df["Feature"], df["Coefficient"]))

INTERCEPT = WIEGHTS.get('intercept')
GC_LOW = WIEGHTS.get('gc_low')
GC_HIGH = WIEGHTS.get('gc_high')
def extract_single_nuc_features(seq):
    """
    Given a nucleotide sequence, return list of single nucleotide features
    in the format NXX, where N is the base and XX is the 2-digit position (1-based).
    """
    return [f"{nuc}{str(i+1).zfill(2)}" for i, nuc in enumerate(seq)]
def extract_dinuc_features(seq):
    """
    Given a nucleotide sequence, return list of dinucleotide features
    in the format NNXX, where NN is the 2-base pair and XX is the 2-digit starting position.
    """
    return [f"{seq[i:i+2]}{str(i+1).zfill(2)}" for i in range(len(seq) - 1)]


def gc_deviation_features(seq, target=10):
    """
    Given a sequence, compute GC deviation from the target.
    Returns:
        - gc_count: number of G/C bases
        - gc_deviation: abs(gc_count - target)
        - gc_low: positive value if gc_count < target, else 0
        - gc_high: positive value if gc_count > target, else 0
    """
    gc_count = seq.upper().count('G') + seq.upper().count('C')
    deviation = abs(gc_count - target)
    gc_low = max(0, target - gc_count)
    gc_high = max(0, gc_count - target)
    gc_low = gc_low*GC_LOW
    gc_high = gc_high*GC_HIGH
    return gc_low+gc_high


def compute_sequence_score(seq):
    """Multiply sequence feature by doench wieghts

    Args:
        seq (str): DNA sequence

    Returns:
        float: score for that sequence
    """
    # Single nucleotide features
    single_feats = [f"{nuc}{str(i+1).zfill(2)}" for i, nuc in enumerate(seq)]
    # Dinucleotide features
    dinuc_feats = [f"{seq[i:i+2]}{str(i+1).zfill(2)}" for i in range(len(seq) - 1)]
    features = single_feats+dinuc_feats
    score = INTERCEPT
    for feat in features:
        score += WIEGHTS.get(feat, 0.0)
   
    score += gc_deviation_features(seq)
    score = expit(score)
    return score

def main():
    data_path = sys.argv[1]
        
    t_data = pd.read_csv(data_path)
    t_data['30mers'] = t_data['sequence'].apply(lambda x: x[26:56])
    t_data['doench14'] = t_data['30mers'].apply(lambda x: compute_sequence_score(x))
    t_data.drop(columns='30mers',inplace=True)
    t_data.to_csv(data_path,index=False)
if __name__ =="__main__":
    main()