import os
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import Encoding_List
import sys


def MFH_encoding(dataset_path):
    dim=7
    print(f"Processing dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)

    encoded_predict, encoded_on, encoded_off = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        on_off_encoded, on_encoded, off_encoded = Encoding_List.MFH(row['on_seq'], row['off_seq'],
                                                                                     dim=dim)
        encoded_predict.append(on_off_encoded)
        encoded_on.append(on_encoded)
        encoded_off.append(off_encoded)

    X_predict = np.array(encoded_predict, dtype=np.float32).reshape((len(encoded_predict), 1, 24, 7))
    X_on = np.array(encoded_on, dtype=np.float32).reshape((len(encoded_on), 1, 24, 5))
    X_off = np.array(encoded_off, dtype=np.float32).reshape((len(encoded_off), 1, 24, 5))
    labels = df['label'].values.astype('int')

    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

    with open(f'{dataset_name}_sci3_dim{dim}.pkl', 'wb') as f:
        pickle.dump({
            'X_predict': X_predict,
            'X_on': X_on,
            'X_off': X_off,
            'labels': labels
        }, f)

    print(f"Encoded data saved for {dataset_name}")
def main():
    data_path = sys.argv[1]
    MFH_encoding(data_path)

if __name__ == "__main__":
    main()