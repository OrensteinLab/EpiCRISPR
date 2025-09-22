
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import sys
model = load_model('CRISPR_MFH_best_model.h5')


def load_encoded_data(dataset_name):
    with open(dataset_name, 'rb') as f:
        data = pickle.load(f)
    return data['X_predict'], data['X_on'], data['X_off'], data['labels']

def main():
    data_path = sys.argv[1]
    X_predict_test,X_on_test,X_off_test,labels = load_encoded_data('data_path')
    predictions = model.predict([X_predict_test, X_on_test, X_off_test])
    predicted_scores = predictions[:, 1]
    df = pd.DataFrame({'CRISPR-MFH':predicted_scores})
    name = data_path.split(".")[-1]
    output_path = f'CRISPR-MFH_{name}.csv'
    df.to_csv(output_path,index=False)
    
if __name__ == "__main__":
    main()