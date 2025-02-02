import pickle
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
import gc
from aeon.classification.hybrid import HIVECOTEV2

def preprocess(df_train, df_test):
    '''
    Arguments:
    ----------
    df_train : pandas.DataFrame
        The training dataset, where the first column contains the labels, and the remaining columns are features.
    df_test : pandas.DataFrame
        The test dataset, where the first column contains the labels, and the remaining columns are features.

    Returns:
    --------
    tuple or None
        - If successful, returns a tuple with:
            1. X_train : numpy.ndarray
                The normalized feature values from the training dataset.
            2. X_test : numpy.ndarray
                The normalized feature values from the test dataset.
            3. y_train : pandas.Series
                The processed labels from the training dataset.
            4. y_test : pandas.Series
                The processed labels from the test dataset.
        - Returns `None` if the input data is invalid or empty (after cleaning or transformation).
    '''
    df_train=df_train.dropna()
    df_test=df_test.dropna()
    
    if df_train.empty or df_test.empty:
        return None
    
    y_train = df_train.iloc[:, 0]
    y_test = df_test.iloc[:, 0]
    unique=y_test.unique()

    if y_test.empty or y_train.empty:
        return None

    # Possible classes are -1 and 1
    if -1 in unique:
        y_train = y_train.replace({-1: 0})
        y_test = y_test.replace({-1: 0})

    # Class labels dont't start from 0
    elif 0 not in unique:
        min_label=min(unique)
        y_train=y_train-min_label
        y_test=y_test-min_label
    
    y_train = y_train.astype('long') 
    y_test = y_test.astype('long')  

    X_train = df_train.iloc[:, 1:].astype('float32')  
    X_test = df_test.iloc[:, 1:].astype('float32')  

    # Normalize input values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    
    if y_test.empty or y_train.empty:
        return None

    return X_train, X_test, y_train, y_test


def run_hc2(file_path):
    '''
    Arguments:
    ----------
    file_path : str
        Path to the file with precomputed results.
    '''
    # Path to file with UCR Metadata
    df_meta=pd.read_csv('../DataSummary.csv')
    
    # Open dictionary with already computed results
    try:
        with open(file_path, 'rb') as f:
            results_hc2 = pickle.load(f)
    except FileNotFoundError:
        # If the file doesn't exist, initialize an empty dictionary
        results_hc2 = {}   
        
    keys_to_remove = results_hc2.keys()

    # Skip datasets with existing results
    df_meta = df_meta[~df_meta['Name'].isin(keys_to_remove)]
    
    for index, dataset in df_meta.iterrows():
        
        name=dataset['Name']
        file_train = f'{name}/{name}_TRAIN.tsv'
        file_test = f'{name}/{name}_TEST.tsv'

        df_train = pd.read_csv('../UCRArchive_2018/'+file_train, sep='\t')
        df_test = pd.read_csv('../UCRArchive_2018/'+file_test, sep='\t')
        
        data= preprocess(df_train, df_test)

        if data is None:
            continue

        X_train, X_test, y_train, y_test = data

        # Fit HC2
        hc2 = HIVECOTEV2(random_state=42, n_jobs=-1)

        start_time_hc2=time.time() 
        hc2.fit(X_train, y_train)
        end_time_hc2=time.time()

        execution_time = end_time_hc2 - start_time_hc2

        # Predict
        predictions = hc2.predict(X_test)

        # Calculate performance metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='macro',zero_division=1)
        recall = recall_score(y_test, predictions, average='macro')
        f1 = f1_score(y_test, predictions, average='macro')

        results_hc2[name]=[execution_time, accuracy, f1, precision, recall]

        with open(file_path, 'wb') as f:
            pickle.dump(results_hc2, f,protocol=pickle.HIGHEST_PROTOCOL)

        del hc2
        gc.collect()
            
def main():
    # Define path to file to save results to
    file_path='../results/HiveCote2/results_hc2.pkl'

    run_hc2(file_path)
    

if __name__ == "__main__":
    main()