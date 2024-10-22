import pickle
import os
import time
import pandas as pd

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


def run_mlp(hidden, lr, X_train, X_test, y_train, y_test, epochs, batch, seed):
    '''
    Arguments:
    ----------
    hidden : tuple
        Tuple containing the number of neurons in each hidden layer.
    lr : float
        Learning rate for the MLP model.
    X_train : numpy.ndarray
        Training feature data.
    X_test : numpy.ndarray
        Test feature data.
    y_train : pandas.Series
        Training labels.
    y_test : pandas.Series
        Test labels.
    epochs : int
        Number of training epochs.
    batch : int
        Batch size for training.
    seed : int
        Random seed for reproducibility.

    Returns:
    --------
    list
        A list containing:
        - execution_time: time taken to train the model,
        - accuracy: model accuracy on the test data,
        - f1: F1-score on the test data,
        - precision: precision on the test data,
        - recall: recall on the test data.
    '''
    # Define MLP Classifier
    sklearn_mlp = MLPClassifier(hidden_layer_sizes=hidden, 
                activation='relu',  # Activation function
                solver='adam',      # Optimization solver
                alpha=0.1,          # L2 penalty (regularization)
                batch_size=batch,   # Minibatch size
                learning_rate='constant', 
                learning_rate_init=lr, 
                max_iter=epochs, 
                random_state=seed)  # Reproducibility

    start_time_cmlp = time.time()
    sklearn_mlp.fit(X_train, y_train)
    end_time_cmlp = time.time()
    
    execution_time = end_time_cmlp - start_time_cmlp
    
    # Predictions and evaluation
    y_pred = sklearn_mlp.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro',zero_division=1)
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    return [execution_time, accuracy, f1, precision, recall]


def preprocess(df_train, df_test):
    '''
    Arguments:
    ----------
    df_train : pandas.DataFrame
        Training dataset with labels in the first column.
    df_test : pandas.DataFrame
        Test dataset with labels in the first column.

    Returns:
    --------
    tuple or None
        Returns a tuple (X_train, X_test, y_train, y_test) where:
        - X_train, X_test: normalized features (numpy arrays),
        - y_train, y_test: labels (pandas Series).
        Returns None if datasets are invalid or empty.
    '''
    # Drop missing values
    df_train = df_train.dropna()
    df_test = df_test.dropna()

    if df_train.empty or df_test.empty:
        return None

    # Preprocess labels
    y_train = df_train.iloc[:, 0]
    y_test = df_test.iloc[:, 0]
    unique = y_test.unique()

    # Adjust label classes to start from 0
    if -1 in unique:
        y_train = y_train.replace({-1: 0})
        y_test = y_test.replace({-1: 0})
    elif 0 not in unique:
        min_label = min(unique)
        y_train = y_train - min_label
        y_test = y_test - min_label

    # Normalize input values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train.iloc[:, 1:])
    X_test = scaler.fit_transform(df_test.iloc[:, 1:])

    return X_train, X_test, y_train, y_test


def multirun(layers, lr, epochs, batch, seeds):
    '''
    Arguments:
    ----------
    layers : list
        List of tuples, each containing the hidden layer configurations.
    lr : float
        Learning rate for training.
    epochs : int
        Number of epochs for training.
    batch : int
        Batch size for training.
    seeds : list
        List of random seeds for reproducibility.

    Returns:
    --------
    None
        Results are saved to pickle files after each dataset run.
    '''
    # Load UCR Metadata
    df_meta = pd.read_csv("/DataSummary.csv")

    for configuration in layers:
        file_path = f'/results/mlp/LearningRate{str(lr).replace(".", ",")}/results_{configuration}.pkl'

        if os.path.exists(file_path):
            # Load previously computed results
            with open(file_path, 'rb') as file:
                results = pickle.load(file)
        else:
            results = {}

        # Skip datasets with existing results
        keys_to_remove = results.keys()
        df_meta = df_meta[~df_meta['Name'].isin(keys_to_remove)]

        for index, dataset in df_meta.iterrows():
            name = dataset['Name']
            input_layer = dataset['Length']
            output_layer = dataset['Class']

            if input_layer == 'Vary':
                continue
            
            # Load train and test datasets
            file_train = f'{name}/{name}_TRAIN.tsv'
            file_test = f'{name}/{name}_TEST.tsv'
            
            df_train = pd.read_csv('/UCRArchive_2018/'+file_train, sep='\t')
            df_test = pd.read_csv('/UCRArchive_2018/'+file_test, sep='\t')

            data = preprocess(df_train, df_test)
            if data is None:
                continue

            X_train, X_test, y_train, y_test = data
            
            # Run MLP for each seed
            for seed in seeds:
                res = run_mlp(configuration, lr, X_train, X_test, y_train, y_test, epochs, batch, seed)

                results.setdefault(name, []).append(res)

                # Save updated results
                with open(file_path, 'wb') as f:
                    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    layers = [
        (300, 300,),
    ]

    # Define parameters
    learning_rates = [0.0001, 0.001, 0.01, 0.1, 1]
    seeds = [0, 1, 2, 5, 42]
    epochs = 500
    batch = 16

    # Run experiments with different learning rates
    for lr in learning_rates:
        multirun(layers, lr, epochs, batch, seeds)


if __name__ == "__main__":
    main()
