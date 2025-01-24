import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks



def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def train_test_split_data(df, test_size=0.2):
    train_data, test_data = train_test_split(df, test_size=test_size, shuffle=False)
    return train_data, test_data


def create_bullish_target(df):
    data = df['Close'].tolist()
    df = df.drop(df.index[-1])
    differences = []
    for i in range(0, len(data)-1):
        differences.append(data[i + 1] - data[i])
    df['target'] = [1 if diff > 0 else 0 for diff in differences]
    return df

def create_minima_target(df):
    peaks, _ = find_peaks(-df['Close'], prominence=1)
    peak_list = np.zeros(len(df))
    peak_list[peaks] = 1
    df['target'] = peak_list
    return df

def create_maxima_target(df):
    peaks, _ = find_peaks(df['Close'], prominence=1)
    peak_list = np.zeros(len(df))
    peak_list[peaks] = 1
    df['target'] = peak_list
    return df

def preprocess_data(train_data, test_data, sequence_length, feature_columns, target_column='Close',classification_mode=False):
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_columns_without_target = [col for col in feature_columns if col != target_column]
    scaled_train_features = feature_scaler.fit_transform(train_data[feature_columns_without_target])
    scaled_test_features = feature_scaler.transform(test_data[feature_columns_without_target])

    def combine_features_and_target(features, target):
        x = np.hstack([features, target])
        return x

    def combine_features_binary_and_target(features,target, cl_target):
        cl_target = cl_target.to_numpy()
        new_shape = (cl_target.shape[0], 1)
        cl_target = cl_target.reshape(new_shape)
        x = np.hstack([features, target, cl_target])
        return x

    if classification_mode:
        cl_train_target = train_data["target"]
        cl_test_target = test_data["target"]
        scaled_train_target = target_scaler.fit_transform(train_data[[target_column]])
        scaled_test_target = target_scaler.transform(test_data[[target_column]])
        scaled_train_data_combined = combine_features_binary_and_target(scaled_train_features, scaled_train_target, cl_train_target)
        scaled_test_data_combined = combine_features_binary_and_target(scaled_test_features, scaled_test_target, cl_test_target)


    else:
        scaled_train_target = target_scaler.fit_transform(train_data[[target_column]])
        scaled_test_target = target_scaler.transform(test_data[[target_column]])
        scaled_train_data_combined = combine_features_and_target(scaled_train_features, scaled_train_target)
        scaled_test_data_combined = combine_features_and_target(scaled_test_features, scaled_test_target)

    def create_sequences(data, sequence_length, classification_mode=False):
        sequences = []
        targets = []
        if classification_mode:
            for i in range(len(data) - sequence_length):
                sequences.append(data[i:i + sequence_length, :-1])
                targets.append(data[i + sequence_length, -1])
            return np.array(sequences), np.array(targets)

        else:
            for i in range(len(data) - sequence_length):
                sequences.append(data[i:i + sequence_length, :])
                targets.append(data[i + sequence_length, -1])
        return np.array(sequences), np.array(targets)



    if classification_mode:
        X_train, y_train = create_sequences(scaled_train_data_combined, sequence_length, classification_mode=True)
        X_test, y_test = create_sequences(scaled_test_data_combined, sequence_length, classification_mode=True)

    else:
        X_train, y_train = create_sequences(scaled_train_data_combined,sequence_length)
        X_test, y_test = create_sequences(scaled_test_data_combined,sequence_length)

    return X_train, y_train, X_test, y_test, feature_scaler, target_scaler





