import torch
import numpy as np
import pandas as pd
from models.lstm_model import LSTMModel
from utils.data_preprocessing import preprocess_data
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")



def make_predictions(model, data, feature_scaler, target_scaler, sequence_length, device, feature_columns, target_column= 'Close',classification_mode=False):
    model.eval()
    feature_columns_without_target = [col for col in feature_columns if col != target_column]
    with torch.no_grad():
        feature_scaled = feature_scaler.transform(data[feature_columns_without_target])
        target_scaled = target_scaler.transform(data[target_column].to_numpy().reshape(-1, 1))

        def combine_features_and_target(features, target):
            return np.hstack([features, target])

        data_scaled = combine_features_and_target(feature_scaled, target_scaled)

        X = []
        for i in range(len(data_scaled) - sequence_length):
            X.append(data_scaled[i:i + sequence_length])
        X = np.array(X)
        X = torch.tensor(X, dtype=torch.float32).to(device)

        predictions = model(X)

        if not classification_mode:
            predictions = target_scaler.inverse_transform(predictions[0])

    return predictions



if __name__ == "__main__":

    sequence_length = 7
    data_file = 'data/dax_features.csv'
    feature_columns = ['SP_Close', 'Open', 'Volume', 'High', 'Low', 'Gdp', 'Zew', 'Gfk', 'Ifo', 'Interest', 'Inf',
                       'Co_usd', 'Oil_usd', 'Gold_usd', 'Bond_2', 'Bond_10', 'Eu_usd', 'Close', 'SMA_20', 'EMA_20',
                       'RSI_14', 'Middle_Band', 'Upper_Band', 'Lower_Band', 'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal',
                       'Low_14', 'High_14', '%k', '%d', 'CMF']


    df = pd.read_csv(data_file, parse_dates=True, index_col='Date')

    _, _, _, _, feature_scaler, target_scaler = preprocess_data(df, df, sequence_length, feature_columns)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(input_size=len(feature_columns), hidden_layer_size=143, num_layers=1, classification_mode=False).to(device)
    model.load_state_dict(torch.load('models/weights/reg_lstm.pth', map_location=device))

    predictions = make_predictions(model, df, feature_scaler, target_scaler, sequence_length, device, feature_columns)


