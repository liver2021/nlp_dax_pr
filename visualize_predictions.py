import matplotlib.pyplot as plt
import pandas as pd
import torch
from utils.data_preprocessing import preprocess_data
from predict import make_predictions
import numpy as np
from models.lstm_model import LSTMModel
from models.lstm_dot_am import LSTMModelWithDotAttention
from models.lstm_add_am import LSTMModel as LSTMModelWithAdditiveAttention
from models.mlp import MLPModel




def visualize_predictions(data, predictions, title="Model Predictions vs Actual", classification_mode=False, output_file="plot.png"):

    plot_data = data[-len(predictions):]
    days = np.arange(len(predictions))

    plt.figure(figsize=(14, 7))

    plt.plot(days, plot_data['Close'], label='Actual', color='black', linewidth=2)
    if classification_mode:
        predictions = (predictions > 0.5).float()

        for i in range(len(predictions)):
            if predictions[i] == 1:
                plt.vlines(i, ymin=plot_data['Close'].min(), ymax=plot_data['Close'].max(), colors='g', linewidth=2,
                           label='Predicted: 1' if i == 0 else "")
            else :
                plt.vlines(i, ymin=plot_data['Close'].min(), ymax=plot_data['Close'].max(), colors='r', linewidth=2,
                           label='Predicted: 0' if i == 0 else "")



    else:
        plt.plot(days, predictions, label='Predicted', color='r')

    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.savefig(output_file)

if __name__ == "__main__":
    data_file = 'data/dax_features.csv'
    df = pd.read_csv(data_file, parse_dates=['Date'], index_col='Date')
    feature_columns = ['SP_Close', 'Open', 'Volume', 'High', 'Low', 'Gdp', 'Zew', 'Gfk', 'Ifo', 'Interest', 'Inf',
                       'Co_usd', 'Oil_usd', 'Gold_usd', 'Bond_2', 'Bond_10', 'Eu_usd', 'Close', 'SMA_20', 'EMA_20',
                       'RSI_14', 'Middle_Band', 'Upper_Band', 'Lower_Band', 'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal',
                       'Low_14', 'High_14', '%k', '%d', 'CMF','si_x','si_news']

    nlp_idx_x = '/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/x_sentiment_index.csv'
    nlp_idx_news = '/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/news_sentiment_index.csv'

    nlp_feature_x = pd.read_csv(nlp_idx_x, header=0)
    nlp_feature_news = pd.read_csv(nlp_idx_news, header=0)

    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    nlp_feature_x['Date'] = pd.to_datetime(nlp_feature_x['Date'])
    nlp_feature_news['Date'] = pd.to_datetime(nlp_feature_news['Date'])
    df = pd.merge(df, nlp_feature_x, on='Date', how='left')
    df = pd.merge(df, nlp_feature_news, on='Date', how='left')
    df = df.fillna(0)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(input_size=len(feature_columns), hidden_layer_size=206, num_layers=1, classification_mode=False).to(device)
    model.load_state_dict(torch.load('/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/models/weights/reg_lstm_sen.pth', map_location=device))

    sequence_length = 14


    X_train, y_train, X_test, y_test, feature_scaler, target_scaler = preprocess_data(df, df, sequence_length, feature_columns)

    predictions = make_predictions(model, df, feature_scaler, target_scaler, sequence_length, device, feature_columns, target_column='Close',classification_mode=False)


    df = df[-260:-3]
    predictions = predictions[-260:-3]

    visualize_predictions(df, predictions, classification_mode=False, output_file="Plots/PLOT_LSTM_CL_FULL.png")

""" visualize_prediction function creates Plot of Predictions and Underlying Asset
    Args:
        data: Dataframe with Close Price Column
        predictions: List of predicted Values
        classification_mode: True Means the predicted Variable is binary
        output_file: Path for the results ot be saved """

















