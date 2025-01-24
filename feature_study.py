from models.lstm_model import LSTMModel
from models.lstm_dot_am import LSTMModelWithDotAttention
from models.lstm_add_am import LSTMModel as LSTMModelWithAdditiveAttention
from models.mlp import MLPModel
from train import train_model
import pandas as pd


def feature_study(data_file='data/dax_features.csv',
                  model_weights='models/best_model.pth',
                  results='reg_results.csv',
                  nlp_idx_x='/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/x_sentiment_index.csv',
                  nlp_idx_news='/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/news_sentiment_index.csv',
                  nlp_mode=False,
                  model_name="LSTM",
                  classification_mode=False,
                  min_mode=False,
                  max_mode=False,
                  hidden_size=50,
                  num_layers=2,
                  sequence_length=7,
                  feature_columns=['Bond_10','Eu_usd','Close'],
                  batch_size=32,
                  learning_rate=0.01,
                  epochs=50,
                  feature_set="All Features"):

    start_date = "2015-01-01"
    end_date = "2023-01-01"
    hidden_size = hidden_size
    num_layers = num_layers
    sequence_length = sequence_length
    feature_columns = feature_columns
    if nlp_mode:
        feature_columns = ['SP_Close', 'Open', 'Volume', 'High', 'Low', 'Gdp', 'Zew', 'Gfk', 'Ifo', 'Interest', 'Inf',
                           'Co_usd', 'Oil_usd', 'Gold_usd', 'Bond_2', 'Bond_10', 'Eu_usd', 'Close', 'SMA_20', 'EMA_20',
                           'RSI_14', 'Middle_Band', 'Upper_Band', 'Lower_Band', 'EMA_12', 'EMA_26', 'MACD',
                           'MACD_Signal', 'Low_14', 'High_14', '%k', '%d', 'CMF','si_x','si_news']
    else:
        feature_columns = ['SP_Close', 'Open', 'Volume', 'High', 'Low', 'Gdp', 'Zew', 'Gfk', 'Ifo', 'Interest', 'Inf',
                           'Co_usd', 'Oil_usd', 'Gold_usd', 'Bond_2', 'Bond_10', 'Eu_usd', 'Close', 'SMA_20', 'EMA_20',
                           'RSI_14', 'Middle_Band', 'Upper_Band', 'Lower_Band', 'EMA_12', 'EMA_26', 'MACD',
                           'MACD_Signal', 'Low_14', 'High_14', '%k', '%d', 'CMF']



    batch_size = batch_size
    learning_rate = learning_rate
    epochs = epochs
    early_stopping_patience = 20
    mlp_model = False

    if model_name == "MLP":
        input_size = (len(feature_columns) * sequence_length)
        model = MLPModel(input_size=input_size, hidden_layer_size=hidden_size, num_layers=num_layers,
                         classification_mode=classification_mode)
        mlp_model = True

    elif model_name == "LSTM_DA":
        model = LSTMModelWithDotAttention(input_size=len(feature_columns), hidden_layer_size=hidden_size,
                                          num_layers=num_layers, classification_mode=classification_mode)
    elif model_name == "LSTM_AA":
        model = LSTMModelWithAdditiveAttention(input_size=len(feature_columns), hidden_layer_size=hidden_size,
                                               num_layers=num_layers, classification_mode=classification_mode)
    else:
        model = LSTMModel(input_size=len(feature_columns), hidden_layer_size=hidden_size, num_layers=num_layers,
                          classification_mode=classification_mode)

    metrics = train_model(start_date=start_date,
                          end_date=end_date,
                          data_file=data_file,
                          sequence_length=sequence_length,
                          feature_columns=feature_columns,
                          batch_size=batch_size,
                          learning_rate=learning_rate,
                          epochs=epochs,
                          early_stopping_patience=early_stopping_patience,
                          model=model,
                          nlp_idx_x=nlp_idx_x,
                          nlp_idx_news=nlp_idx_news,
                          model_weights=model_weights,
                          nlp_mode=nlp_mode,
                          classification_mode=classification_mode,
                          mlp_model=mlp_model,
                          min_mode=min_mode,
                          max_mode=max_mode,
                          auc=True)
    if classification_mode and not min_mode and not max_mode:
        mode = "UP/DOWN"
    if min_mode:
        mode = "MIN"
    if max_mode:
        mode = "MAX"

    df = pd.read_csv(results)
    if classification_mode:
        experiments = {'model': model_name,"features" : feature_set, 'auc': metrics[0], 'acc': metrics[1], 'f1': metrics[2], "mode": mode}
    else:
        experiments = {'model': model_name,"features" : feature_set,'rmse': metrics[0], 'mae': metrics[1], 'r2': metrics[2]}

    experiments = pd.DataFrame([experiments])
    df = pd.concat([df, experiments], ignore_index=True)
    df.to_csv(results, index=False)


feature_study(model_weights='models/weights/reg_lstm.pth',
              results='reg_feature_study.csv',
              feature_set="Fin+Macro+Tech",
              classification_mode=False,
              epochs=118,
              learning_rate=0.0001941359785111,
              batch_size=32,
              sequence_length=7,
              num_layers=1,
              hidden_size=143,
              nlp_mode=False,
              model_name="LSTM",
              max_mode=False)

""" Feature Study Function Compares Model Metrics with and without specified Sets of Features
    Args:
        data_file: Path for the Feature Dataset
        model_weights: Path for the Model Weights 
        results: Path to save the results 
        nlp_idx_x: Path to the Sentiment Index from X
        nlp_idx_news: Path to the Sentiment Index from Börse Frankfurt 
        nlp_mode: True means Sentiment Features will be included in the Dataset else NOT 
        model_name: Name of the Neural Model to train ("MLP","LSTM_AA","LSTM_DA","LSTM"), default is LSTM Model
        classification_mode: True means the continuous Target will be transformed to classification target
        min_mode: True means the Continuous Target will be transformed to Minimum Classification Target
        max_mode: True means the Continuous Target will be transformed to Maximum Classification Target
        hidden_size: Given Hidden Size 
        num_layers: Number of Layers in the Model
        sequence_length: Sequence Length for the lagged Values 
        feature_columns: Names of Features in a List 
        batch_size: Batch Size for Training 
        learning_rate: Learning Rate for Training 
        epochs: Amount of Epochs
        feature_set: String which describes the Feature Set used 
        
    returns:
        CSV File with Results 
        
"""
