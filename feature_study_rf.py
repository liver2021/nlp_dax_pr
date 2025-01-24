from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from train import train_rf
import pandas as pd


def feature_study_rf(data_file='data/dax_features.csv',
                  results='reg_results.csv',
                  nlp_idx_x='/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/x_sentiment_index.csv',
                  nlp_idx_news='/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/news_sentiment_index.csv',
                  nlp_mode=False,
                  classification_mode=False,
                  min_mode=False,
                  max_mode=False,
                  sequence_length=7,
                  feature_columns=['Bond_10','Eu_usd','Close'],
                  feature_set="All Features",
                  n_estimators=100,
                  max_depth=10,
                  min_samples_split=5,
                  min_samples_leaf=10,
                  save=False,
                  model_save="model.pkl"):

    start_date = "2015-01-01"
    end_date = "2023-01-01"

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



    params = {
        "n_estimators":n_estimators,
        "max_depth": max_depth,
        "min_samples_split":min_samples_split,
        "min_samples_leaf": min_samples_leaf

    }

    if classification_mode:
        model = RandomForestClassifier(**params)
    else:
        model = RandomForestRegressor(**params)

    metrics = train_rf(start_date=start_date,
                       end_date=end_date,
                       data_file=data_file,
                       sequence_length=sequence_length,
                       feature_columns=feature_columns,
                       model=model,
                       nlp_idx_x=nlp_idx_x,
                       nlp_idx_news=nlp_idx_news,
                       nlp_mode=nlp_mode,
                       classification_mode=classification_mode, rf=True, min_mode=min_mode, max_mode=max_mode,save=save,model_save=model_save)

    if classification_mode and not min_mode and not max_mode:
        mode = "UP/DOWN"
    if min_mode:
        mode = "MIN"
    if max_mode:
        mode = "MAX"

    df = pd.read_csv(results)
    if classification_mode:
        experiments = {'model': "RF","features" : feature_set, 'auc': -metrics[0], 'acc': metrics[1], 'f1': metrics[2], "mode": mode}
    else:
        experiments = {'model': "RF","features" : feature_set,'rmse': metrics[0], 'mae': metrics[1], 'r2': metrics[2]}

    experiments = pd.DataFrame([experiments])
    df = pd.concat([df, experiments], ignore_index=True)
    df.to_csv(results, index=False)


feature_study_rf(results='reg_feature_study.csv',
                 feature_set=" ",
                 classification_mode=False,
                 sequence_length=4,
                 nlp_mode=False,
                 min_mode=False,
                 n_estimators=271,
                 max_depth=126,
                 min_samples_split=6,
                 min_samples_leaf=4,
                 save=True,
                 model_save='models/weights/reg_rf.pkl')

""" 
    Feature Study Function Compares Model Metrics with and without specified Sets of Features
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
        max_depth: Maximum Depth  
        min_samples_split: Minimum Splits
        sequence_length: Sequence Length for the lagged Values 
        n_estimators: Amount of Trees 
        min_samples_leaf: Minimum Sample Leafs
        save: True means the RF Parameters will be saved 
        model_save: Path to save pickle file for RF
        feature_set: String which describes the Feature Set used 

    returns:
        CSV File with Results 

"""