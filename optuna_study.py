import optuna
import optuna.visualization as vis
from models.lstm_model import LSTMModel
from models.lstm_dot_am import LSTMModelWithDotAttention
from models.lstm_add_am import LSTMModel as LSTMModelWithAdditiveAttention
from models.mlp import MLPModel
from train import train_model
import pandas as pd


def optuna_study(data_file='data/dax_features.csv', model_weights='/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/models/weights/reg_lstm.pth', results='reg_results.csv', plot="data/parallel_coordinate.html", nlp_idx_x='/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/x_sentiment_index.csv', nlp_idx_news='/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/news_sentiment_index.csv', n_trials=100, nlp_mode=False, model_name="LSTM", classification_mode=False,min_mode=False,max_mode=False):
    def objective(trial):

        start_date = "2015-01-01"
        end_date = "2023-01-01"
        hidden_size = trial.suggest_int('hidden_size', 50, 300)
        num_layers = trial.suggest_int('num_layers', 1, 3)
        sequence_length = trial.suggest_categorical('sequence_length', [4, 7, 14, 21, 28])
        if nlp_mode:
            feature_columns = ['SP_Close','Open','Volume','High','Low','Gdp','Zew','Gfk','Ifo','Interest','Inf','Co_usd','Oil_usd','Gold_usd','Bond_2','Bond_10','Eu_usd','Close','SMA_20','EMA_20','RSI_14','Middle_Band','Upper_Band','Lower_Band','EMA_12','EMA_26','MACD','MACD_Signal','Low_14','High_14','%k','%d','CMF','si_n','si_news']
        else:
            feature_columns = ['SP_Close','Open','Volume','High','Low','Gdp','Zew','Gfk','Ifo','Interest','Inf','Co_usd','Oil_usd','Gold_usd','Bond_2','Bond_10','Eu_usd','Close','SMA_20','EMA_20','RSI_14','Middle_Band','Upper_Band','Lower_Band','EMA_12','EMA_26','MACD','MACD_Signal','Low_14','High_14','%k','%d','CMF']

        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
        epochs = trial.suggest_int('epochs', 10, 200)
        early_stopping_patience = 20
        mlp_model = False

        if model_name == "MLP":
            input_size = (len(feature_columns) * sequence_length)
            model = MLPModel(input_size=input_size, hidden_layer_size=hidden_size, num_layers=num_layers, classification_mode=classification_mode)
            mlp_model = True

        elif model_name == "LSTM_DA":
            model = LSTMModelWithDotAttention(input_size=len(feature_columns), hidden_layer_size=hidden_size,
                                              num_layers=num_layers, classification_mode=classification_mode)
        elif model_name == "LSTM_AA":
            model = LSTMModelWithAdditiveAttention(input_size=len(feature_columns), hidden_layer_size=hidden_size,
                                              num_layers=num_layers, classification_mode=classification_mode)
        else:
            model = LSTMModel(input_size=len(feature_columns), hidden_layer_size=hidden_size, num_layers=num_layers, classification_mode=classification_mode)




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
                    max_mode=max_mode)


        trial.set_user_attr("metric_1", metrics[1])
        trial.set_user_attr("metric_2", metrics[2])


        return metrics[0]


    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    df = pd.read_csv(results)
    best_params = study.best_params
    print("Best Hyperparameters: ", best_params)
    experiment, model, metric_0, metric_1, metric_2, hidden_size, num_layers, sequence_length, batch_size, learning_rate, epochs, mode = [],[],[],[],[],[],[],[],[],[],[],[]
    for trial in study.trials:
        trial_params = trial.params
        experiment.append(trial.number)
        metric_0.append(trial.value)
        model.append(model_name)
        metric_1.append(trial.user_attrs['metric_1'])
        metric_2.append(trial.user_attrs['metric_2'])

        hidden_size.append(trial_params.get('hidden_size', None))
        num_layers.append(trial_params.get('num_layers', None))
        sequence_length.append(trial_params.get('sequence_length', None))
        batch_size.append(trial_params.get('batch_size', None))
        learning_rate.append(trial_params.get('learning_rate', None))
        epochs.append(trial_params.get('epochs', None))
        if classification_mode and not min_mode and not max_mode:
            mode.append("UP/DOWN")
        if min_mode:
            mode.append("MIN")
        if max_mode:
            mode.append("MAX")


    if classification_mode:
        experiments = {'trial': experiment, 'model': model, 'loss': metric_0, 'acc': metric_1, 'f1': metric_2,
                       'hidden_size': hidden_size,
                       'num_layers': num_layers, 'sequence_length': sequence_length, 'batch_size': batch_size,
                       'learning_rate': learning_rate, 'epochs': epochs, "mode": mode}
    else:
        experiments = {'trial': experiment, 'model': model, 'rmse': metric_0, 'mae': metric_1, 'r2': metric_2, 'hidden_size': hidden_size,
                       'num_layers': num_layers, 'sequence_length': sequence_length, 'batch_size': batch_size,
                       'learning_rate': learning_rate, 'epochs': epochs}

    experiments = pd.DataFrame(experiments)
    df = pd.concat([df, experiments], ignore_index=True)

    parallel_coord = vis.plot_parallel_coordinate(study)
    parallel_coord.write_html(plot)
    df.to_csv(results, index=False)

optuna_study(data_file='data/dax_features.csv', model_weights='/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/models/weights/reg_mlp.pth', results='cl_results.csv', plot="data/hptuning/cl_parallel_coordinate.html", nlp_idx_x='/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/x_sentiment_index.csv', nlp_idx_news='/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/news_sentiment_index.csv', n_trials=4, nlp_mode=False, model_name="MLP", classification_mode=False,min_mode=False,max_mode=False)

""" optuna study performs Range of Experiments for selected range of Hyperparameters and calculates for each Model approach 3 Metrics 
    Args: 
        data_file: Path to Feature Dataset
        model_weights: Path to Weights for desired Model (Weights are in the Models/Weights Folder)
        results: Path to save the results in CSV File (Optuna_results Folder contains all the results)
        plot: Path to save the Hyper Parameter Tuning Plots (Plots are in data/hp_tuning Folder)
        nlp_idx_x: Path to Sentiment Index generated with X(Twitter) Data and German FinBert
        nlp_idx_news: Path to Sentiment Index generated with Börse Frankfurt Data and German FinBert
        n_trails: Number of Experiments or Hyperparameter Combinations to perform
        nlp_mode: True means Sentiment Features will be included in the Dataset else NOT 
        model_name: Name of the Neural Model to train ("MLP","LSTM_AA","LSTM_DA","LSTM"), default is LSTM Model
        classification_mode: True means the continuous Target will be transformed to classification target
        min_mode: True means the Continuous Target will be transformed to Minimum Classification Target
        max_mode: True means the Continuous Target will be transformed to Maximum Classification Target
    Returns:
        CSV File with Results    
"""
