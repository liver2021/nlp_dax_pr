from train import train_rf
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import optuna.visualization as vis
def optuna_study(data_file='data/dax_features.csv', results='/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/optuna_results/reg_results_rf.csv', plot="/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/hp_tuning/parallel_coordinate_test.html", nlp_idx_x='/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/x_sentiment_index.csv', nlp_idx_news='/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/news_sentiment_index.csv', n_trials=5, nlp_mode=False, classification_mode=False,model_name="RF",min_mode=False,max_mode=False):

    def objective(trial):
        start_date = "2015-01-01"
        end_date = "2023-01-01"
        if nlp_mode:
            feature_columns = ['SP_Close','Open','Volume','High','Low','Gdp','Zew','Gfk','Ifo','Interest','Inf','Co_usd','Oil_usd','Gold_usd','Bond_2','Bond_10','Eu_usd','Close','SMA_20','EMA_20','RSI_14','Middle_Band','Upper_Band','Lower_Band','EMA_12','EMA_26','MACD','MACD_Signal','Low_14','High_14','%k','%d','CMF','si_n','si_news']
        else:
            feature_columns = ['SP_Close','Open','Volume','High','Low','Gdp','Zew','Gfk','Ifo','Interest','Inf','Co_usd','Oil_usd','Gold_usd','Bond_2','Bond_10','Eu_usd','Close','SMA_20','EMA_20','RSI_14','Middle_Band','Upper_Band','Lower_Band','EMA_12','EMA_26','MACD','MACD_Signal','Low_14','High_14','%k','%d','CMF']


        sequence_length = trial.suggest_categorical('sequence_length', [4, 7, 14, 21, 28])

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4)

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
                           classification_mode=classification_mode, rf=True, min_mode=min_mode, max_mode=max_mode)
        trial.set_user_attr("metric_1", metrics[1])
        trial.set_user_attr("metric_2", metrics[2])

        return metrics[0]


    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    df = pd.read_csv(results)




    experiment, model, metric_0, metric_1, metric_2, n_estimators, max_depth, min_samples_split, min_samples_leaf, sequence_length, mode= [],[],[],[],[],[],[],[],[],[],[]


    for trial in study.trials:
        trial_params = trial.params
        experiment.append(trial.number)
        metric_0.append(-trial.value)
        model.append(model_name)
        metric_1.append(trial.user_attrs['metric_1'])
        metric_2.append(trial.user_attrs['metric_2'])

        n_estimators.append(trial_params.get('n_estimators', None))
        max_depth.append(trial_params.get('max_depth', None))
        min_samples_split.append(trial_params.get('min_samples_split', None))
        min_samples_leaf.append(trial_params.get('min_samples_leaf', None))
        sequence_length.append(trial_params.get('sequence_length', None))

        if classification_mode and not min_mode and not max_mode:
            mode.append("UP/DOWN")
        if min_mode:
            mode.append("MIN")
        if max_mode:
            mode.append("MAX")


    if classification_mode:
        experiments = {'trial': experiment, 'model': model_name, 'auc': metric_0, 'acc': metric_1, 'f1': metric_2,
                       'n_estimators': n_estimators,
                       'max_depth': max_depth, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, "sequence_length":sequence_length, "mode":mode
                       }
    else:
        experiments = {'trial': experiment, 'model': model_name, 'rmse': metric_0, 'mae': metric_1, 'r2': metric_2, 'n_estimators': n_estimators,
                       'max_depth': max_depth, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, "sequence_length":sequence_length}
    experiments = pd.DataFrame(experiments)
    df = pd.concat([df, experiments], ignore_index=False)

    parallel_coord = vis.plot_parallel_coordinate(study)
    parallel_coord.write_html(plot)
    df.to_csv(results, index=False)


optuna_study(classification_mode=False,results='reg_results_rf.csv', plot="data/parallel_coordinate_rf_reg.html",n_trials=100)

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
        model_name: Name of the Model to train "RF" (Random Forest)
        classification_mode: True means the continuous Target will be transformed to classification target
        min_mode: True means the Continuous Target will be transformed to Minimum Classification Target
        max_mode: True means the Continuous Target will be transformed to Maximum Classification Target
    Returns:
        CSV File with Results    
"""
