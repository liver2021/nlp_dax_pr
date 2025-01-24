# Forecasting DAX Index Movements based on Qualitative and Quantitative Data: German News and Deep Learning for Signal Generation

**Type:** Master's Thesis 

**Author:** Oliver Klatt-Tustanowski

**1st Examiner:** Prof. Dr. Lessmann

**2nd Examiner:** Dr. Matthias Weidlich


![results](/Plots/PLOT_LSTM_AA_REG.png)
## Table of Content

- [Summary](#summary)
- [Working with the repo](#Working-with-the-repo)
    - [Dependencies](#Dependencies)
    - [Setup](#Setup)
- [Reproducing results](#Reproducing-results)
    - [Training code](#Training-code)
    - [Evaluation code](#Evaluation-code)
    - [Pretrained models](#Pretrained-models)
- [Results](#Results)
- [Project structure](-Project-structure)

## Summary

NLP based financial forecasting approaches gained increased attention by providing more accurate decision signals for the valuation of certain assets.
This development sparked the use of finetuned transformer models for feature creation in time series approaches.
This project evaluates the usefulness of such german specific models for german DAX Index forecasting and its improvements in accuracy.


**Keywords**: NLP DAX LSTM Timeseries Forecasting


## Working with the repo

### Dependencies

The project was built with python 3.12. It should also work with another python version 

Code dependencies are given in requirements.txt

### Setup

1. Clone this repository

2. Install requirements
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Reproducing results

To reproduce the results you should do the following:

### Feature Engineering and Preprocessing

1. Open utils/dataset_engineering.py

2. Run dataset_engineering.py which creates technical features and deals with nan values, the results will be saved to dax_features.csv

### Create Quantitative Sentiment Features based on German FinBert

1. Run utils/extracting_urls.py NewsScrapper() function in order to collect News Headers from     
   BörseFranfurt.de which will be saved in data/urls_data.txt

2. Run utils/web_scraping_tweets.py Tweet_Scraper() function in order to to collect desired Tweets     
   which will be saved in data/x_data.json (Tweets are saved in batches in Subfiles
   (Example:x_data_2023-01-01_2024-01-01.json) and than merged for multiple years in x_data.json

3. Run utils/extracting_sentiment.py G_FinBert_Sen() function in order to save quantified sentiments
   for Tweets: data/x_sentiment_data.json for Börse News: data/news_sentiment_data.json

4. Run utils/sentiment_index_function sentiment_index_function() which creates aggregated daily sentiments for Tweets: data/x_sentiment_data.csv for News: data/news_sentiment_index.csv

### Perform Optuna Studies in order to find the best Model Hyperparameter

1. Open optuna_study.py

2. For Neural Classification (MLP,LSTM,LSTM_AA,LSTM_DA) run optuna_stundy() function with selected neural Model for 100 Trails and Classification Target set to True the results should be saved in optuna_results/cl_results.csv for Hyperparameter Optimization results

3. For Neural Regression run optuna_stundy() function with selected neural Model for 100 Trails and Classification Target set to False the results should be saved in optuna_results/reg_results.csv for Hyperparameter Optimization results

4. For Random Forest Classification run optuna_stundy() function with selected RF Model for 100 Trails and Classification Target set to True the results should be saved in optuna_results/cl_results_rf.csv for Hyperparameter Optimization results

5. For Random Forest Regression run optuna_stundy() function with selected RF Model for 100 Trails and Classification Target set to False the results should be saved in optuna_results/reg_results_rf.csv for Hyperparameter Optimization results

6. For Minimum or Maximum Target Models select Classification to True and ether min mode to True or max mode to True for specified Target if Model is RF results should be saved to optuna_results/cl_results_rf.csv else to optuna_results/cl_results.csv

### Evaluate Models with and without Sentiment Features

1. Open feature_study.py feature_study() function

2. Run features_study() function with neural model the results are saved 
   to feature_study_results/cl_feature_study.csv for binary classification target for regression    
   target to feature_study_results/reg_feature_study.csv and for optima target 
   to feature_study_results/opt_feature_study.csv

3. For Random Forest Model run feature_study_rf.py feature_study_rf() function results will be saved     
   in the same files like in Step 2

### Create Plots for selected Model predictions

1. Open visualize_predictions.py

2. Create Regression Plots for LSTM,LSTM_DA,LSTM_AA with visualize_predictions() function and 
   classification mode set to False results will saved to Plots directory

3. Create Classification Plots (horizontal green lines for positive days and red for negative)
   With LSTM Model and classification_mode set to True the results will saved to Plots directory


### Training code

The Training Functions for Neural Models and RF model are in train.py the functions are also utilized in optuna_study.py and feature_study.py

### Evaluation code

evaluate.py contains the necessary evaluation functions which are utilized in optuna_study.py and feature_study.py

### Pretrained models

Pretrained Models can be found in the weights directory (Example: model/weigths/reg_lstm.pth (For Regression LSTM))

## Results

Results are in the feature_study_results directory and further evaluated in the thesis 
 
## Project structure

```bash

sentiment_driven DAX predictor/                               --Main Project Directory
├── data/                                                     --Data Directory
│   ├── hp_tuning/                                            --Hyperparameter tuning results
│   │   ├── cl_parallel_coordinate_LSTM.html                  --Binary Target LSTM Plot
│   │   ├── cl_parallel_coordinate_LSTM_AA.html               --Binary Target LSTM AA Plot
│   │   ├── cl_parallel_coordinate_LSTM_DA.html               --Binary Target LSTM DA Plot
│   │   ├── cl_parallel_coordinate_MLP.html                   --Binary Target MLP Plot
│   │   ├── cl_parallel_coordinate_rf.html                    --Binary Target RF Plot
│   │   ├── max_parallel_coordinate_LSTM.html                 --Maximum Target LSTM Plot
│   │   ├── max_parallel_coordinate_LSTM_AA.html              --Maximum Target LSTM AA Plot
│   │   ├── max_parallel_coordinate_LSTM_DA.html              --Maximum Target LSTM DA Plot
│   │   ├── max_parallel_coordinate_MLP.html                  --Maximum Target MLP Plot
│   │   ├── max_parallel_coordinate_rf.html                   --Maximum Target RF Plot
│   │   ├── min_parallel_coordinate_LSTM.html                 --Minimum Target LSTM Plot
│   │   ├── min_parallel_coordinate_LSTM_AA.html              --Minimum Target LSTM AA Plot
│   │   ├── min_parallel_coordinate_LSTM_DA.html              --Minimum Target LSTM DA Plot
│   │   ├── min_parallel_coordinate_MLP.html                  --Minimum Target MLP Plot
│   │   ├── min_parallel_coordinate_rf.html                   --Minimum Target RF Plot
│   │   ├── parallel_coordinate_LSTM.html                     --Continuous Target LSTM Plot
│   │   ├── parallel_coordinate_LSTM_AA.html                  --Continuous Target LSTM AA Plot
│   │   ├── parallel_coordinate_LSTM_DA.html                  --Continuous Target LSTM DA Plot
│   │   ├── parallel_coordinate_MLP.html                      --Continuous Target MLP Plot
│   ├── raw_data/                                             --Raw data for Dataset creation
│   │   ├── 2_year_bondrate.csv                               --Raw data for 2 year bond rate
│   │   ├── 10_year_bondrate.csv                              --Raw data for 10 year bond rate
│   │   ├── brent_crudeoil_usd.csv                            --Raw data for Brent crude oil
│   │   ├── copper_usd.csv                                    --Raw data for copper price
│   │   ├── dax_data.csv                                      --Raw data for DAX data
│   │   ├── dax_features.csv                                  --Raw dummy Dataset
│   │   ├── eur_usd.csv                                       --Raw data for EUR/USD exchange
│   │   ├── gdp.csv                                           --Raw data for GDP
│   │   ├── Germany_InterestRate.csv                          --Raw data for German interest
│   │   ├── Gfk_cons_Index.csv                                --Raw data for Gfk index
│   │   ├── gold_usd.csv                                      --Raw data for gold price
│   │   ├── ifo_climate_index.csv                             --Raw data for IFO climate index
│   │   ├── inflation.csv                                     --Raw data for inflation rate
│   │   ├── SP_data.csv                                       --Raw data for S&P 500 index
│   │   ├── ZEW_Sen_Index.csv                                 --Raw data for ZEW index
│   ├── dax_features.csv                                      --Feature Dataset
│   ├── news_sentiment_data.json                              --FB quantitative Sentiments
│   ├── news_sentiment_index.csv                              --FB aggregated Sentiment Index
│   ├── urls_data.txt                                         --FrankfurtBörse(FB) News Data
│   ├── x_data.json                                           --X Tweets Data Merged 
│   ├── x_data_2018-01-01_2019-01-01.json                     --X data for specified Period 
│   ├── x_data_2019-01-01_2020-01-01.json                     --X data for specified Period
│   ├── x_data_2020-01-01_2021-01-01.json                     --X data for specified Period
│   ├── x_data_2021-01-01_2022-01-01.json                     --X data for specified Period
│   ├── x_data_2022-01-01_2023-01-01.json                     --X data for specified Period
│   ├── x_data_2023-01-01_2024-01-01.json                     --X data for specified Period
│   ├── x_sentiment_data.json                                 --X quantiative Sentiments 
│   ├── x_sentiment_index.csv                                 --X aggregated Sentiment Index
├── feature_study_results/                                    --Feature study (FS) results
│   ├── cl_feature_study.csv                                  --Binary target FS results 
│   ├── opt_feature_study.csv                                 --Optima target FS results
│   ├── reg_feature_study.csv                                 --Continuous target FS results
├── models/                                                   --Models directory
│   ├── weights/                                              --Model Weights(w)
│   │   ├── cl_lstm.pth                                       --Binary target LSTM w
│   │   ├── cl_lstm_aa.pth                                    --Binary target LSTM AA w
│   │   ├── cl_lstm_aa_sen.pth                                --Binary target LSTM AA Sentiment w
│   │   ├── cl_lstm_da.pth                                    --Binary target LSTM DA w
│   │   ├── cl_lstm_da_sen.pth                                --Binary target LSTM DA Sentiment w
│   │   ├── cl_lstm_mlp.pth                                   --Binary target MLP w
│   │   ├── cl_lstm_mlp_sen.pth                               --Binary target MLP Sentiment w
│   │   ├── cl_lstm_sen.pth                                   --Binary target LSTM Sentiment w
│   │   ├── cl_rf.pkl                                         --Binary target RF w
│   │   ├── cl_rf_sen.pkl                                     --Binary target RF Sentiment w
│   │   ├── max_lstm.pth                                      --Maximum Target LSTM w
│   │   ├── max_lstm_aa.pth                                   --Maximum Target LSTM AA w
│   │   ├── max_lstm_aa_sen.pth                               --Maximum Target LSTM Sentiment w
│   │   ├── max_lstm_da.pth                                   --Maximum Target LSTM DA w
│   │   ├── max_lstm_da_sen.pth                               --Maximum Target LSTM DA Sentiment w
│   │   ├── max_lstm_sen.pth                                  --Maximum Target LSTM Sentiment w
│   │   ├── max_mlp.pth                                       --Maximum Target MLP w
│   │   ├── max_mlp_sen.pth                                   --Maximum Target MLP Sentiment w
│   │   ├── max_rf.pkl                                        --Maximum Target RF pkl
│   │   ├── max_rf_sen.pkl                                    --Maximum Target RF Sentiment pkl
│   │   ├── min_lstm.pth                                      --Minimum Target LSTM w
│   │   ├── min_lstm_aa.pth                                   --Minimum Target LSTM AA w
│   │   ├── min_lstm_aa_sen.pth                               --Minimum Target LSTM AA Sentiment w
│   │   ├── min_lstm_da.pth                                   --Minimum Target DA w
│   │   ├── min_lstm_da_sen.pth                               --Minimum Target DA Sentiment w
│   │   ├── min_lstm_mlp.pth                                  --Minimum Target MLP w
│   │   ├── min_lstm_mlp_sen.pth                              --Minimum Target MLP Sentiment w
│   │   ├── min_lstm_sen.pth                                  --Minimum Target LSTM Sentiment w
│   │   ├── min_rf.pkl                                        --Minimum Target RF pkl
│   │   ├── min_rf_sen.pkl                                    --Minimum Target RF Sentiment pkl
│   │   ├── reg_lstm.pth                                      --Continuous target LSTM w
│   │   ├── reg_lstm_aa.pth                                   --Continuous target AA LSTM w
│   │   ├── reg_lstm_aa_sen.pth                               --Continuous target AA LSTM Sentiment w
│   │   ├── reg_lstm_da.pth                                   --Continuous target LSTM DA w
│   │   ├── reg_lstm_da_sen.pth                               --Continuous target LSTM DA Sentiment w
│   │   ├── reg_lstm_sen.pth                                  --Continuous target LSTM Sentiment w
│   │   ├── reg_mlp.pth                                       --Continuous target MLP w
│   │   ├── reg_mlp_sen.pth                                   --Continuous target MLP Sentiment w
│   │   ├── reg_rf.pkl                                        --Continuous target RF pkl
│   │   ├── reg_rf_sen.pkl                                    --Continuous target RF Sentiment pkl
│   ├── lstm_add_am.py                                        --LSTM AA Model
│   ├── lstm_dot_am.py                                        --LSTM DA Model
│   ├── lstm_model.py                                         --LSTM Model
│   ├── mlp.py                                                --MLP Model
├── optuna_results/                                           --Optuna optimization results
│   ├── cl_results.csv                                        --Classification Neural results
│   ├── cl_results_rf.csv                                     --Classification RF Model results
│   ├── reg_results.csv                                       --Neural Regression Results
│   ├── reg_results_rf.csv                                    --RF Regression Results
├── Plots/                                                    --Plots
│   ├── PLOT_LSTM_AA_REG.png                                  --Plot LSTM AA Regression
│   ├── PLOT_LSTM_CL.png                                      --Plot LSTM Classification
│   ├── PLOT_LSTM_DA_REG.png                                  --Plot LSTM DA Regression
│   ├── PLOT_LSTM_REG.png                                     --Plot LSTM Regression
├── utils/                                                    --Utils directory
│   ├── data_preprocessing.py                                 --Data preprocessing functions
│   ├── data_retrieval.py                                     --Data retrieval functions
│   ├── dataset_engineering.py                                --Dataset engineering functions
│   ├── extracting_sentiment.py                               --Sentiment extraction from data
│   ├── extracting_sentiment_index.py                         --Sentiment index extraction function
│   ├── extracting_urls.py                                    --Web Scraping News Headers
│   ├── optima_prediction.py                                  --Optima prediction function
│   ├── web_scraping_tweets.py                                --Web scraping Tweets 
├── evaluate.py                                               --Model evaluation functions
├── feature_study.py                                          --Feature study script
├── feature_study_rf.py                                       --Feature study script for RF
├── optuna_study.py                                           --Neural Model Hyperparameter Tuning 
├── optuna_study_rf.py                                        --RF Model Hyperparameter Tuning
├── predict.py                                                --Prediction function
├── README.md                                                 --Project documentation
├── requirements.txt                                          --Python dependencies
├── train.py                                                  --Train functions 
├── visualize_predictions.py                                  --Visualization of predictions
                
```


                               

