import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils.data_preprocessing import load_data, train_test_split_data, preprocess_data, create_bullish_target, create_maxima_target, create_minima_target
from evaluate import evaluate_model
import pandas as pd
import numpy as np
import pickle

def train_model(start_date, end_date, data_file, sequence_length, feature_columns, batch_size, learning_rate, epochs, early_stopping_patience, model, model_weights, nlp_idx_x='/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/x_sentiment_index.csv', nlp_idx_news='/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/news_sentiment_index.csv', mlp_model=False, classification_mode= False, nlp_mode=False,min_mode=False,max_mode=False,auc=False):

    # Initialize lists to track loss values
    train_losses = []
    val_losses = []

    df = load_data(data_file)
    if classification_mode and (not min_mode or max_mode):
        df = create_bullish_target(df)


    if min_mode:
        df = create_minima_target(df)
    if max_mode:
        df = create_maxima_target(df)

    if nlp_mode:
        nlp_feature_x = pd.read_csv(nlp_idx_x, header=0)
        nlp_feature_news = pd.read_csv(nlp_idx_news, header=0)

        df['Date'] = pd.to_datetime(df['Date'])
        nlp_feature_x['Date'] = pd.to_datetime(nlp_feature_x['Date'])
        nlp_feature_news['Date'] = pd.to_datetime(nlp_feature_news['Date'])
        df = pd.merge(df, nlp_feature_x, on='Date', how='left')
        df = pd.merge(df, nlp_feature_news, on='Date', how='left')
        df = df.fillna(0)

    train_data, test_data = train_test_split_data(df)



    X_train, y_train, X_test, y_test, feature_scaler, target_scaler = preprocess_data(train_data, test_data,
                                                                                      sequence_length, feature_columns, classification_mode= classification_mode)


    if mlp_model:
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_train = X_train.view(X_train.shape[0], -1)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_train = y_train.view(y_train.shape[0], -1)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        X_test = X_test.view(X_test.shape[0], -1)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        y_test = y_test.view(y_test.shape[0], -1)

    else:
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

    total_size = len(X_train)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    train_dataset = TensorDataset(X_train[:train_size], y_train[:train_size])
    val_dataset = TensorDataset(X_train[train_size:], y_train[train_size:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = len(feature_columns)
    model = model.to(device)
    if classification_mode:
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)

            if mlp_model:
                y_pred = y_pred[0]
            else:
                y_pred = y_pred[0].squeeze()
            if classification_mode:
                loss = criterion(y_pred, y_batch)
            else:
                loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)


        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                y_val_pred = model(X_val_batch)
                if mlp_model:
                    y_val_pred = y_val_pred[0]
                else:
                    y_val_pred = y_val_pred[0].squeeze()
                val_loss += criterion(y_val_pred, y_val_batch).item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_weights)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    model = model.to(device)
    model.load_state_dict(torch.load(model_weights, map_location=device))
    metrics = evaluate_model(model, X_test, y_test, target_scaler, device=device, classification_mode=classification_mode, auc=auc)

    return metrics


def train_rf(start_date, end_date, data_file, sequence_length, feature_columns, model, nlp_idx_x='/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/x_sentiment_index.csv', nlp_idx_news='/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/news_sentiment_index.csv', classification_mode= False, nlp_mode=False, rf=False,min_mode=False,max_mode=False,save=False, model_save="model.pkl"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = load_data(data_file)
    if classification_mode and (not min_mode or max_mode):
        df = create_bullish_target(df)
        print(df['Close'])

    if min_mode:
        df = create_minima_target(df)
    if max_mode:
        df = create_maxima_target(df)


    if nlp_mode:
        nlp_feature_x = pd.read_csv(nlp_idx_x, header=0)
        nlp_feature_news = pd.read_csv(nlp_idx_news, header=0)

        df['Date'] = pd.to_datetime(df['Date'])
        nlp_feature_x['Date'] = pd.to_datetime(nlp_feature_x['Date'])
        nlp_feature_news['Date'] = pd.to_datetime(nlp_feature_news['Date'])
        df = pd.merge(df, nlp_feature_x, on='Date', how='left')
        df = pd.merge(df, nlp_feature_news, on='Date', how='left')
        df = df.fillna(0)

    train_data, test_data = train_test_split_data(df)


    X_train, y_train, X_test, y_test, feature_scaler, target_scaler = preprocess_data(train_data, test_data,sequence_length, feature_columns,classification_mode=classification_mode)
    x = X_train.shape[0]
    X_train = np.reshape(X_train, (x, -1))
    model.fit(X_train, y_train)
    if save:
        with open(model_save, 'wb') as f:
            pickle.dump(model, f)

    metrics = evaluate_model(model, X_test, y_test, target_scaler, device, classification_mode=classification_mode, rf=rf)
    return metrics




