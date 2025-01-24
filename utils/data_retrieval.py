import yfinance as yf
import os
import json
import pandas as pd



def download_dax_data(start_date, end_date, file_path, ticker):
    """
    Downloads DAX data from Yahoo Finance and saves it as a CSV file.

    :param start_date: Start date for the data (format: 'YYYY-MM-DD').
    :param end_date: End date for the data (format: 'YYYY-MM-DD').
    :param file_path: Path where the CSV file should be saved.
    """
    if not os.path.exists(file_path):

        dax_data = yf.download(ticker, start=start_date, end=end_date)
        dax_data.to_csv(file_path)
        print(f'Data downloaded and saved to {file_path}')
    else:
        print(f'Data already exists at {file_path}')


def merge_json_files(directory="/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data"):
    merged_data, dates, date, tweets, tweet = [], [], [], [], []
    path = '/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/x_data.json'

    if os.path.exists(path):
        print(f"Tweets already merged!")
        exit()
    else:
        for filename in os.listdir(directory):
            if filename.startswith('x_data'):
                with open(os.path.join(directory, filename), 'r') as f:
                    data = json.load(f)
                    merged_data.append(data)



        dates = [dates['dates'] for dates in merged_data]
        tweets = [tweets['tweets'] for tweets in merged_data]

        for x in dates:
            for y in x:
                date.append(y)
        for x in tweets:
            for y in x:
                tweet.append(y)

        x_data = {'dates': date, 'tweets': tweet}
        df = pd.DataFrame(x_data)
        df['dates'] = pd.to_datetime(df['dates'])
        df.sort_values('dates', inplace=True)
        df['dates'] = df['dates'].astype(str)
        df = df.reset_index(drop=True)
        dict_data = df.to_dict(orient="list")
        with open(path, 'w') as f:
            json.dump(dict_data, f)
    return dict_data

#merge_json_files()
""" 
start_date = "1999-01-01"
end_date = "2024-06-02"
file_path = '/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/dax_data.csv'
download_dax_data(start_date, end_date, file_path,)


"""
