import pandas as pd
import json

def sentiment_index_function(data, is_x_data=True,output="default"):
    df = pd.DataFrame(data)
    if is_x_data == True :
        df['Date'] = pd.to_datetime(df['dates'])
        df['day'] = df['Date'].dt.date
        df['sentiment'] = df['tweets'].astype(int)
    else:
        df['dates'] = pd.to_datetime(df['dates'], format='%d.%m.%y %H:%M')
        df['Date'] = df['dates'].dt.date
        df['sentiment'] = df['tweets'].astype(int)

    grouped = df.groupby('Date').agg(
        sentiment_sum=('sentiment', 'sum'),
        observation_count=('sentiment', 'count'))
    if is_x_data == True :
        grouped['si_x'] = grouped['sentiment_sum'] / grouped['observation_count']
        grouped['si_x'] = grouped['si_x'].round(3)
    else:
        grouped['si_news'] = grouped['sentiment_sum'] / grouped['observation_count']
        grouped['si_news'] = grouped['si_news'].round(3)


    grouped = grouped.reset_index()
    grouped = grouped.drop(columns=['sentiment_sum', 'observation_count'])
    grouped.to_csv(output, index=False)

    return grouped







with open("/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/x_sentiment_data.json", "r", encoding="utf-8") as file:
    x_sentiment_data = json.load(file)

x_output = '/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/x_sentiment_index.csv'
x_grouped = sentiment_index_function(data=x_sentiment_data,output=x_output,is_x_data=True)

with open("/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/news_sentiment_data.json", "r", encoding="utf-8") as file:
    news_sentiment_data = json.load(file)

news_output = '/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/news_sentiment_index.csv'
news_grouped = sentiment_index_function(data=news_sentiment_data, is_x_data=False, output=news_output)

"The Sentiment Index function creates the quantitative Daily Aggregated Sentiment Score based on the daily News Sentiments "