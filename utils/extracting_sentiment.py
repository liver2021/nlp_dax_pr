from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json

def G_FinBert_Sen(path_x_sentiment="default", path_news_sentiment="default"):
    times = []
    urls = []
    headlines = []
    sentiment = []
    sentiments = {}
    if path_x_sentiment == "/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/x_sentiment_data.json":
        with open("/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/x_data.json", 'r') as f:
            tweets = json.load(f)
    else:
        filename_in = "/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/urls_data.txt"

        with open(filename_in, 'r', encoding='utf-8') as file:
            for line in file:
                elements = line.strip().split('\t')
                if len(elements) == 3:
                    times.append(elements[0])
                    urls.append(elements[1])
                    headlines.append(elements[2])
        tweets = {"tweets": headlines, "dates": times}

    model_name = "scherrmann/GermanFinBert_SC_Sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    for i, tweet in enumerate(tweets['tweets']):

        inputs = tokenizer(tweet, return_tensors="pt", padding=True)

        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)

        predicted_class = torch.argmax(probs, dim=1).item()
        labels = [-1, 0, 1]

        predicted_sentiment = labels[predicted_class]
        sentiment.append(predicted_sentiment)

    sentiments['dates'] = tweets['dates']
    sentiments['tweets'] = sentiment

    if path_x_sentiment == "/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/x_sentiment_data.json":
        with open(path_x_sentiment,"w") as file:
            json.dump(sentiments, file, ensure_ascii=False)
    else:
        with open(path_news_sentiment,"w") as file:
            json.dump(sentiments, file, ensure_ascii=False)

























path_x_sentiment = "/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/x_sentiment_data.json"
path_news_sentiment = "/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/news_sentiment_data.json"

G_FinBert_Sen(path_x_sentiment=path_x_sentiment)
"""Calculates Quantitative Sentiments with German FinBert
 Args:
   path_x_sentiment: path to save quantitative sentiments for Twitter(X)
   path_news_sentiment: path to save quantitative sentiments for Börse Frankfurt
 Returns: Quantitative Sentiments 
 Example: Input:["The DAX Development is good ","The DAX Development is not good"] Output: [1,-1]
   """







