from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
import time
import datetime
import json

def create_date_object(date_string):
    try:
        return datetime.datetime.strptime(date_string, '%Y-%m-%d')
    except ValueError:
        print("Invalid date format. Please use YYYY-MM-DD.")
        return None

def tweet_logger(Email="Email",Account_Name="Account_Name",Password="1234"):
    driver = webdriver.Firefox()
    driver.get('https://twitter.com/login')
    time.sleep(5)
    username = driver.find_element(By.NAME, 'text')
    username.send_keys(Email)
    username.send_keys(Keys.RETURN)
    time.sleep(5)
    try:
        username = driver.find_element(By.NAME, 'text')
        username.send_keys(Account_Name)
        username.send_keys(Keys.RETURN)
        time.sleep(5)
        password = driver.find_element(By.NAME, 'password')
        password.send_keys(Password)
        password.send_keys(Keys.RETURN)
        time.sleep(5)
    except NoSuchElementException:
        password = driver.find_element(By.NAME, 'password')
        password.send_keys(Password)
        password.send_keys(Keys.RETURN)
        time.sleep(5)

    return driver

def daily_tweet_scrapper(target_tweets, search_url, driver,x):

    collected_tweets = set()
    if x == 1:
        start = 'https://x.com/explore'
        driver.get(start)
        x = x + 1

    time.sleep(3)
    search_box = driver.find_element(By.CSS_SELECTOR, "div div input[data-testid='SearchBox_Search_Input']")
    text_to_enter = search_url
    search_box.send_keys(text_to_enter)
    search_box.send_keys(Keys.RETURN)
    time.sleep(5)



    def scroll_and_extract(driver=driver):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        tweets = soup.find_all('div', {'data-testid': 'tweetText'})
        new_tweet_count = 0
        for tweet in tweets:
            tweet_text = tweet.get_text()
            parent_div = tweet.find_parent('div')
            if tweet_text not in collected_tweets:
                collected_tweets.add(tweet_text)
                new_tweet_count += 1
        return new_tweet_count

    while len(collected_tweets) < target_tweets:
        new_tweets = scroll_and_extract(driver)
        if new_tweets == 0:
            break
        print(f"New Tweets {new_tweets}")
        print(f"Collected {len(collected_tweets)} unique tweets so far...")

    return collected_tweets


def Tweet_Scraper(start_date_str, next_date_str, end_date_str,account=None,keyword=None,Email="Email",Password="1234",Account_Name="Account_Name"):
    x = 1
    collected_tweets_total = []
    dates = []
    tweets = {}
    start_date = create_date_object(start_date_str)
    next_date = create_date_object(next_date_str)
    end_date = create_date_object(end_date_str)
    driver = tweet_logger(Email=Email,Password=Password,Account_Name=Account_Name)
    if keyword:
        tokens = keyword.split()
        if len(tokens) >= 2:
            keyword = ' OR '.join(tokens)
            keyword = "({})".format(keyword)
        else:
            keyword = "({})".format(keyword)
        query = keyword
    elif account and keyword:
        accounts = account.split()
        query = keyword + " " + accounts
    else:
        accounts = account.split()
        if len(accounts) >= 2:
            accounts = account.split()
            acc = []
            for a in accounts:
                acc.append('from:')
                acc.append(a)
                acc.append(' OR ')
            acc.pop()
            seperator = ''
            accounts = seperator.join(acc)
            accounts = "({})".format(accounts)
            query = accounts

        else:
            acc_0 = 'from:' + account
            accounts = "({})".format(acc_0)
            query = accounts



    while next_date <= end_date:

        collected_tweets = []
        experiments = 0

        while len(collected_tweets)== 0:
            experiments = experiments + 1
            search_url = query + " " + f"until:{next_date.strftime('%Y-%m-%d')}" + " " + f"since:{start_date.strftime('%Y-%m-%d')}"
            collected_tweets = daily_tweet_scrapper(100, search_url, driver,x)
            driver.back()
            if experiments == 5:
                break

        date = [start_date.strftime('%Y-%m-%d') for item in range(len(collected_tweets))]
        print(f"date: {start_date.strftime('%Y-%m-%d')} collected tweets: {len(collected_tweets)}")
        for tweet in list(collected_tweets):
            collected_tweets_total.append(tweet)
        for date in date:
            dates.append(date)

        print( f"total tweets {len(collected_tweets_total)} ...")
        start_date += datetime.timedelta(days=1)
        next_date += datetime.timedelta(days=1)
    tweets["dates"] = dates
    tweets["tweets"] = collected_tweets_total
    print(f"total amount of collected tweets {len(collected_tweets_total)}")

    with open(
            f"/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/x_data_{start_date_str}_{end_date_str}.json",
            "w") as file:
        json.dump(tweets, file, ensure_ascii=False)

    return tweets






start_time = time.time()
start_date_str = "2024-01-01"
next_date_str = "2018-01-02"
end_date_str = "2024-06-02"
account = "handelsblatt wiwo faznet boersefrankfurt focusfinanzen finanzfluss"
Email = "Email"
Password = "Password"
Account_Name = "Account_Name"


tweets = Tweet_Scraper(start_date_str=start_date_str,next_date_str=next_date_str,end_date_str=end_date_str,account=account,Email=Email,Password=Password,Account_Name=Account_Name)
"""Saves X's (Tweets) from X(Twitter) for specified Period from specified Accounts
    Args:
        start_date_str: Start Date to Filter 
        next_date_str: Next dax after Start Date 
        end_date_str: End Date to Filter
        account: Accounts to Filter 
        Email: Email of User
        Password: Password of User
        Account_Name: Account Name of User 
        
        
      
"""
