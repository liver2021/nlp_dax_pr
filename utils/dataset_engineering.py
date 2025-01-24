import pandas as pd
file_path_sp = "/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/raw_data/SP_data.csv"
file_path_dax = "/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/raw_data/dax_data.csv"
eu_usd = "/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/raw_data/eur_usd.csv"
bond_2 = "/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/raw_data/2_year_bondrate.csv"
bond_10 = "/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/raw_data/10_year_bondrate.csv"
copper_usd = "/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/raw_data/copper_usd.csv"
oil = "/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/raw_data/brent_crudeoil_usd.csv"
gold = "/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/raw_data/gold_usd.csv"
inf = "/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/raw_data/inflation.csv"
rate = "/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/raw_data/Germany_InterestRate.csv"
gdp = "/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/raw_data/gdp.csv"
zew = "/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/raw_data/ZEW_Sen_Index.csv"
gfk = "/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/raw_data/Gfk_cons_Index.csv"
ifo = "/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/raw_data/ifo_climate_index.csv"

data = pd.DataFrame()
df = pd.read_csv(file_path_dax)
inf = pd.read_csv(inf)
inf['date'] = pd.to_datetime(inf['DateTime'])
inf['Date'] = inf['date'].dt.strftime('%Y-%m-%d')


sp = pd.read_csv(file_path_sp)
sp["dax_close"] = df["Close"]
df['SP_Close'] = sp["Close"]
data['SP_Close'] = sp["Close"]
data['Date'] = df["Date"]
data['Close'] = df["Close"]
data['SP_Close'] = sp["Close"]

data['Open'] = sp["Open"]
data['Volume'] = sp["Volume"]
data['High'] = sp["High"]
data['Low'] = sp["Low"]
data['Adj Close'] = sp["Adj Close"]

data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')

infla = inf[["Date","Value"]]
infla['Date'] = pd.to_datetime(infla['Date'], format='%Y-%m-%d')
infla['Date'] = infla['Date'].dt.strftime('%Y-%m-%d')

gdp = pd.read_csv(gdp)
gdp = gdp[["DateTime","Value"]]
gdp['Date'] = pd.to_datetime(gdp['DateTime'])
gdp['Date'] = gdp['Date'].dt.strftime('%Y-%m-%d')
data.set_index('Date', inplace=True)
gdp.set_index('Date', inplace=True)
data = pd.concat([data, gdp], axis=1, join='outer')
data.reset_index(inplace=True)
first_nan_index = data['Close'].isna().idxmax()
data = data.loc[:first_nan_index - 1]
data['gdp'] = data['Value'].ffill()
data['Gdp'] = data['gdp'].bfill()
data = data.drop(columns=["Value","DateTime"], axis=1)

zew = pd.read_csv(zew)
zew = zew[["DateTime","Value"]]
zew['Date'] = pd.to_datetime(zew['DateTime'])
zew['Date'] = zew['Date'].dt.strftime('%Y-%m-%d')
data.set_index('Date', inplace=True)
zew.set_index('Date', inplace=True)
data = pd.concat([data, zew], axis=1, join='outer')
data.reset_index(inplace=True)
first_nan_index = data['Close'].isna().idxmax()
data = data.loc[:first_nan_index - 1]
data['Zew'] = data['Value'].ffill()
data['Zew'] = data['Zew'].bfill()
data = data.drop(columns=["Value","DateTime"], axis=1)

gfk = pd.read_csv(gfk)
gfk = gfk[["DateTime","Value"]]
gfk['Date'] = pd.to_datetime(gfk['DateTime'])
gfk['Date'] = gfk['Date'].dt.strftime('%Y-%m-%d')
data.set_index('Date', inplace=True)
gfk.set_index('Date', inplace=True)
data = pd.concat([data, gfk], axis=1, join='outer')
data.reset_index(inplace=True)
first_nan_index = data['Close'].isna().idxmax()
data = data.loc[:first_nan_index - 1]
data['Gfk'] = data['Value'].ffill()
data['Gfk'] = data['Gfk'].bfill()
data = data.drop(columns=["Value","DateTime","gdp"], axis=1)

ifo = pd.read_csv(ifo)
ifo = ifo[["DateTime","Value"]]
ifo['Date'] = pd.to_datetime(ifo['DateTime'])
ifo['Date'] = ifo['Date'].dt.strftime('%Y-%m-%d')
data.set_index('Date', inplace=True)
ifo.set_index('Date', inplace=True)
data = pd.concat([data, ifo], axis=1, join='outer')
data.reset_index(inplace=True)
first_nan_index = data['Close'].isna().idxmax()
data = data.loc[:first_nan_index - 1]
data['Ifo'] = data['Value'].ffill()
data['Ifo'] = data['Ifo'].bfill()
data = data.drop(columns=["Value","DateTime"], axis=1)



rate = pd.read_csv(rate)
rate = rate[["DateTime","Value"]]
rate['Date'] = pd.to_datetime(rate['DateTime'])
rate['Date'] = rate['Date'].dt.strftime('%Y-%m-%d')
data.set_index('Date', inplace=True)
rate.set_index('Date', inplace=True)
data = pd.concat([data, rate], axis=1, join='outer')
data.reset_index(inplace=True)

first_nan_index = data['Close'].isna().idxmax()
data = data.loc[:first_nan_index - 1]
data['interest'] = data['Value'].ffill()
data['interest'] = data['interest'].bfill()
data = data.drop(columns=["Value","DateTime"], axis=1)







data.set_index('Date', inplace=True)
infla.set_index('Date', inplace=True)

result = pd.concat([data, infla], axis=1, join='outer')
result.reset_index(inplace=True)

first_nan_index = result['Close'].isna().idxmax()
data = result.loc[:first_nan_index - 1]
data['inf'] = data['Value'].ffill()
data['inf'] = data['inf'].bfill()
data = data.drop(columns=["Value"], axis=1)




copper_usd = pd.read_csv(copper_usd,sep=';')
copper_usd['Date'] = pd.to_datetime(copper_usd['Date'], format='%d/%m/%Y')
copper_usd['Date'] = copper_usd['Date'].dt.strftime('%Y-%m-%d')

oil = pd.read_csv(oil, sep=';')
oil['date'] = pd.to_datetime(oil['Date'], format='%d/%m/%Y')
oil['Date'] = oil['date'].dt.strftime('%Y-%m-%d')


gold = pd.read_csv(gold, sep=';')
gold['date'] = pd.to_datetime(gold['Date'], format='%d/%m/%Y')
gold['Date'] = gold['date'].dt.strftime('%Y-%m-%d')


eu_usd = pd.read_csv(eu_usd,sep=';')
eu_usd['Date'] = pd.to_datetime(eu_usd['Date'], format='%d/%m/%Y')
eu_usd['Date'] = eu_usd['Date'].dt.strftime('%Y-%m-%d')

bond_2 = pd.read_csv(bond_2,sep=';')
bond_10 = pd.read_csv(bond_10,sep=';')
bond_2['Date'] = pd.to_datetime(bond_2['Date'], format='%d/%m/%Y')
bond_2['Date'] = bond_2['Date'].dt.strftime('%Y-%m-%d')



bond_10['Date'] = pd.to_datetime(bond_10['Date'], format='%d/%m/%Y')
bond_10['Date'] = bond_10['Date'].dt.strftime('%Y-%m-%d')



data = pd.merge(data, copper_usd[["Close","Date"]], on='Date', how='left')
data["co_usd"]=data["Close_y"]
data["Close"]=data["Close_x"]
data = data.drop("Close_y", axis=1)
data = data.drop("Close_x", axis=1)

data["co_usd"] = data["co_usd"].ffill()
last_nan_index = data["co_usd"][data["co_usd"].isna()].index[-1]

data = data.loc[last_nan_index+1:]

data = pd.merge(data, oil[["Close","Date"]], on='Date', how='left')
data["oil_usd"]=data["Close_y"]
data = data.drop("Close_y", axis=1)
data["Close"]=data["Close_x"]
data = data.drop("Close_x", axis=1)
data["oil_usd"] = data["oil_usd"].ffill()

data = pd.merge(data, gold[["Close","Date"]], on='Date', how='left')
data["gold_usd"]=data["Close_y"]
data = data.drop("Close_y", axis=1)
data["Close"]=data["Close_x"]
data = data.drop("Close_x", axis=1)
data["gold_usd"] = data["gold_usd"].ffill()

data = pd.merge(data, bond_2[["Close","Date"]], on='Date', how='left')
data["bond_2"]=data["Close_y"]
data = data.drop("Close_y", axis=1)
data["Close"]=data["Close_x"]
data = data.drop("Close_x", axis=1)

data = pd.merge(data, bond_10[["Close","Date"]], on='Date', how='left')
data["bond_10"]=data["Close_y"]
data = data.drop("Close_y", axis=1)
data["Close"]=data["Close_x"]
data = data.drop("Close_x", axis=1)
data["bond_10"] = data["bond_10"].ffill()

data = pd.merge(data, eu_usd[["Close","Date"]], on='Date', how='left')
data["eu_usd"]=data["Close_y"]
data = data.drop("Close_y", axis=1)
data["Close"]=data["Close_x"]
data = data.drop("Close_x", axis=1)

df = data


df['SMA_20'] = df['Close'].rolling(window=20).mean()

df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

def calculate_RSI(data, window=14):
    delta = data.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI_14'] = calculate_RSI(df['Close'])

df['Middle_Band'] = df['Close'].rolling(window=20).mean()
df['Upper_Band'] = df['Middle_Band'] + (df['Close'].rolling(window=20).std() * 2)
df['Lower_Band'] = df['Middle_Band'] - (df['Close'].rolling(window=20).std() * 2)

df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

def calculate_stochastic_oscillator(df, window=14):
    df['Low_14'] = df['Close'].rolling(window=window).min()
    df['High_14'] = df['Close'].rolling(window=window).max()
    df['%K'] = (df['Close'] - df['Low_14']) * 100 / (df['High_14'] - df['Low_14'])
    df['%D'] = df['%K'].rolling(window=3).mean()

calculate_stochastic_oscillator(df)


df['MFM'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])

df['MFV'] = df['MFM'] * df['Volume']

period = 20
df['CMF'] = df['MFV'].rolling(window=period).sum() / df['Volume'].rolling(window=period).sum()
df.drop(['MFM', 'MFV'], axis=1, inplace=True)
last_nan_index = df["CMF"][df["CMF"].isna()].index[-1]
df = df.loc[last_nan_index+1:]
def capitalize_column_names(df):
    df.columns = [col.capitalize() if col and not col[0].isupper() else col for col in df.columns]
    return df

df = capitalize_column_names(df)

df.to_csv('/Users/macintosh/PycharmProjects/Script/sentiment_driven DAX predictor/data/raw_data/dax_features.csv', index=False)

"""This Process Describes the Creation of the Dataset"""