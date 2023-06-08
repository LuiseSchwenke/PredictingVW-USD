import mwclient
import time
import pandas as pd
from datetime import datetime
from statistics import mean
from transformers import pipeline

# first: getting the Wikipedia comments on the page of Wikipedia->Volkswagen (VW)
site = mwclient.Site("en.wikipedia.org")
page = site.pages["Volkswagen"]
# order entries starting with the earliest entry to find, so it later fits with the
# stock price dataframe of VW
revs = list(page.revisions())
# print(revs[0])
revs = sorted(revs, key=lambda rev: rev["timestamp"])
# print(revs[0])

# second: sentimental analysis with transformers
sent_pip = pipeline("sentiment-analysis")


# text length here limited to 250 Char, otherwise it will take forever to create de edits dictionary
# than take first element from the response list (dict with keys ["score"] and ["label"]) and turn
# texts with label "negative" into a neg. number of the score
def find_sentiment(text):
    sentiment = sent_pip([text[:250]])[0]
    score = sentiment["score"]
    if sentiment["label"] == "NEGATIVE":
        score *= -1
    return score


# applying the sentiment function to the wikipedia comments: add date of the comments to dict with a
# lists of the sentiment counts and amount of edits on an entry on that days, increment this with every
# time the loop goes through the entry
# finally run the comment through the get_sentiment function and append the outcome number to the list of
# sentiments
edits = {}
for rev in revs:
    date = time.strftime("%Y-%m-%d", rev["timestamp"])

    if date not in edits:
        edits[date] = dict(sentis=list(), edit_count=0)

    edits[date]["edit_count"] += 1

    comment = rev.get("comment", "")
    edits[date]["sentis"].append(find_sentiment(comment))

# getting the mean value of the sentiments for each day and percentage of negative/ positive sentiment
# pos. sentient = 0, neg. sentiments = neg. number
for key in edits:
    if len(edits[key]["sentis"]) > 0:
        edits[key]["each_sentiment"] = mean(edits[key]["sentis"])

        # percentage of negative comments
        edits[key]["neg_sentiment"] = len([s for s in edits[key]["sentis"] if s < 0]) / len(edits[key]["sentis"])
    else:
        edits[key]["each_sentiment"] = 0
        edits[key]["neg_sentiment"] = 0

    del edits[key]["sentis"]

# print(edits)

# create de pandas dataframe and convert the data to pandas datetime
df = pd.DataFrame.from_dict(edits, orient="index")
df.to_csv("edits.csv", encoding='utf-8', index=False)
print(df.head())

df.index = pd.to_datetime(df.index)

# fill empty days (days with no edits on wikipedia) with 0

dates = pd.date_range(start="2001-05-03", end=datetime.today())
df = df.reindex(dates, fill_value=0)

# get the rolling average each 15 days and remove NaN and create the .csv from it
rolling_edits = df.rolling(15).mean()
rolling_edits = rolling_edits.dropna()
print(rolling_edits)
rolling_edits.to_csv("vw_edits.csv")

##################################################################################
# getting the stock prices with yfinance:

import yfinance as yf

vw_ticker = yf.Ticker("VWAGY")
vw = vw_ticker.history(period="max")
vw.columns = [c.lower() for c in vw.columns]
vw.index = vw.index.tz_localize(None)
# print(vw.head())
# print(vw_tail())

# del columns that are unnecessary for the analysis
del vw["stock splits"]
del vw["dividends"]

# for overview:
vw.plot.line(y="close", use_index=True)

# merge the wikipedia .csv into the stock dataframe:
wiki = pd.read_csv("vw_edits.csv", index_col=0, parse_dates=True)
vw = vw.merge(wiki, left_index=True, right_index=True)
# print(vw)

# to predicte the price of the next day, the closing price can be shifted back one day so the closing
# price will be the price of the new "tomorrow column (to predict prices of days further ahead, change
# -1 value
vw["tomorrow"] = vw["close"].shift(-1)

# to predict, defining price will get up with 1 or go down with 0
# result of the .value_counts is 0: 2682; 1: 2461 --> quite equal distribution, which is good
vw["predict"] = (vw["tomorrow"] > vw["close"]).astype(int)
print(vw["predict"].value_counts())

# n_estimators = individual decision trees to be trained
# min_sapmple_split = individual decicion trees should not split notes if it has <50

# first base estiation with sklearn's RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=1)
train = vw.iloc[:-200]
test = vw[-200:]

# features - here: predictors and target == "predict
predictors = ["close", "volume", "open", "high", "low", "edit_count", "each_sentiment", "neg_sentiment"]
model.fit(train[predictors], train["predict"])

# to see how the model is performing, caluclate the precision score: Result: 0.49411764705882355 which
# could be better (>0,5)
from sklearn.metrics import precision_score

precision = model.predict(test[predictors])
precision = pd.Series(precision, index=test.index)
precision_score(test["predict"], precision)

def predict(train, test, predictors, model):
  model.fit(train[predictors], train["predict"])
  precision = model.predict(test[predictors])
  precision = pd.Series(precision, index=test.index, name ="predictions")
  precision_score(test["predict"], precision)
  combined =pd.concat([test["predict"], precision], axis=1)
  return combined

# trying to improve the model with backtesting and xgboost so not only the first 200 etries are used
# for training but many years:
# start at 3 years (3*395, spets 5 month)
def backtest(data, model, predictors, start=1095, step=150):
  all_predictions=[]

  for i in range(start, data.shape[0], step):
    train = data.iloc[0:i].copy()
    test = data.iloc[i:(i+step)].copy()
    predictions = predict(train, test, predictors, model)
    all_predictions.append(predictions)
  return pd.concat(all_predictions)

from xgboost import XGBClassifier
# the lower the learning rate, the less the model is likely to overfit
model = XGBClassifier(random_state=1, learning_rate=.1, n_estimators=200)

predicts = backtest(vw, model, predictors)

# checking the precision score: unfortunately it got lower to 0.4886927744070601
precision_score(predicts["predict"], predicts["predictions"])

# to try improve the model, add more predictors, here:
def more_predictors(vw):
  horizons = [2, 7, 60, 180, 365] #days
  new_predictors =["close", "each_sentiment", "neg_sentiment" ]
  for horizon in horizons:
    rolling_averages = vw.rolling(horizon, min_periods=1).mean()

    # ratio current close price and historical close price:
    ratio_column = f"close_ratio_{horizon}"
    vw[ratio_column]= vw["close"]/ rolling_averages["close"]

    # Interest in Bitcoin measured on Wikipedia entries
    edit_column = f"edit_{horizon}"
    vw[edit_column] = rolling_averages["edit_count"]

    # all over trend (closed=left, so current target ["predict"] is not included - no data leakage)
    rolling = vw.rolling(horizon, closed="left", min_periods=1).mean()
    trend_column = f"trend_{horizon}"
    vw[trend_column] = rolling["predict"]

    new_predictors += [ratio_column, edit_column, trend_column]
  return vw, new_predictors

# checking the precision score:
# unfortunately got lower by 0.003: result=0.4855555555555556
vw, new_predictors = more_predictors(vw.copy())
predicts = backtest(vw, model, new_predictors)
precision_score(predicts["predict"], predicts["predictions"])

# to see the actual predict:
print(predicts)