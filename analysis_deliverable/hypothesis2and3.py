import pandas as pd
import numpy as np
import sqlite3
from collections import Counter
import nltk
from datetime import datetime, timedelta
from scipy.stats import ttest_1samp, ttest_ind, ttest_rel, chi2_contingency
# from util import train_test_split


def run_hypotheis_2_and_3():
    df = pd.read_csv('../raw_databases/reddit_wsb.csv')
    df = df.drop(columns=['created', 'id', 'url', 'comms_num'])
    df['body'] = df['body'].fillna("")
    df['text'] = df['title'] + ' ' + df['body']
    df = df.drop(columns=['body', 'title'])
    df['timestamp'] = df['timestamp'].apply(lambda x: x[0:10])


    # stock = (pd.read_csv('../input/amex-nyse-nasdaq-stock-histories/all_symbols.txt').iloc[:, 0]).to_list()
    stock = ['GME', 'AAL', 'AAPL', 'AMD', 'APHA', 'BILI',
             'CLOV', 'DKNG', 'ECOR', 'FB', 'INO', 'JD', 'MSFT',
             'MVIS', 'PLUG', 'CENN', 'SNDL', 'TLRY', 'TSLA', 'WKHS', 'ZM']

    stock_lower = [x.lower() for x in stock]
    words = ["an", "the", "of", "and", "a", "to", "in", "is", "you", "that", "it", "he", "was", "for", "on", "are",
             "as", "with", "his", "they", "I", "my", "than", "first", "water", "been",
             "call", "who", "oil", "its", "now", "find", "long", "down", "day", "did", "get", "come", "made", "may",
             "part", "some", "her", "would", "make", "like", "him", "into", "time", "has", "look", "two", "more",
             "write",
             "go", "see", "number", "no", "way", "could", "people", "there", "use", "an", "each", "which", "she", "do",
             "how", "their", "if", "will", "up", "other", "about", "out", "many", "then", "them", "these", "so", "at",
             "be",
             "this", "have", "from", "or", "one", "had", "by", "word", "but", "not", "what", "all", "were", "we",
             "when", "your", "can", "said"]

    for element in stock_lower:
        if element in words:
            stock_lower.remove(element)

    for i in stock:
        df[i] = df["text"].str.contains(i, regex=False, case=False)

    group_by_timestamp = df.groupby("timestamp").agg({i: 'sum' for i in stock})

    # we changed apha to aph and fb to meta
    lower_names = ['gme', 'aal', 'aapl', 'amd', 'aph', 'bili', 'clov', 'dkng', 'ecor', 'meta', 'ino', 'jd', 'msft',
                   'mvis', 'cenn', 'plug', 'sndl', 'tlry', 'tsla', 'wkhs', 'zm']
    group_by_timestamp.columns = lower_names
    group_by_timestamp.drop_duplicates(inplace=True)


    # twitter and ticker data
    twitter_dataframe = pd.read_csv('../cleaning_scripts/cleanedtwitterdata.csv')
    twitter_dataframe.drop_duplicates(inplace=True)
    twitter_dataframe.drop(columns=['$BBRK.B'], inplace=True)

    column_names = twitter_dataframe.columns
    twitter_dataframe.columns = [x.replace('$', '').lower() for x in column_names]
    twitter_dataframe.rename(columns={'fb': 'meta'}, inplace=True)

    # Insert stock data
    yahoo_1_dataframe = pd.read_csv('../cleaning_scripts/yahoo_stock_1.csv')
    yahoo_1_dataframe.drop_duplicates(inplace=True)
    yahoo_2_dataframe = pd.read_csv('../cleaning_scripts/yahoo_stock_2.csv')
    yahoo_2_dataframe.drop_duplicates(inplace=True)


    ################################################################################
    ################################################################################
    # GENERATING ANALYSIS DATA

    # HYPOTHESIS 2
    reddit_df = reddit_generate_pairs(group_by_timestamp, yahoo_2_dataframe)
    reddit_df = reddit_merge_volatility(reddit_df, yahoo_2_dataframe)
    red_day_plus_one_col = list(reddit_df['dayplus1vol'])
    red_play_minus_one_col = list(reddit_df['dayminus1vol'])

    red_day_plus_one_col = [abs(x) for x in red_day_plus_one_col]
    red_play_minus_one_col = [abs(x) for x in red_play_minus_one_col]

    twitter_timestamp_stock_pairs = twitter_generate_pairs(twitter_dataframe, yahoo_1_dataframe)
    twitter_timestamp_stock_pairs = twitter_merge_volatility(twitter_timestamp_stock_pairs, yahoo_1_dataframe)
    twit_day_plus_one_col = list(twitter_timestamp_stock_pairs['dayplus1vol'])
    twit_play_minus_one_col = list(twitter_timestamp_stock_pairs['dayminus1vol'])

    full_plus_one = red_day_plus_one_col + twit_day_plus_one_col
    full_minus_one = red_play_minus_one_col + twit_play_minus_one_col
    full_plus_one = [abs(x) for x in full_plus_one]
    full_minus_one = [abs(x) for x in full_minus_one]

    tstats, pvalue = ttest_rel(full_plus_one, full_minus_one)

    # """
    print("HYPOTHESIS 2")
    print("T-statistics: ", tstats)
    print("p-value: ", pvalue)
    print("p-value < 0.05", pvalue < 0.05)
    print()
    # """

    ################################################################################
    # HYPOTHESIS 3
    reddit_df = reddit_generate_pairs(group_by_timestamp, yahoo_2_dataframe)
    reddit_df = reddit_merge_volatility(reddit_df, yahoo_2_dataframe)
    smaller_reddit_df = reddit_df[(reddit_df['stock'] == 'aapl') |
                                  (reddit_df['stock'] == 'meta') |
                                  (reddit_df['stock'] == 'msft')]
    plus_one_red_list = list(smaller_reddit_df['dayplus1vol'])

    twitter_df = twitter_timestamp_stock_pairs
    smaller_twitter_df = twitter_df[(twitter_df['stock'] == 'aapl') |
                                    (twitter_df['stock'] == 'meta') |
                                    (twitter_df['stock'] == 'msft')]

    list_twit = list(smaller_twitter_df['dayplus1vol'])

    tstats, pvalue = ttest_ind(plus_one_red_list, list_twit)
    print("HYPOTHESIS 3")
    print("ttest: " + str(tstats))
    print("pvalue: " + str(pvalue))

    ################################################################################
    ################################################################################

    return (df)


def reddit_generate_pairs(df, yahoo_2_dataframe):
    mean = list(df.mean(axis=0))
    std = list(df.std())

    list_data = df.values
    stock_list = df.columns
    timestamps = df.index

    build_time_stock = []
    for row in range(len(list_data)):  # for each row
        for col in range(len(list_data[0])):  # for each col
            num_mentions = list_data[row][col]
            if num_mentions > std[col] + mean[col]:
                build_time_stock.append([timestamps[row], stock_list[col], num_mentions])

    new_df = pd.DataFrame(build_time_stock, columns=['timestamp', 'stock', 'num_mentions'])
    return new_df


def reddit_merge_volatility(new_df, yahoo_2_dataframe):
    new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])
    yahoo_2_dataframe['Date'] = pd.to_datetime(yahoo_2_dataframe['Date'])
    new_df['dayplus1'] = new_df['timestamp'].apply(lambda x: x + timedelta(days=1))
    new_df['dayminus1'] = new_df['timestamp'].apply(lambda x: x - timedelta(days=1))
    new_df['dayplus1vol'] = ''
    new_df['dayminus1vol'] = ''

    for index, rows in new_df.iterrows():
        if (rows["stock"] != 'fb' and rows["stock"] != 'nakd' and
                rows["stock"] != 'apha'):
            volatility = yahoo_2_dataframe.loc[yahoo_2_dataframe["Date"] == rows["dayplus1"], [rows["stock"]]]
            if (len(volatility.values) != 0):
                new_df.loc[index, "dayplus1vol"] = volatility.values[0, 0]
            else:
                new_df.loc[index, "dayplus1vol"] = None

        if (rows["stock"] != 'fb' and rows["stock"] != 'nakd' and
                rows["stock"] != 'apha'):
            volatility = yahoo_2_dataframe.loc[yahoo_2_dataframe["Date"] == rows["dayminus1"], [rows["stock"]]]
            if (len(volatility.values) != 0):
                new_df.loc[index, "dayminus1vol"] = volatility.values[0, 0]
            else:
                new_df.loc[index, "dayminus1vol"] = None

    # remove rows were volatility doesn't exist (and is NaN)
    new_df = new_df.dropna()
    return new_df


def twitter_generate_pairs(df, yahoo_1_dataframe):
    # formatting for twitter is different than for reddit so we had to drop
    # the timestamps column
    timestamps = list(df.loc[:, "created_at"])
    df.drop(columns=df.columns[0], axis=1, inplace=True)

    mean = list(df.mean(axis=0))
    std = list(df.std())

    list_data = df.values
    stock_list = df.columns

    build_time_stock = []
    for row in range(len(list_data)):  # for each row
        for col in range(len(list_data[0])):  # for each col
            num_mentions = list_data[row][col]
            if num_mentions > std[col] + mean[col]:
                build_time_stock.append([timestamps[row], stock_list[col], num_mentions])

    new_df = pd.DataFrame(build_time_stock, columns=['timestamp', 'stock', 'num_mentions'])
    return new_df


def twitter_merge_volatility(new_df, yahoo_1_dataframe):
    # NATHAN CODE BEGINS HERE
    new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])
    yahoo_1_dataframe['Date'] = pd.to_datetime(yahoo_1_dataframe['Date'])
    new_df['dayplus1'] = new_df['timestamp'].apply(lambda x: x + timedelta(days=1))
    new_df['dayminus1'] = new_df['timestamp'].apply(lambda x: x - timedelta(days=1))
    new_df['dayplus1vol'] = ''
    new_df['dayminus1vol'] = ''

    for index, rows in new_df.iterrows():
        if (rows["stock"] != 'fb' and rows["stock"] != 'nakd' and
                rows["stock"] != 'apha' and rows["stock"] != 'spy'):
            volatility = yahoo_1_dataframe.loc[yahoo_1_dataframe["Date"] == rows["dayplus1"], [rows["stock"]]]
            if (len(volatility.values) != 0):
                new_df.loc[index, "dayplus1vol"] = volatility.values[0, 0]
            else:
                new_df.loc[index, "dayplus1vol"] = None
        else:
            new_df.loc[index, "dayplus1vol"] = None

        if (rows["stock"] != 'fb' and rows["stock"] != 'nakd' and
                rows["stock"] != 'apha' and rows["stock"] != 'spy'):
            volatility = yahoo_1_dataframe.loc[yahoo_1_dataframe["Date"] == rows["dayminus1"], [rows["stock"]]]
            if (len(volatility.values) != 0):
                new_df.loc[index, "dayminus1vol"] = volatility.values[0, 0]
            else:
                new_df.loc[index, "dayminus1vol"] = None
        else:
            new_df.loc[index, "dayplus1vol"] = None

    # remove rows were volatility doesn't exist (and is NaN)
    new_df = new_df.dropna()

    return new_df


run_hypotheis_2_and_3()