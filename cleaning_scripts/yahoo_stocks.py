import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
#from sqlalchemy import create_engine


import yfinance as yf

#pull history data for stocks
df = pd.DataFrame()
df3 = pd.DataFrame()
securities = ['msft', 'aapl', 'tsla', 'amzn', 'meta', 'brk-b', 'goog', 'jnj', 'jpm', 'v', 'pg',
             'ma', 'intc', 'unh', 'bac', 't', 'hd', 'xom', 'dis', 'vz', 'ko', 'mrk', 'cmcsa',
             'cvx', 'pep', 'pfe', '^gspc']

for security in securities:
   df[security] = ((yf.Ticker(security).history(start='2020-04-08', end = '2020-07-17').Close\
       - yf.Ticker(security).history(start='2020-04-08', end = '2020-07-17').Open)\
        /yf.Ticker(security).history(start='2020-04-08', end = '2020-07-17').Close) * 100
   df3[security] = yf.Ticker(security).history(start='2020-04-08', end = '2020-07-17')['Volume']

df.to_csv('intermediate1.csv', index = True)
df3.to_csv('intermediate3.csv', index = True)
df = pd.read_csv('intermediate1.csv')
df3 = pd.read_csv('intermediate3.csv')

df['Date'] = pd.to_datetime(df['Date'].apply(lambda x: x.split()[0]))
df3['Date'] = pd.to_datetime(df3['Date'].apply(lambda x: x.split()[0]))
df.set_index('Date', inplace = True)
df3.set_index('Date', inplace = True)

df2 = pd.DataFrame()
df4 = pd.DataFrame()
securities2 = ['gme', 'aal', 'aapl', 'amd', 'aph', 'bili', 'clov', 'dkng', 'ecor', 'meta', 'ino', 'jd', 'msft',
               'mvis', 'cenn', 'plug', 'sndl', 'tlry', 'tsla', 'wkhs', 'zm']
for security in securities2:
   df2[security] = ((yf.Ticker(security).history(start='2021-01-27', end = '2021-08-18').Close\
       - yf.Ticker(security).history(start='2021-01-27', end = '2021-08-18').Open)\
        /yf.Ticker(security).history(start='2021-01-27', end = '2021-08-18').Close) * 100
   df4[security] = yf.Ticker(security).history(start='2021-01-27', end = '2021-08-18')['Volume']

#modify date column and set back to index
df2.to_csv('intermediate2.csv', index = True)
df4.to_csv('intermediate4.csv', index = True)
df2 = pd.read_csv('intermediate2.csv')
df4 = pd.read_csv('intermediate4.csv')

df2['Date'] = pd.to_datetime(df2['Date'].apply(lambda x: x.split()[0]))
df4['Date'] = pd.to_datetime(df4['Date'].apply(lambda x: x.split()[0]))
df2.set_index('Date', inplace = True)
df4.set_index('Date', inplace = True)

#save to csv 
df.to_csv('yahoo_stock_1.csv')
df2.to_csv('yahoo_stock_2.csv')
df3.to_csv('stock1volume.csv')
df4.to_csv('stock2volume.csv')


