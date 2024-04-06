import pandas as pd
import matplotlib.pyplot as plt

filepath = 'raw_databases/tweets_remaining.csv'

stocks = ['#SPX500','#SP500','SPX500','$SPX',"$MSFT", "$AAPL", "$AMZN", "$FB", "$BBRK.B", "$GOOG", "$JNJ", "$JPM", "$V", "$PG", "$MA", "$INTC", "$UNH", "$BAC", "$T", "$HD", "$XOM","$DIS", "$VZ", "$KO", "$MRK", "$CMCSA", "$CVX", "$PEP", "$PFE"]

tweets = pd.read_csv(filepath, error_bad_lines=False, delimiter=';')

tweets['created_at'] = pd.to_datetime(tweets['created_at'].apply(lambda x: x.split()[0]))

for i in stocks:
    tweets[i] = tweets["full_text"].str.contains(i, regex=False)

print(tweets)


filtered = tweets.groupby("created_at").agg({i:'sum' for i in stocks})

print(filtered.columns)
print(filtered["#SPX500"])
print(filtered["#SP500"])
print(filtered["SPX500"])
print(filtered["$SPX"])
filtered["$SPY"] = filtered[["#SPX500","#SP500","SPX500","$SPX"]].sum(axis=1)

filtered = filtered.drop(["#SPX500","#SP500","SPX500","$SPX"], axis=1)

print(filtered)

filtered.replace(False,0)

filtered.to_csv('cleanedtwitterdata.csv', index=True)

# plt.figure();
# filtered.plot();
# plt.show()