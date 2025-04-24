import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob

df=pd.read_csv("youtube_womens_safety_full_data(in).csv")

df=df.dropna(subset=['transcript'])

# function for the polarity
def get_sentiment(text):
    # fir text blob use karo
    blob=TextBlob(str(text))
    #blob me sentiment and polarity hai usko return kardo
    return blob.sentiment.polarity


df['polarity']=df['transcript'].apply(get_sentiment)

def sentiment(polarity):
    if polarity>0:
        return 'Positive'
    elif polarity<0:
        return 'Negative'
    else:
        return 'Neutral'

#make a sentiment column
df['Sentiment']=df['polarity'].apply(sentiment)

sentiment_count=df['Sentiment'].value_counts()
sentiment_count.plot(kind="bar",color="red")
plt.show()

x=df['Sentiment']
y=df['likes']
plt.bar(x, y, color='skyblue')

plt.show()

y=df['Sentiment']
x=df['country']
plt.bar(x, y)
plt.show()

x1=df['Sentiment']
y2=df['views']
plt.pie(x=y2,labels=x1,autopct="%1.1f%%")
plt.show()

