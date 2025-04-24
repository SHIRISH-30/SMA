import re

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from wordcloud import WordCloud

df=pd.read_csv("datafile4.csv")

def remove_hash(text):
    return re.findall(r'#\w+',str(text))


df['Hashtags'] = df['text'].apply(remove_hash)
print(df['Hashtags'])

all_tags = ' '.join(sum(df['Hashtags'], []))
print(all_tags)

wordcloud=WordCloud(width=800,height=400,background_color='white').generate(all_tags)
plt.imshow(wordcloud,interpolation='bilinear')
plt.show()

all_hash=df['Hashtags'].value_counts()
all_hash.plot(kind='bar',color='red')
plt.show()


all_text=df['Hashtags'].apply(lambda x:', '.join(x))
print(all_text)

x=all_text
y=df['likes']
plt.bar(x,y,color='skyblue')
plt.show()