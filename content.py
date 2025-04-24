import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud

df=pd.read_csv('youtube_womens_safety_full_data(in).csv')

#this drops the null values
df=df.dropna(subset=['title'])


vectorizer=TfidfVectorizer(max_features=10,stop_words='english')

X=vectorizer.fit_transform(df['title'])

keyword=vectorizer.get_feature_names_out()
print(keyword)


#LDA kar
lda=LatentDirichletAllocation(random_state=42,n_components=3)
lda.fit(X)

for idx,topic in enumerate(lda.components_):
    top_words = vectorizer.get_feature_names_out()[topic.argsort()[-10:][::-1]]
    print(f"Topic #{idx}: {', '.join(top_words)}")


#word cloud
wordcloud=WordCloud(width=800,height=400,background_color='white').generate(' '.join(df['title']))
plt.imshow(wordcloud, interpolation='bilinear')
plt.show()

keywords = vectorizer.get_feature_names_out()
scores = X.toarray().sum(axis=0)

plt.figure(figsize=(8, 5))
sns.barplot(x=scores, y=keywords, palette='mako')
plt.title('Top Keywords in Video Titles')
plt.xlabel('TF-IDF Score')
plt.ylabel('Keywords')
plt.tight_layout()
plt.show()


