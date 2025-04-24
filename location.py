
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("location.csv")


#value_counts => count the number of times the country
top_countries=df['Country'].value_counts().head(100)
top_countries.plot(kind='bar',color='skyblue',title='Top 10 countries')
plt.xlabel('Country')
plt.ylabel('Number of the post')
#tight_layout se acha layout aata
plt.tight_layout()
plt.show()

# 4. Average retweets per country (top 10)
df.groupby('Country')['Retweets'].mean().sort_values(ascending=False).head(10).plot(kind='bar', color='orange')
plt.title('Average Retweets per Country')
plt.xlabel('Country')
plt.ylabel('Average Retweets')
plt.tight_layout()
plt.show()

df.groupby('Country')['Likes'].mean().sort_values(ascending=False).head(10).plot(kind='bar',color='green')
plt.title('Average Likes per Country ')
plt.xlabel('Country')
plt.ylabel('Average Likes')
plt.tight_layout()
plt.show()

df.groupby('Country')['Day'].mean().sort_values(ascending=False).head(20).plot(kind='line',color='red')
plt.title('Average Likes per Country ')
plt.xlabel('Country')
plt.ylabel('Average Likes')
plt.tight_layout()
plt.show()

sentiments={'Positive':1,'Negative':-1,'Neutral':0}

df['SentimentsScore']=df['Sentiment'].map(sentiments)

df.groupby('Country')['SentimentsScore'].mean().sort_values(ascending=False).head(30).plot(kind='line',color='blue')
plt.title('Average Sentiment per COuntry')
plt.xlabel('Country')
plt.ylabel('Average Sentiment')
plt.tight_layout()
plt.show()


df['Country'].value_counts().head(5).plot(kind='pie',autopct="%1.1f%%",title="Pie chart",ylabel="")
plt.axis("equal")
plt.tight_layout()
plt.show()


