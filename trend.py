import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
df=pd.read_csv("SmaData.csv")

print(df.head(10))

#date conversion
df['Post_Date']=pd.to_datetime(df['Post_Date'])


df['Year']=df['Post_Date'].dt.year
df['Month']=df['Post_Date'].dt.month
df['Day']=df['Post_Date'].dt.day

print(df['Day'].head(10))

print(df['Month'].value_counts().head(10))

df.groupby('Month')['Likes'].mean().sort_values().head(100).plot(kind='line',color='blue')
plt.title("Likes per month")
plt.xlabel("Months")
plt.ylabel("Likes")
plt.tight_layout()
plt.show()

df.groupby('Month')['Comments'].mean().sort_values(ascending=False).head(150).plot(kind='bar',color='orange')
plt.title("Comments per year")
plt.xlabel("Month")
plt.ylabel("Comments")
plt.tight_layout()
plt.show()

df.groupby('Month')['Comments'].mean().sort_values(ascending=False).head(150).plot(kind='pie',autopct="%1.1f%%",color='orange')
plt.title("Comments per year")
plt.xlabel("Month")
plt.ylabel("Comments")
plt.tight_layout()
plt.show()

heatmap_data=df.groupby(['Month','Day'])['Likes'].mean().unstack()
sns.heatmap(heatmap_data)
plt.title("Heatmap")
plt.ylabel("Month")
plt.xlabel("Average likes ")
plt.tight_layout()
plt.show()

plt.scatter(df['Likes'], df['Comments'], alpha=0.5, color='green')
plt.title("Likes vs Comments")
plt.xlabel("Likes")
plt.ylabel("Comments")
plt.grid(True)
plt.tight_layout()
plt.show()

