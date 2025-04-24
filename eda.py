
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



df = pd.DataFrame(data)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 1. Boxplot for Likes
sns.boxplot(data=df, y='likes')
plt.title("Boxplot of Likes")
plt.show()

# 2. Boxplot for Shares
sns.boxplot(data=df, y='shares')
plt.title("Boxplot of Shares")
plt.show()

# 3. Barplot for Total Likes per User
likes_per_user = df.groupby('username')['likes'].sum().reset_index()
sns.barplot(data=likes_per_user, x='username', y='likes')
plt.title("Total Likes per User")
plt.show()

# 4. Scatter Plot (Likes vs. Shares)
sns.scatterplot(data=df, x='likes', y='shares', hue='username')
plt.title("Likes vs Shares")
plt.show()

# 5. Line Plot of Likes over Time
sns.lineplot(data=df.sort_values('timestamp'), x='timestamp', y='likes', marker='o')
plt.title("Likes over Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
