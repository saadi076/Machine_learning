import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(r"C:\Users\DELL\Desktop\ML dataset\LogisticRegression\Social_Network_Ads.csv")

print(data.head(10))
le = LabelEncoder()
data['Gender_Encoded'] = le.fit_transform(data['Gender'])

# Select features for clustering
X = data[['Gender_Encoded', 'Age', 'EstimatedSalary']]

# Apply KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

# Plot the clusters
plt.figure(figsize=(10, 6))
plt.scatter(data['Age'], data['EstimatedSalary'], c=data['Cluster'], cmap='viridis', edgecolors='k')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.title('KMeans Clusters')

# Plot the cluster centers
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 1], centers[:, 2], c='red', s=200, alpha=0.75, marker='x')

plt.show()
