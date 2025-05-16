# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import dataset and print head,info of the dataset
2.Check for null values
3.Import kmeans and fit it to the dataset
4.Plot the graph using elbow method
5..Print the predicted array
6.Plot the customer segments

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Rishab p doshi
RegisterNumber:  212224240134
*/
```
```
# importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
# load the dataset
data = pd.read_csv('Mall_Customers.csv')
# display first five records
print(data.head())
# display dataframe information
print(data.info())
# display the count of null values in each column
print(data.isnull().sum())
# using Elbow Method to determine optimal number of clusters
from sklearn.cluster import KMeans
# within-cluster sum of squares
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++")
    # using 'Annual Income' and 'Spending Score'
    kmeans.fit(data.iloc[:, 3:])
    wcss.append(kmeans.inertia_)

# plot the Elbow graph
plt.plot(range(1, 11), wcss)
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()
# fitting KMeans to the data with 5 clusters
km = KMeans(n_clusters=5)
print( km.fit(data.iloc[:, 3:]))
# predicting cluster labels for each record
y_pred = km.predict(data.iloc[:, 3:])
print(y_pred)
# adding predicted cluster labels to the dataframe
data["cluster"] = y_pred
# splitting data into clusters for visualization
df0 = data[data["cluster"] == 0]
df1 = data[data["cluster"] == 1]
df2 = data[data["cluster"] == 2]
df3 = data[data["cluster"] == 3]
df4 = data[data["cluster"] == 4]
# visualizing customer segments using scatter plot
plt.scatter(df0["Annual Income (k$)"], df0["Spending Score (1-100)"], c="red", label="cluster0")
plt.scatter(df1["Annual Income (k$)"], df1["Spending Score (1-100)"], c="black", label="cluster1")
plt.scatter(df2["Annual Income (k$)"], df2["Spending Score (1-100)"], c="blue", label="cluster2")
plt.scatter(df3["Annual Income (k$)"], df3["Spending Score (1-100)"], c="green", label="cluster3")
plt.scatter(df4["Annual Income (k$)"], df4["Spending Score (1-100)"], c="magenta", label="cluster4")
plt.legend()
plt.title("Customer Segments")
plt.show()
```
## Output:
![image](https://github.com/user-attachments/assets/3580aae3-6af9-4a6b-ab04-6f2b8b002e69)

![image](https://github.com/user-attachments/assets/80757dd5-18f5-4c8f-ae83-fa157ee0bf93)
![image](https://github.com/user-attachments/assets/37ae5f9e-c8fc-4f2f-9cc3-4e3b6b175f23)

![image](https://github.com/user-attachments/assets/ee05d16e-1e0a-49fc-a554-326e683c5cec)

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
