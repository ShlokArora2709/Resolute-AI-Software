import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

train=pd.read_excel('train.xlsx')

scaler = StandardScaler()
train_sc = scaler.fit_transform(train.drop(['target'],axis=1))

kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
kmeans.fit(train_sc)

def predict_cluster(data_point):
    data_point_sc=scaler.transform(data_point)
    cluster=kmeans.predict(data_point_sc)
    return cluster[0]

new_data_point = np.array([[-67, -69, -64, -64, -59, -53, -71, -72, -71, -60, -61, -58, -54, -73, -60, -66, -72, -79 ]])
predicted_cluster = predict_cluster(new_data_point)
print("Predicted Cluster for New Data Point:", predicted_cluster)