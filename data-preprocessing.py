'''
Chapter 2 - Data Processing / Feature Engineering
'''
import pandas as pd
import os
from dotenv import load_dotenv
from imblearn.under_sampling import RandomUnderSampler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 

load_dotenv()

csv_file_path = os.getenv("CSV_FILE_PATH")
credit_card_data = pd.read_csv(csv_file_path)
print(credit_card_data.head())

print("Valid transactions: ")
print("Fraud transactions: ")
print(credit_card_data[credit_card_data["Class"] == 0].shape[0])
print(credit_card_data[credit_card_data["Class"] == 1].shape[0])
# from the information gained about the dataset during the data exploration stage, there is no missing values. Therefore, this does not need to be handled.

# 1. Balance the dataset - from the dataset exploration stage, it was gathered that the large majority of the transactions were valid as fraudulent transactions are a rarity. Therefore, trying to find trends with  fraudelent transactions is difficult because it will be overpowered by the majority class. To get more accurate trends and less bias, there must be a more balanced dataset.
# There are two ways to do this: unsampling the majority class or oversampling the minority class. Due to the nature of this dataset I will be unsampling the majority class, although this means that there is a chance that valuable data could be lost in this process, I do not want to add synthetic data to the dataset. There are other methods such as class weight adjustment, which neither, but because the dataset being used is so unbalanced, there will be an affect on thresult. The result of this format will make the dataset suitable for being trained in a machine learning model
# this is the tool that I will be using for under sampling:
# https://imbalanced-learn.org/stable/under_sampling.html
# class is the target class which will be resampled to balance the dataset
# FEATURES
X = credit_card_data.drop(['Class'], axis = 1) 
# TARGET
y = credit_card_data["Class"]


under_sampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = under_sampler.fit_resample(X, y)

resampled_data = pd.concat([pd.DataFrame(X_resampled, columns=X.columns),
                            pd.Series(y_resampled, name='Class')], axis=1)

resampled_data = resampled_data[credit_card_data.columns]
print("Resampled data: ")
print(resampled_data.head())

# 2. visualising the new resampled data to see if any trends ammerge
# Based on the visualisation of this data - there is no trends between intermediate classes nor amount/time
sns.scatterplot(data=resampled_data[resampled_data["Class"] == 0], x="Time", y="Amount", color="red")
sns.scatterplot(data=resampled_data[resampled_data["Class"] == 1], x="Time", y="Amount", color="green")
plt.title("Resampled Data : Time vs Amount")
plt.show()
plt.close()

# correlation analysis
correlation_matrix = credit_card_data.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, square=True, cmap='coolwarm')
plt.show()
plt.close()

print("Scaling/Dimension reduction...")
# 2. treating fraud detection as anomoly detection - an alternative method of looking at trends rather than binary classification - this will be done on the resampled dataset
from sklearn.preprocessing import StandardScaler

# scaling the features - adjusting the numerical features to fall within a distribution
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# dimnensional reduction - to visualise patterns more clearly
# reducing the dimension (features) while preserving the relationships between data
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import confusion_matrix, classification_report

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Density based clustering is my choice for anonomly detection as the data has already been partioned when the dimensionality was reduced usign PCA  - this method identifies high-density clusters and labels low-density regions as outliers
# visualising fraud and non-fraud clusters from the dimensionally reduced data set
print("Density based clustering...")
dbscan = DBSCAN(eps=0.5, min_samples=5)
y_prediction = dbscan.fit_predict(X_pca)

y_prediction_binary = [1 if x == -1 else 0 for x in y_prediction_binary]

# compare predicted anomolies against the actual transactions labelled as fraud
from sklearn.metrics import confusion_matrix
c_matrix = confusion_matrix(y, y_prediction_binary)
print("confusion matrix")
print(c_matrix)

# other metics to identify the quality of the clustering anomoly detection
print(classification_report(y, y_prediction_binary))

# visualising the clusters that are detecting fraudulent transactions through anomolies
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_prediction_binary, cmap='coolwarm', alpha=0.7)
plt.title("Anomaly Detection with PCA")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()
plt.close()

