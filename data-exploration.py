import pandas as pd
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
import sweetviz as sv
from dotenv import load_dotenv
import os
import numpy as np

# loading the variables within - which contains the local file path of where the data set is stored - an alternative is to install the kaggle data set python module and load it from there
load_dotenv()

csv_file_path = os.getenv("CSV_FILE_PATH")
credit_card_data = pd.read_csv(csv_file_path)

# visualising small sample of dataset
print(credit_card_data.head())

# go into more detail on microtransactions trying to find trends, as they are sometimes used in fraud
valid_mirco = credit_card_data[(credit_card_data["Class"] == 0) & (credit_card_data["Amount"] <= 0.2)].shape[0]
fraud_mircro = credit_card_data[(credit_card_data["Class"] == 1) & (credit_card_data["Amount"] <= 0.2)].shape[0]

total_valid = credit_card_data[(credit_card_data["Class"] == 0)].shape[0]
total_fraud = credit_card_data[(credit_card_data["Class"] == 1)].shape[0]

# from the calculation below - it can be seen that microtransactions have over 5x more chance of being fraudulent compared to the percentage over the whole dataset
print("Percent of fradulent transactions: " + str((total_fraud/(total_valid + total_fraud)) * 100))
print("Percentage of fraudenlent micro-transactions" + str((fraud_mircro/(valid_mirco + fraud_mircro)) * 100))

'''
Chapter 1. 
Analysis of individual column variables
'''

# amount vs time histogram plot
# KDE=true to smooth out the data points
# from this visualisation - it is possible to identify some anomolies in the amount - however this doesnt immediately equate to fraud
time_amount = credit_card_data[["Time", "Amount"]]
# the threshold for annomily transactions is 1000, these will be highlighted in the visualisation
threshold_amount = 10000

# Base scatterplot for all data points
sns.scatterplot(data=time_amount, x="Time", y="Amount", label="All Data", alpha=0.6, color="blue")
# horizontal line to separate the anomalies
plt.axhline(y=threshold_amount, label="Threshold Amount", color="red")

plt.title("Time vs Amount")
plt.show()
plt.close()

# ANSWER THIS QUESTION USING A PIE CHART - OUT OF THE POINTS FLAGGED AS AN ANOMOLY - HOW MANY ARE FRADULENT

# frequency amount analysis for each column
# due the great disparaty between the two values - plot on seperate graphs
valid_transactions = credit_card_data[credit_card_data["Class"] == 0]
fraudulent_transactions = credit_card_data[credit_card_data["Class"] == 1]

valid_count = len(valid_transactions)
fraudulent_count = len(fraudulent_transactions)

fig, axes = plt.subplots(1, 2)
plt.suptitle("Frequency Analysis")
sns.barplot(ax=axes[0], x=["Valid"], y=[valid_count], color="green")
sns.barplot(ax=axes[1], x=["Fradulent"], y=[fraudulent_count], color="red")
plt.show()
plt.close()

# correlation analysis - looking at any correlation between the variables - if a transactions variables does not fit with the correlation this may be a potential flag for fraud
correlation_matrix = credit_card_data.corr()
# generate a mask to avoid repeat display of correlation values in the heatmap
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, square=True, cmap='coolwarm')
plt.show()
plt.close()

# looking at the pairwise relationship between two key variables of amount and time
# FOCUS ON THIS 
valid_time_amount = credit_card_data[credit_card_data["Class"] == 0][["Time", "Amount"]]
fraud_time_amount = credit_card_data[credit_card_data["Class"] == 1][["Time", "Amount"]]
fig, axes = plt.subplots(2, 2)

# both the scatterplots are visually random therefore no correlation can be deduced between in terms of the transaction being valid or fraudulent (backed up by the correlation matrix heatmap)
# FROM THIS DIAGRAM I CAN SEE THAT THE FRAUDLENT TRANSACTIONS MAY HAVE A MONOPOLY ON THE LOWER AMOUNTS - NEED TO CHECK THIS
sns.scatterplot(ax=axes[0,0], x=valid_time_amount["Time"], y=valid_time_amount["Amount"], color="green")
sns.scatterplot(ax=axes[0,1], data=fraud_time_amount, x=fraud_time_amount["Time"], y=fraud_time_amount["Amount"], color="red")
sns.kdeplot(ax=axes[1, 0], data=valid_time_amount, x="Time", y="Amount", color="green")
sns.kdeplot(ax=axes[1, 1], data=fraud_time_amount, x="Time", y="Amount", color="red")

plt.show()
plt.close()
# from the subplot (1, 1) there is an identification of an anomoly

# Anomoly detection
# from this plot, it can be seen as there are individual anominal values within the amount column
# The values within the time column are almost uniformly distributed, which is a good sign to show that there is, which could be flagged as irregular behaviour.
# However, there are extremely large anomolies (highlighted in figure 1), which are disproportionately higher than the median
# Box plot is the best visualisation graph for continous variables
time = credit_card_data["Time"]
amount = credit_card_data["Amount"]
fig, axes = plt.subplots(1, 2)

plt.suptitle("Box Plot of Time and Amount Variables")

sns.boxplot(ax=axes[0], data=time)
axes[0].set_title("Time")

sns.boxplot(ax=axes[1], data=amount)
axes[1].set_title("Amount")

plt.show()
plt.close()


# Automated statistical breakdown analysis
# looking at the amount of missing data
# from this we can see that there is not missing values of data within this dataset
missing_data = credit_card_data.isnull().sum().sum()
print("Amount of missing data points: " + str(missing_data))


# automated summary statistics report using ydate_profiling
profile = ProfileReport(credit_card_data, title="Profiling Report")
profile.to_file("Credit_Card_Data_Report.html")


report = sv.analyze(credit_card_data)
report.show_html('sweetviz_report.html')