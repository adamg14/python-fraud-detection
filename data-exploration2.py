from scipy.stats import pointbiserialr
from dotenv import load_dotenv
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import ttest_ind

load_dotenv()

csv_file_path = os.getenv("CSV_FILE_PATH")
credit_card_data = pd.read_csv(csv_file_path)

# correlation analysis 2
correlations = {f"V{i}": pointbiserialr(credit_card_data[f"V{i}"], credit_card_data["Class"])[0] for i in range(1, 29)}

# strongest correlation
# the strongest correlation between all the intermediate variables and the class (valid or fraudulent) is variable is with the vaue
# ultimately, this means there is no true correlation between any of the individual intermediate variables and the class of the transasction none of the 
strongest_positive_correlation = max(correlations, key=correlations.get)
strongest_negative_correlation = min(correlations, key=correlations.get)

print("Strongest positive correlation variable: " + strongest_positive_correlation)
print("Strongest negative correlation value: " + strongest_negative_correlation)
post_corr = correlations[strongest_positive_correlation]
neg_corr = correlations[strongest_negative_correlation]

print("Strongest postivie correlation value: " + str(post_corr))
print("Strongest negative correlation value: "  + str(neg_corr))

print(credit_card_data.head())

# group statistics
# the average of each variable grouped by class
# There is disparity between the average values of the amount of transactions betwen classes which was observed in the first data exploration 
group_statistics = credit_card_data.groupby("Class").mean()
print(group_statistics)

# using machine learning to determine variable importance
# training a classifier using the data within the dataframe (logistic regression) to determine the importance of each individual feature variable
x = credit_card_data[[f"V{i}" for i in range(1, 29)]]
y = credit_card_data["Class"]

rf = RandomForestClassifier()
# fit the model with the numeric feature variables with the corresponding to the categorical class variables
rf.fit(x, y)

numeric_variable_importance = rf.feature_importances_
variable_importance = pd.DataFrame({
    "Variable": x.columns,
    "Importance": numeric_variable_importance
}).sort_values(by="Importance", ascending=False)

# variable importance visualisation
sns.barplot(data=numeric_variable_importance, x="Variable", y="Importance")
plt.title("Feature Importance")
plt.show()
plt.close()
print(variable_importance)


# hypothesis testing for each of the numeric feature classes to identify differences between the variables in the different categories
for col in [f"V{i}" for i in range(1, 29)]:
    valid = credit_card_data[credit_card_data["Class"] == 0][col]
    fraudulent = credit_card_data[credit_card_data["Class"] == 1][col]
    t_stat, p_value = ttest_ind(valid, fraudulent)
    print(f"V{col}")
    print(f"T-stat: {t_stat}")
    print(f"P-value: {p_value}")
    
# additional visualiasation
variable_colums = credit_card_data.loc[:, 'V1': 'V28']
# converting the dataframe into a pivot table for easier visualisation
numeric_variable_columns_melt = variable_colums.melt(var_name="Variable", value_name="Value")
sns.violinplot(x="Variable",
               y="Value",
               data=numeric_variable_columns_melt,
               )
plt.xticks(rotation=90)
plt.title("Numeric intermediate column.")
plt.show()
