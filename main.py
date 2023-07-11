import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Specify the file path
file_path = "Dataset.csv"

# Load the CSV file with tab delimiter and headers
data = pd.read_csv(file_path, delimiter="\t", header=0)

# Print the loaded DataFrame
#print(data.head(5))

#calculating the predictor Y
Y = data['AcceptedCmp1'] + data['AcceptedCmp2'] + data['AcceptedCmp3'] + data['AcceptedCmp4'] + data['AcceptedCmp5'] + data['Response']

#MISSING VLAUES
#code checking for missing values for every coloumn
# is any coloumn displays a non-zero value then we have missing values in the dataset
missing_values = data.isnull().sum()
#print("Missing Values:\n", missing_values)

#OUTLIERS
# code for showcasing outliers in the data
# Get the columns with numeric data

#test_coloumns=data[['Year_Birth','Income','MntMeatProducts']]
numeric_columns_var= data.select_dtypes(include='number')
# Iterate over each numeric column and create box plots
'''
for column in numeric_columns_var:
    plt.figure(figsize=(8, 6))
    plt.boxplot(data[column].dropna())
    plt.title(f"Outliers - {column}")
    plt.xticks([1], [column])
    plt.show()
 #uncomment later 
 '''   
    
#CLASS IMBALANCES
# Iterate over each column and create bar plots for class imbalance
'''
test_coloumns_1=data[['Kidhome','AcceptedCmp3','NumDealsPurchases']]
for column in test_coloumns_1:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=column, data=data)
    plt.title(f"Class Imbalance - {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.show()
'''    
    
#HANDLINE MISSING VALUES AND OUTLIERS 

# We only handle the income coloumn since it is the only one with missing values
data['Income'].fillna(data['Income'].mean(numeric_only=True).round(1),inplace=True)

# We only handle the outliers using quartile bonding [Don't know which coloumns to choose, think about it] # WHY DO WE USE QUARTILE BOUNDING ?
for column in numeric_columns_var:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)
    data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]



#PLOTS FOR VISUALISING THE DATASET

## 1ST PLOT 
# Select the numeric columns
numeric_columns = ['Year_Birth', 'Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits',
                   'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
                   'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
                   'NumWebVisitsMonth']

# Plot the histogram
data[numeric_columns].hist(figsize=(12, 10), bins=20)
plt.tight_layout()
plt.show() 

#2ND PLOT

# Select the categorical columns
categorical_columns = ['Education', 'Marital_Status', 'Complain', 'AcceptedCmp1', 'AcceptedCmp2',
                       'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']

# Plot the bar plot for each categorical column

for column in categorical_columns:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=column, data=data)
    plt.title(f"Bar Plot - {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.show()
  


#3RD PLOT   
# Select the columns for box plot
boxplot_columns = ['Income', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
                   'MntGoldProds']

# Plot the box plot
plt.figure(figsize=(8, 6))
sns.boxplot(data=data[boxplot_columns])
plt.title("Box Plot of Selected Columns")
plt.xlabel("Columns")
plt.xticks(rotation=90)
plt.show()



    




