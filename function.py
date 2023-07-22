import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score,recall_score, f1_score, accuracy_score
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import BaggingClassifier
import shap
import lime
import lime.lime_tabular
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.ensemble import GradientBoostingRegressor
from yellowbrick.model_selection import FeatureImportances
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import PartialDependenceDisplay
import joblib




# Define a function that preprocesses the data and returns a DataFrame
def get_preprocessed_data(file_path):
    # Load the CSV file with tab delimiter and headers
    data = pd.read_csv(file_path, delimiter="\t", header=0)

    # Perform preprocessing on the data
    ############################# QUESTION 2 ################################
    data['Income'].fillna(data['Income'].mean(numeric_only=True).round(1),inplace=True)
    # select only the numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    df_numeric = data[numeric_cols]

    # calculate z-scores for all numeric columns in the dataframe
    z_scores = np.abs((df_numeric - df_numeric.mean()) / df_numeric.std())

    # select the top 3 columns with the highest z-scores
    top_cols = z_scores.max().nlargest(3).index.tolist()

    # remove outliers from the selected columns
    for col in top_cols:
        Q1 = df_numeric[col].quantile(0.25)
        Q3 = df_numeric[col].quantile(0.75)
        IQR = Q3 - Q1
        data = data[(data[col] >= Q1 - 1.5*IQR) & (data[col] <= Q3 + 1.5*IQR)]


    #######  Attribute: Education #######
    # create a dictionary to map education levels to ranks
    education_rank= {'Basic': 1, '2n Cycle': 2, 'Graduation': 3, 'Master': 4, 'PhD': 5}

    # replace education levels with their corresponding ranks
    data['Education'] = data['Education'].map(education_rank)



    #######  Attribute: Marital_Status #######
    # create one-hot encoded columns using pandas.get_dummies()
    one_hot_encoded = pd.get_dummies(data['Marital_Status'], prefix='Marital_Status')

    # convert boolean values to int values of 0 and 1
    one_hot_encoded = one_hot_encoded.astype(int)

    # join the one-hot encoded columns with the original dataframe
    data = pd.concat([data, one_hot_encoded], axis=1)


    # drop the original "Marital_Status" column
    data.drop('Marital_Status', axis=1, inplace=True)

    #######  Attribute: Income #######

    # normalize the "income" column
    data['Income'] = (data['Income'] - data['Income'].min()) / (data['Income'].max() - data['Income'].min())


    #### Mnt variables ####
    # select the columns to be normalized
    cols_to_normalize = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']

    # create a MinMaxScaler object
    scaler = MinMaxScaler()

    # fit and transform the selected columns using the scaler
    data[cols_to_normalize] = scaler.fit_transform(data[cols_to_normalize])


    ##### 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth' #####

    # select the columns to be normalized
    cols_to_normalize = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']

    # create a MinMaxScaler object
    scaler = MinMaxScaler()

    # fit and transform the selected columns using the scaler
    data[cols_to_normalize] = scaler.fit_transform(data[cols_to_normalize])


    ## Attribute: Z_CostContact and Z_revenue ##
    # drop the Z_CostContact and Z_Revenue columns from the dataframe
    data= data.drop(columns=['Z_CostContact', 'Z_Revenue'])


    ####### Attribute: ID ####### 
    data= data.drop(columns=['ID'])

    #### Attribute: Dt_Customer #####
    # convert the date column to a datetime object
    data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], format='%d-%m-%Y')

    # extract features from the date column
    data['Year'] = data['Dt_Customer'].dt.year
    data['Month'] = data['Dt_Customer'].dt.month
    data['Day'] = data['Dt_Customer'].dt.day
    data['DayOfWeek'] = data['Dt_Customer'].dt.dayofweek
    
    data= data.drop(columns=['AcceptedCmp1', 'AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','Response'])

    data = data.drop('Dt_Customer', axis=1)

    # Return the preprocessed data as a DataFrame
    return data
