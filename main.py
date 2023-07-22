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
from sklearn.calibration import CalibratedClassifierCV
from pycebox.ice import ice, ice_plot
import scikitplot as skplt
from sklearn.metrics import precision_recall_curve, average_precision_score
import joblib

# Specify the file path
file_path = "Dataset.csv"

# Load the CSV file with tab delimiter and headers
data = pd.read_csv(file_path, delimiter="\t", header=0)

# Print the loaded DataFrame




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

for column in numeric_columns_var:
    plt.figure(figsize=(8, 6))
    plt.boxplot(data[column].dropna())
    plt.title(f"Outliers - {column}")
    plt.xticks([1], [column])
    plt.show()
 #uncomment later 

    
#CLASS IMBALANCES
# Iterate over each column and create bar plots for class imbalance

test_coloumns_1=data[['Kidhome','AcceptedCmp3','NumDealsPurchases']]
for column in test_coloumns_1:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=column, data=data)
    plt.title(f"Class Imbalance - {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.show()
   
    
#HANDLINE MISSING VALUES AND OUTLIERS 


# We only handle the income coloumn since it is the only one with missing values
data['Income'].fillna(data['Income'].mean(numeric_only=True).round(1),inplace=True)

# We only handle the outliers using quartile bonding [Don't know which coloumns to choose, think about it] # WHY DO WE USE QUARTILE BOUNDING ?
# for column in numeric_columns_var:
#     Q1 = data[column].quantile(0.25)
#     Q3 = data[column].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - (1.5 * IQR)
#     upper_bound = Q3 + (1.5 * IQR)
#     data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
   
   


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
# Group the data by 'Kidhome' and calculate the mean amount spent on each product
grouped_data = data.groupby('Kidhome')[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].mean()

# Reset the index to make 'Education' a column instead of the index
grouped_data = grouped_data.reset_index()

# Melt the data to create a long-form dataframe suitable for plotting
melted_data = pd.melt(grouped_data, id_vars=['Kidhome'], value_vars=['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'], var_name='Product', value_name='Amount')

# Create a grouped bar chart using Seaborn
sns.barplot(x='Kidhome', y='Amount', hue='Product', data=melted_data)
plt.title('Amount Spent on Products by Education Level')
plt.xlabel('KidHome')
plt.ylabel('Amount Spent (in USD)')
plt.show()   
    


#3RD PLOT   
# Select the variables of interest and group by education level
vars_of_interest = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']
grouped_data = data.groupby('Marital_Status')[vars_of_interest].mean().reset_index()

# Create a barplot for each variable of interest
for var in vars_of_interest:
    sns.barplot(x='Marital_Status', y=var, data=grouped_data)
    plt.title('Average ' + var + ' by Marital_Status')
    plt.xlabel('Marital_Status')
    plt.ylabel('Average ' + var)
    plt.show()

    
  
############################# QUESTION 2 ################################



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

data = data.drop('Dt_Customer', axis=1)

Y = data['AcceptedCmp1'] + data['AcceptedCmp2'] + data['AcceptedCmp3'] + data['AcceptedCmp4'] + data['AcceptedCmp5'] + data['Response']

data= data.drop(columns=['AcceptedCmp1', 'AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','Response'])





############################# QUESTION 3 ################################

# split the data into training and testing sets

# print(Y.shape)
# print(data.shape)

X_train, X_test, y_train, y_test = train_test_split(data, Y, test_size=0.3, random_state=42)

# print the shapes of the training and testing sets
print(f'Training set shape: {X_train.shape}, {y_train.shape}')
print(f'Testing set shape: {X_test.shape}, {y_test.shape}')

### ML MODELS ###

#1. LOGISTIC REGRESSION

# create a new instance of the logistic regression model
logreg = LogisticRegression(max_iter=400,solver='sag')

# fit the logistic regression model to the training data
logreg.fit(X_train, y_train)

# generate predicted output values for the testing data
y_pred_log = logreg.predict(X_test)

# Save the trained model to disk
joblib.dump(logreg, 'model1.pkl')



# 2.  RANDOM FORESTS

# create a Random Forest classifier with 400 trees
rfc = RandomForestClassifier(n_estimators=400, random_state=42)

# train the classifier on the training data
rfc.fit(X_train, y_train)

# make predictions on the testing data
y_pred_rf = rfc.predict(X_test)

# Save the trained model to disk
joblib.dump(rfc, 'model2.pkl')


#3. Gradient boosting

# create a Gradient Boosting classifier with 100 trees
gbc = GradientBoostingClassifier(n_estimators=400, random_state=42)

# train the classifier on the training data
gbc.fit(X_train, y_train)

# make predictions on the testing data
y_pred_gb = gbc.predict(X_test)

# Save the trained model to disk
joblib.dump(gbc, 'model3.pkl')

#4. SVM 

# create an SVM classifier with a radial basis function kernel
svm = SVC(kernel='rbf', random_state=42)


# train the classifier on the training data
svm.fit(X_train, y_train)

# make predictions on the testing data
y_pred_SVM = svm.predict(X_test)

# Save the trained model to disk
joblib.dump(svm, 'model4.pkl')




############################# QUESTION 4 ################################

# Create a list of y_pred variables for each model
y_pred_list = [y_pred_log, y_pred_rf, y_pred_gb, y_pred_SVM]
models=[logreg,rfc,gbc,svm]
model_names=['Logistic Regression','Random Forests','Gradient Boosting','SVM',]
# define positive classes
positive_classes = [1, 2, 3, 4, 5, 6]
i=0

for i in range(len(y_pred_list)):
    print("Model:"+ model_names[i]+"\n")
    # Accuracy score
    print("Accuracy :", accuracy_score(y_test, y_pred_list[i])) 

    # Precision score
    print("Precision:", precision_score(y_test,y_pred_list[i],average="weighted",zero_division=1))

    #Recall score
    print('Recall:', recall_score(y_test,y_pred_list[i],average="weighted"))

    #F1 Score
    print('F1 score:', f1_score(y_test, y_pred_list[i],average="weighted"))
    
    # AUC score
    if hasattr(models[i], 'predict_proba'):
        proba = models[i].predict_proba(X_test)
    else:
        svm = models[i]
        svm_calibrated = CalibratedClassifierCV(svm, cv='prefit')
        svm_calibrated.fit(X_test, y_test)
        proba = svm_calibrated.predict_proba(X_test)
    auc_score = roc_auc_score(y_test, proba, multi_class='ovr', average='weighted')
    print("AUC score:", auc_score)
    
    
    
    # get all unique classes
    classes = np.unique(np.concatenate((y_test, y_pred_list[i])))

    # generate confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred_list[i], labels=classes)
    
    # create a new figure and axis object
    fig, ax = plt.subplots()

    # display confusion matrix
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=classes)
    cm_display.plot(include_values=True, cmap='Blues', ax=ax, xticks_rotation='horizontal')
    plt.show()    
    
    #update i
    i=i+1
    print("\n")
    

#ROC CURVE 
# initialize plot
plt.figure()

# plot ROC curve for each model
for i in range(len(models)):
    # generate predicted probabilities or decision function scores depending on the model
    if i < 3:
        y_prob = models[i].predict_proba(X_test)
    else:
        y_score = models[i].decision_function(X_test)
        y_prob = 1 / (1 + np.exp(-y_score))

    # combine positive classes into one class
    y_true = np.zeros_like(y_test)
    positive_classes = [3, 5]
    for cls in positive_classes:
        y_true += (y_test == cls)
    y_true = (y_true > 0).astype(int)
    y_score = y_prob[:, positive_classes].max(axis=1)

    # compute micro-averaged ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    # plot ROC curve
    plt.plot(fpr, tpr, label='%s (AUC = %0.2f)' % (model_names[i], roc_auc))

# finalize plot
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc='lower right')
plt.show()





# # ############################# QUESTION 5 ################################

#Q5(a)
# create a Gradient Boosting classifier 
gbc_hyper = GradientBoostingClassifier(n_estimators=50,
                                 learning_rate=0.1,
                                 max_depth=3,
                                 subsample=1.0,
                                 min_samples_split=2,
                                 min_samples_leaf=1,
                                 random_state=42)

# train the classifier on the training data
gbc_hyper.fit(X_train, y_train)

# make predictions on the testing data
y_pred_gb = gbc_hyper.predict(X_test)

# Accuracy score
print("Accuracy  after tuning the hyperparameters:", accuracy_score(y_test, y_pred_gb))


# #Q5(b)


# define the hyperparameter space
param_dist = {
    'n_estimators': sp_randint(200, 300),
    'learning_rate': sp_uniform(0.001, 0.01),
    'max_depth': sp_randint(3, 4),
    'subsample': sp_uniform(0.5, 0.75),
    'min_samples_split': sp_randint(2, 3),
    'min_samples_leaf': sp_randint(1, 2)
}

# create a Gradient Boosting classifier
gbc = GradientBoostingClassifier(random_state=42)

# create a RandomizedSearchCV object to search over the parameter space
random_search = RandomizedSearchCV(
    estimator=gbc,
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    random_state=42,
    n_jobs=-1,
    scoring='accuracy')

# fit the RandomizedSearchCV object to the training data
random_search.fit(X_train, y_train)

# print the best hyperparameters and best score
print("Best hyperparameters:", random_search.best_params_)
print("Best score:", random_search.best_score_)

# asssigning the best parameters to the gbc model
gbc = GradientBoostingClassifier(random_state=42).set_params(**random_search.best_params_)

 ## dump gbc 
joblib.dump(gbc, 'model_gbc.pkl')
 
# Q5(c)

# Step 3. Add ensembling methods on top of Base model
ensemble_model = BaggingClassifier(estimator=GradientBoostingClassifier(), n_estimators=5, random_state=42)
## it can't process this on my PC and change the n_estimators with more parameters

# Step 4. Fit the ensemble model to the training data
ensemble_model.fit(X_train, y_train)

# Step 5. Evaluate the performance of the ensemble model on the test data
accuracy = ensemble_model.score(X_test, y_test)
print("Accuracy for Bagging:", accuracy)



############################# QUESTION 6 ################################


# create a GradientBoostingRegressor object with the same hyperparameters as the GradientBoostingClassifier
gbr = GradientBoostingRegressor(
    loss='squared_error',
    learning_rate=gbc.learning_rate,
    n_estimators=gbc.n_estimators,
    subsample=gbc.subsample,
    criterion=gbc.criterion,
    min_samples_split=gbc.min_samples_split,
    min_samples_leaf=gbc.min_samples_leaf,
    max_depth=gbc.max_depth,
    random_state=42
)

# fit the GradientBoostingRegressor to the training data
gbr.fit(X_train, y_train)

joblib.dump(gbr,'gbr.pkl')



# create a TreeExplainer for the GradientBoostingRegressor
shap_explainer = shap.TreeExplainer(gbr)

# generate SHAP values for the GradientBoostingClassifier
shap_values = shap_explainer.shap_values(X_test)

# create a SHAP summary plot
shap.summary_plot(shap_values, X_test,plot_type='dot')


# save the plot as an image
#plt.savefig('shap_summary_plot.png')


#2. FEATURE IMPORTANCE PLOT
# Creating the feature importances plot
visualizer = FeatureImportances(gbc,relative=True)

visualizer.fit(X_train, y_train)

joblib.dump(visualizer,'visualizer.pkl')
# Saving plot in PNG format
visualizer.show(outpath="Feature_Importances_Plot.png")


gbc.fit(X_train,y_train)
y_gbc_proba = gbc.predict_proba(X_test)
y_gbc_pred = np.where(y_gbc_proba[:,1] > 0.5, 1, 0)
skplt.metrics.plot_precision_recall(y_test, y_gbc_proba, title = 'PR Curve for GBC',figsize=(12,8))
plt.legend(loc='lower right')
plt.show()



































    




