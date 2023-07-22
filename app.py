from dash import Dash, dcc, html
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.tools as tls
from dash.dependencies import Input, Output
from function import get_preprocessed_data
import plotly.graph_objs as go
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import precision_recall_curve
import numpy as np
import plotly.io as pio
import base64
import shap
import os

# Specify the file path
file_path = "Dataset.csv"

# Load the CSV file with tab delimiter and headers
data = pd.read_csv(file_path, delimiter="\t", header=0)

# Select the numeric columns
numeric_columns = ['Year_Birth', 'Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits',
                   'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
                   'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
                   'NumWebVisitsMonth']

# Create a dictionary of histograms, one for each numeric column
histograms = {}
for column in numeric_columns:
    histograms[column] = px.histogram(data, x=column, nbins=10)

# Group the data by 'Kidhome' and calculate the mean amount spent on each product
grouped_data = data.groupby('Kidhome')[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].mean()

# Reset the index to make 'Kidhome' a column instead of the index
grouped_data = grouped_data.reset_index()

# Melt the data to create a long-form dataframe suitable for plotting
melted_data = pd.melt(grouped_data, id_vars=['Kidhome'], value_vars=['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'], var_name='Product', value_name='Amount')

# Create a grouped bar chart using Plotly Express
fig2 = px.bar(melted_data, x='Kidhome', y='Amount', color='Product', barmode='group')

# Select the variables of interest and group by marital status
vars_of_interest = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']
grouped_data = data.groupby('Marital_Status')[vars_of_interest].mean().reset_index()

# Create a facet grid of barplots using Seaborn
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, var in enumerate(vars_of_interest):
    sns.barplot(x='Marital_Status', y=var, data=grouped_data, ax=axes[i//2][i%2])
    axes[i//2][i%2].set_title('Average ' + var + ' by Marital_Status')
    axes[i//2][i%2].set_xlabel('Marital_Status')
    axes[i//2][i%2].set_ylabel('Average ' + var)
plt.tight_layout()

# Convert the plot to a Plotly figure
fig3 = tls.mpl_to_plotly(fig)


def get_confusion_matrix(model, X_test, y_test):
    # Get the predicted labels from the model
    y_pred = model.predict(X_test)
    
    # Calculate the confusion matrix
    confusion_matrix_array = confusion_matrix(y_test, y_pred)
    # Assume y_test is a 1D array of true class labels
    classes = np.unique(y_test)

    # Create a Plotly graph object for the confusion matrix
    confusion_matrix_graph = {
        'data': [go.Heatmap(
            z=confusion_matrix_array,
            x=classes,
            y=classes,
            colorscale='YlGnBu'
        )],
        'layout': go.Layout(
            title='Confusion Matrix',
            xaxis=dict(title='Predicted Label'),
            yaxis=dict(title='True Label')
        )
    }
    

    return confusion_matrix_graph


def get_precision_recall_curve(model, X_test, y_test):
    # Get the predicted probabilities from the model
    y_pred_proba = model.predict_proba(X_test)
    # Assume y_test is a 1D array of true class labels
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba[:, 1])

    # Create a Plotly graph object for the precision-recall curve
    precision_recall_graph = {
        'data': [go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            line=dict(color='blue')
        )],
        'layout': go.Layout(
            title='Precision-Recall Curve',
            xaxis=dict(title='Recall'),
            yaxis=dict(title='Precision')
        )
    }

    return precision_recall_graph

#get the pre-processed data from main.py
data = get_preprocessed_data(file_path)

Y = data['AcceptedCmp1'] + data['AcceptedCmp2'] + data['AcceptedCmp3'] + data['AcceptedCmp4'] + data['AcceptedCmp5'] + data['Response']

#Split data
X_train, X_test, y_train, y_test = train_test_split(data, Y, test_size=0.3, random_state=42)

# Load the trained models
logreg = joblib.load('model1.pkl')
rfc = joblib.load('model2.pkl')
gbc = joblib.load('model3.pkl')
svm = joblib.load('model4.pkl')


# load the feature importances plot as a Yellowbrick visualizer object
visualizer = joblib.load('visualizer.pkl')

#load the latest gbc tuned model
gbc_tuned=joblib.load('model_gbc.pkl')

# Define the layout of the dashboard
app = Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Customer Data Dashboard'),

    html.Div(children='''
        Histogram of Numeric Columns
    '''),
    
    dcc.Dropdown(
        id='variable-dropdown',
        options=[{'label': col, 'value': col} for col in numeric_columns],
        value='Year_Birth'
    ),

     dcc.Graph(
        id='histogram-graph',
        figure=histograms['Year_Birth']
    ),

    html.Div(children='''
        Amount Spent on Products by Kidhome
    '''),

    dcc.Graph(
        id='grouped_bar_chart',
        figure=fig2
    ),

    html.Div(children='''
        Average Purchases by Marital Status
    '''),

    dcc.Graph(
        id='facet_grid',
        figure=fig3
    ),
    
    html.Div([
        html.Label('Select a model:'),
        dcc.Dropdown(
            id='model-dropdown',
            options=[{'label': i, 'value': i} for i in ['Logistic Regression', 'Random Forest', 'SVM', 'Gradient Boosting Classifier']],
            value='Logistic Regression'
        )
    ]),

    html.Div(id='confusion-matrix'),
    
      html.Div(children='''
        ROC Curve:
    '''),
     html.Img(
        id='ROC Plot',
        src='data:image/png;base64,{}'.format(base64.b64encode(open("roc.png", 'rb').read()).decode())
    ),
    
     
    html.Div(children='''
        Feature Importances Plot:
    '''),
    
    html.Img(
        id='feature-importances-img',
        src='data:image/png;base64,{}'.format(base64.b64encode(open("Feature_Importances_Plot.png", 'rb').read()).decode())
     ),
    
      html.Div(children='''
        SHAP Summary Plot:
    '''),
     html.Img(
        id='SHAP Summary Plot',
        src='data:image/png;base64,{}'.format(base64.b64encode(open("Shap.png", 'rb').read()).decode())
    ),
     
       html.Div(children='''
        Residual Plot
    '''),
     html.Img(
        id='Residual Plot',
        src='data:image/png;base64,{}'.format(base64.b64encode(open("Shap.png", 'rb').read()).decode())
    ),
     
     html.Div(children='''
        Precision-Recall Curve for Gradient Boosting Classifier:
    '''),

    dcc.Graph(
        id='precision-recall-graph'
    )
    
])


#Define the callback function to get the precision recall graph
@app.callback(
    Output('precision-recall-graph', 'figure')
)
def update_precision_recall_graph():
    # Get the gbc model object
    model = gbc_tuned

    # Generate the precision-recall curve for gbc
    precision_recall_graph = get_precision_recall_curve(model, X_test, y_test)

    # Return the precision-recall graph as a Plotly figure
    return precision_recall_graph


# Define a callback function to update the confusion matrix based on user input
@app.callback(
    Output('confusion-matrix', 'children'),
    [Input('model-dropdown', 'value')]
)
def update_confusion_matrix(model_name):
    # Get the selected model object
    if model_name == 'Random Forest':
        model = rfc
    elif model_name == 'Logistic Regression':
        model = logreg
    elif model_name == 'SVM':
        model = svm
    elif model_name == 'Gradient Boosting Classifier':
        model = gbc
    else:
        raise ValueError('Invalid model name')

    # Generate the confusion matrix for the selected model
    confusion_matrix_graph = get_confusion_matrix(model, X_test, y_test)

    return dcc.Graph(
        id='confusion-matrix-graph',
        figure=confusion_matrix_graph
    )


# Define a callback function to update the histogram figure based on user input
@app.callback(
    Output('histogram-graph', 'figure'),
    [Input('variable-dropdown', 'value')]
)
def update_histogram(variable):
    return histograms[variable]

#Run the app
if __name__ == '__main__':
    app.run_server(debug=True)