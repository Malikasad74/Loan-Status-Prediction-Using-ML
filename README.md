# Loan Status Prediction Using Machine Learning

# This project aims to predict the loan status of applicants using various machine learning algorithms. 
# The dataset contains information about applicants' demographics, education, employment, income, loan details, 
# and credit history. The objective is to build models that can accurately classify whether a loan will be approved or not 
# (represented by 'Y' or 'N' in the target variable 'Loan_Status'). The project involves data preprocessing, 
# handling missing values, encoding categorical variables, feature scaling, and applying multiple classification algorithms 
# such as Logistic Regression, SVM, Decision Tree, Random Forest, and Gradient Boosting. Additionally, hyperparameter 
# tuning is performed to optimize the models' performance. The best model is selected based on accuracy and cross-validation scores.

import pandas as pd
data = pd.read_csv('loan_prediction.csv')
# Loan_ID : Unique Loan ID

# Gender : Male/ Female

# Married : Applicant married (Y/N)

# Dependents : Number of dependents

# Education : Applicant Education (Graduate/ Under Graduate)

# Self_Employed : Self employed (Y/N)

# ApplicantIncome : Applicant income

# CoapplicantIncome : Coapplicant income

# LoanAmount : Loan amount in thousands of dollars

# Loan_Amount_Term : Term of loan in months

# Credit_History : Credit history meets guidelines yes or no

# Property_Area : Urban/ Semi Urban/ Rural

# Loan_Status : Loan approved (Y/N) this is the target variable

# 1. Display Top 5 Rows of The Dataset
data.head()

# 2. Check Last 5 Rows of The Dataset
data.tail()

# 3. Find Shape of Our Dataset (Number of Rows And Number of Columns)
data.shape
print("Number of Rows",data.shape[0])
print("Number of Columns",data.shape[1])

# 4. Get Information About Our Dataset Like Total Number Rows, Total Number of Columns, Datatypes of Each Column And Memory Requirement
data.info()

# 5. Check Null Values In The Dataset
data.isnull().sum()
data.isnull().sum()*100 / len(data)

# 6. Handling The missing Values
data = data.drop('Loan_ID',axis=1)
data.head(1)
columns = ['Gender','Dependents','LoanAmount','Loan_Amount_Term']
data = data.dropna(subset=columns)
data.isnull().sum()*100 / len(data)
data['Self_Employed'].mode()[0]
data['Self_Employed'] =data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])
data.isnull().sum()*100 / len(data)
data['Credit_History'].mode()[0]
data['Credit_History'] =data['Credit_History'].fillna(data['Credit_History'].mode()[0])
data.isnull().sum()*100 / len(data)

# 7. Handling Categorical Columns
data.sample(5)
data['Dependents'] =data['Dependents'].replace(to_replace="3+",value='4')
data['Gender'] = data['Gender'].map({'Male':1,'Female':0}).astype('int')
data['Married'] = data['Married'].map({'Yes':1,'No':0}).astype('int')
data['Education'] = data['Education'].map({'Graduate':1,'Not Graduate':0}).astype('int')
data['Self_Employed'] = data['Self_Employed'].map({'Yes':1,'No':0}).astype('int')
data['Property_Area'] = data['Property_Area'].map({'Rural':0,'Semiurban':2,'Urban':1}).astype('int')
data['Loan_Status'] = data['Loan_Status'].map({'Y':1,'N':0}).astype('int')
data.head()

# 8. Store Feature Matrix In X And Response (Target) In Vector y
X = data.drop('Loan_Status',axis=1)
y = data['Loan_Status']

# 9. Feature Scaling
cols = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']
from sklearn.preprocessing import StandardScaler
st = StandardScaler()
X[cols]=st.fit_transform(X[cols])
X

# 10. Splitting The Dataset Into The Training Set And Test Set & Applying K-Fold Cross Validation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np
model_df={}
def model_val(model,X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.20, random_state=42)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print(f"{model} accuracy is {accuracy_score(y_test,y_pred)}")
    score = cross_val_score(model,X,y,cv=5)
    print(f"{model} Avg cross val score is {np.mean(score)}")
    model_df[model]=round(np.mean(score)*100,2)
    
model_df

# 11. Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model_val(model,X,y)

# 12. SVC
from sklearn import svm
model = svm.SVC()
model_val(model,X,y)

# 13. Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model_val(model,X,y)

# 14. Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
model =RandomForestClassifier()
model_val(model,X,y)

# 15. Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
model =GradientBoostingClassifier()
model_val(model,X,y)

# 16. Hyperparameter Tuning
from sklearn.model_selection import RandomizedSearchCV

# Logistic Regression
log_reg_grid={"C":np.logspace(-4,4,20), "solver":['liblinear']}
rs_log_reg=RandomizedSearchCV(LogisticRegression(), param_distributions=log_reg_grid, n_iter=20,cv=5,verbose=True)
rs_log_reg.fit(X,y)
rs_log_reg.best_score_
rs_log_reg.best_params_

# SVC
svc_grid = {'C':[0.25,0.50,0.75,1],"kernel":["linear"]}
rs_svc=RandomizedSearchCV(svm.SVC(), param_distributions=svc_grid, cv=5, n_iter=20, verbose=True)
rs_svc.fit(X,y)
rs_svc.best_score_
rs_svc.best_params_

# Random Forest Classifier
rf_grid={'n_estimators':np.arange(10,1000,10), 'max_features':['auto','sqrt'], 'max_depth':[None,3,5,10,20,30], 'min_samples_split':[2,5,20,50,100], 'min_samples_leaf':[1,2,5,10]}
rs_rf=RandomizedSearchCV(RandomForestClassifier(), param_distributions=rf_grid, cv=5, n_iter=20, verbose=True)
rs_rf.fit(X,y)
rs_rf.best_score_
rs_rf.best_params_
```
