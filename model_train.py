import pandas as pd
import numpy as np

data = pd.read_csv('breast_cancer_data.csv')

X = data.iloc[:,0:-1]
y = data.iloc[:,-1]

print(X.head())
print(y.isna().sum())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_X = scaler.fit_transform(X_train)

lr = LogisticRegression()
model = lr.fit(scaled_X, y_train)

scaled_test = scaler.transform(X_test)
predicted = pd.DataFrame(model.predict(scaled_test))

# comparing actual response values (y_test) with predicted response values (y_pred) 
from sklearn import metrics 
print("Logistic Regression model accuracy:", metrics.accuracy_score(y_test, predicted)) 

from pickle import dump
dump(model, open('breast_cancer_model.pkl', 'wb'))
dump(scaler, open('scaler.pkl', 'wb'))