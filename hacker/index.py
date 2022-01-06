import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

Machine_model_dic = {
    "predict" : LinearRegression()
}

train = pd.read_csv('C:/Users\오동현/Desktop/AI_TEST/hacker/train.csv')
test = pd.read_csv('C:/Users\오동현/Desktop/AI_TEST/hacker/test.csv')

Dtrain = pd.DataFrame(train)
Dtest = pd.DataFrame(test)

X_train = Dtrain.drop(columns=['body fat_%', 'gender'])
Y_train = Dtrain['body fat_%']

X_test = Dtest.drop(columns=['gender'])

model_p = {}

for key in Machine_model_dic.keys() :
    Machine_model_dic[key].fit(X_train, Y_train)
    model_p[key] = Machine_model_dic[key].predict(X_test)

model_pdf = pd.DataFrame(model_p)

p = pd.DataFrame(model_pdf['predict'])

p.to_csv("C:/Users\오동현/Desktop/AI_TEST/hacker/submit_sample.csv", index=False)