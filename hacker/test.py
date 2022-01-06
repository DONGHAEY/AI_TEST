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

X_train = Dtrain.drop(columns=['body fat_%']) #학습시킬 값들
Y_train = Dtrain['body fat_%'] #결과 값
X_train['gender'] = X_train['gender'].map({'M':1, 'F':0}) #성별 데이터 수치화
X_test = Dtest
X_test['gender'] = X_test['gender'].map({'M':1, 'F':0}) #테스트할 값들과 동시에 성별데이터 수치화

# X_train['gender'] = X_train['gender'].astype('category')
# X_test['gender'] = X_test['gender'].astype('category')

print(X_train, X_test)

model_p = {}

for key in Machine_model_dic.keys() :
    Machine_model_dic[key].fit(X_train, Y_train)
    model_p[key] = Machine_model_dic[key].predict(X_test)

model_pdf = pd.DataFrame(model_p)

p = pd.DataFrame(model_pdf['predict'])

p.to_csv("C:/Users\오동현/Desktop/AI_TEST/hacker/opportunity.csv", index=False)