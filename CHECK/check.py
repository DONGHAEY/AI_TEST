import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

Machine_model_dic = {
    "Linear" : LinearRegression(),
    "SVM" : svm.SVR(),
    "RandomForest" : RandomForestRegressor(),
    "xgboost" : XGBRegressor()
}

train = pd.read_csv('C:/Users\오동현/Desktop/AI_TEST/hacker/train.csv')

Dtrain = pd.DataFrame(train)

X_train = Dtrain.drop(columns=['body fat_%']) #학습시킬 값들
Y_train = Dtrain['body fat_%'] #결과 값
X_train['gender'] = X_train['gender'].map({'M':1, 'F':0}) #성별 데이터 수치화



from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

model_p = {}

for key in Machine_model_dic.keys() :
    Machine_model_dic[key].fit(x_train, y_train)
    model_p[key] = Machine_model_dic[key].predict(x_valid)

model_pdf = pd.DataFrame(model_p)

model_pdf['check_this'] = list(y_valid)

print(model_pdf)

import pandas as pd
import matplotlib.pyplot as plt

model_acc = {} 

for key in Machine_model_dic.keys() :
  model_acc[key] = Machine_model_dic[key].score(x_valid, y_valid) #sum(model_pdf['Nearest Neighbors'] == model_pdf['test']) / len(model_pdf)와 같은 값을 지닌다

print(model_acc) #정확도

plt.plot(list(model_acc.keys()), list(model_acc.values()))
plt.xticks(rotation=90)
plt.show()