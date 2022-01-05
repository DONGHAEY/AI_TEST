from sklearn.datasets import load_boston

boston = load_boston()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame(boston['data'])

df.columns = boston['feature_names']

df['target'] = boston['target']

print(df.loc[:, 'CRIM' : 'LSTAT'])


##################################################################################################################

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.loc[:, 'CRIM':'LSTAT'], df['target'], random_state=0)

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

model_p = {}

for key in Machine_model_dic.keys() :
    Machine_model_dic[key].fit(X_train, y_train)
    model_p[key] = Machine_model_dic[key].predict(X_test)

print(model_p)

model_pdf = pd.DataFrame(model_p)

model_pdf['y_test'] = list(y_test)

def gh(col) :
    plt.plot(model_pdf.index, model_pdf['y_test'], 'r-')
    plt.plot(model_pdf.index, model_pdf[col], 'b--')
    plt.title(col)
    plt.show()

for col in model_pdf.columns :
    gh(col)



