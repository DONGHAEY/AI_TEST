#사이킷런의 iris 데이터셋 가져오기 
from sklearn.datasets import load_iris #iris의 데이터를 받기 위해 가져온다
from sklearn.model_selection import train_test_split #학습데이터와 테스트 데이터를 나누기 위해 사용된다
import pandas as pd #dataframe을 사용하기 위해

import matplotlib.pyplot as plt #그래프를 그려 결과값을 보기위해서 가져온다
import seaborn as sns

iris = load_iris() #iris 데이터를 불러온다
print(iris.keys()) #iris 데이터프레임의 속성값을 확인한다

"""
data : 피처의 데이터 세트 , numpy 배열
target : 분류 시 레이블 값, 회귀일 때는 숫자 결과값 데이터 Set
target_names : 개별 레이블의 이름
feature_names : 피처의 이름
DESCR : 데이터 세트에 대한 설명과 각 피처의 설명
"""

print(iris['DESCR'])

print(iris['data']) #이 특성들의 데이터의 

print(iris['target_names']) #0 setosa, 1 versicolor, 2 virginica #결과값의 영문이름

print(iris['target']) #결과는 이렇다, 결과값이다

print(iris['data'].shape) #data 데이터 세트

print(iris['feature_names']) #피처의 이름 #데이터들의 이름

df = pd.DataFrame(iris['data']) #df의 변수에 iris['data'] 특성 데이터 로 새로운 데이터프레임을 만든다

df.columns = iris['feature_names']

df['target'] = iris['target'] #데이터 프레임에 target 이름으로 결과값을 추가해준다

df['target'] = df['target'].astype('category') #'target'을 범주형 데이터로 변경한다

X_train, X_test, y_train, y_test =  train_test_split(df.iloc[:, :4], df['target'], random_state=0) #Xtrain은 학습데이터, y_train은 X_train의 결과값이고, X_test는 테스트할 값이다, y_test는 X_test의 결과값이다

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

Machine_model_dic = {
    "Nearest Neighbors" : KNeighborsClassifier(3),
    "Linear SVM" : SVC(kernel="linear", C=0.025),
    "RBF SVM" : SVC(gamma=2, C=1),
    "Gaussian Process" : GaussianProcessClassifier(1.0 * RBF(1.0)),
    "Decision Tree" : DecisionTreeClassifier(max_depth=5),
    "Random Forest" : RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    "Neural Net" : MLPClassifier(alpha=1, max_iter=1000),
    "AdaBoost" : AdaBoostClassifier(),
    "Naive Bayes" : GaussianNB(),
    "QDA" : QuadraticDiscriminantAnalysis()
}

model_p = {} #결과값을 저장할 dic이다

for key in Machine_model_dic.keys() :
   Machine_model_dic[key].fit(X_train, y_train) #학습시키는 LOGIC이다 fit() X_data의 데이터는 y_train의 결과값을 가진다고 학습시키는 코드
   model_p[key] = Machine_model_dic[key].predict(X_test) #한 모델의 데이터를 학습 시킨 후, 남겨두었던 x_test를 통해 예측 Logic을 실행한다 그리고 결과값을 model_p딕셔너리에 저장한다


#model_p에는 여러 모델의 결과값이 들어가있다

model_pdf = pd.DataFrame(model_p) #여러 모델로 예측했던 결과값을 새로운 데이터 프레임을 만들어 model_pdf의 저장한다
model_pdf['test'] = list(y_test) #그리고 실제의 값을 test로 추가하여 넣는다 # 이는 예측했던 결과값과 실제를 비교하기 위해서이다

#정확도 LOGIC이 아래에 쓰여져있습니다

def gh(col) :
  plt.scatter(x=model_pdf.index, y = model_pdf['test'])
  plt.scatter(x=model_pdf.index, y = model_pdf[col])
  plt.title(col)

  plt.show()


for col in model_pdf.columns :
  gh(col)


model_acc = {} 

for key in Machine_model_dic.keys() :
  model_acc[key] = Machine_model_dic[key].score(X_test, y_test) #sum(model_pdf['Nearest Neighbors'] == model_pdf['test']) / len(model_pdf)와 같은 값을 지닌다

print(model_acc) #정확도
"""
{'AdaBoost': 0.8947368421052632,
 'Decision Tree': 0.9736842105263158,
 'Gaussian Process': 0.9736842105263158,
 'Linear SVM': 0.9210526315789473,
 'Naive Bayes': 1.0,
 'Nearest Neighbors': 0.9736842105263158,
 'Neural Net': 0.9736842105263158,
 'QDA': 0.9736842105263158,
 'RBF SVM': 0.9736842105263158,
 'Random Forest': 0.9736842105263158}
"""

plt.plot(list(model_acc.keys()), list(model_acc.values()))
plt.xticks(rotation=90)
plt.show()

#result.png를 보면 정확도는 Naive Bayes가 가장 높다