#Decision Tree: 조건문 자동 생성 AI

from sklearn.tree import DecisionTreeClassifier
import pandas as pd 

#데이터
data = {
    "basement" : [1, 0, 1, 0, 1, 0], 
    "distance" : [1, 0, 1, 0, 1, 0], 
    "rain" : [0, 1, 0, 1, 1, 0], 
    "danger" : [1, 0, 1, 0, 1, 1]
}

#데이터 프레임 생성 
df = pd.DataFrame(data)

#input, output 설정
x = df[["basement", "distance", "rain"]]
y = df["danger"]

#모델 생성 및 학습 
model = DecisionTreeClassifier()
model.fit(x, y)

print(model.predict([[0,1,1]]))