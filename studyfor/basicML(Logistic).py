import pandas as pd 
from sklearn.linear_model import LogisticRegression 

#데이터 만들기 (반지하 여부, 재난과 거리, 강수량 많음, 위험 여부(정답))
data = {
    "basement" : [1, 0, 1, 0, 1, 0], 
    "distance" : [1, 0, 1, 0, 1, 0], 
    "rain" : [0, 1, 0, 1, 1, 0], 
    "danger" : [1, 0, 1, 0, 1, 1]
}
#pandas로 데이터를 표 형태로 만드는 코드 
df = pd.DataFrame(data)

#input, output 나누기 
x = df[["basement", "distance", "rain"]]
y = df["danger"]

#로지스틱 회귀 모델 설정 
model = LogisticRegression()

#지도학습 
model.fit(x, y)

#예측 
prediction = model.predict([[1, 0, 0]])

print(prediction)