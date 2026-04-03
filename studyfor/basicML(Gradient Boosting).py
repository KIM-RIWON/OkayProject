#Gradient Boosting: 틀린 걸 계속 고쳐가며 점점 똑똑해지는 모델 
#openai api
from openai import OpenAI
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd

data = {
    "basement": [1, 0, 1, 0, 1, 0],
    "distance": [1, 0, 1, 0, 1, 0],
    "rain": [1, 0, 1, 0, 1, 0],
    "danger": [1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)
client = OpenAI()
X = df[["basement", "distance", "rain"]]
y = df["danger"]

model = GradientBoostingClassifier()
model.fit(X, y)

#proba: 예측 결과에 대한 확률
#[0][1] -> 위험 확률, 현재 위험 확률에 대한 예측값만 필요하기 때문에
input_list = [1, 1, 1]
if input_list[0] == 1: 
    basement_input = "반지하 거주"
if input_list[2] == 1:
    rain_input = "비 많이 옴"
proba = model.predict_proba([input_list])[0][1]

prompt = f""" 
사용자의 위험도는 {proba:.2f}입니다. 
상황: 홍수 
조건: {basement_input}, {rain_input}

이 상황에서 사용자가 반드시 해야하는 행동을 짧고 명확하게 알려줘
"""

response = client.chat.completions.create(
    model = "gpt=4o-mini", 
    messages = [
        {"role":"system", "content":"당신은 재난 대응 전문가이다."},
        {"role": "user", "content" :prompt}
    ]
)
print("위험도", proba)
print("GPT 메시지")
print(response.choices[0].message.content)
