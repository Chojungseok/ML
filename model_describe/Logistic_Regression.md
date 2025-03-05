# Logistic_Regression(로지스틱 회귀)  
**회귀이지만 분류 모델이다**  

공식
$$ z = a * x_{0} + b * x_{1} + \dots + f $$
위의 식이 로지스틱 회귀모델의 식으로 $z$값을 통하여 어느 분류에 속할지 **확률**을 구하는 것이다.  
***단 $z$값은 엄청 큰/작은 숫자가 나올것이다.*** -> 따라서 $z$값을 확률로 변환해주는 과정이 필요하다.
- 시그모이드 함수(0~1사이의 값을 갖는다)  
    양의 무한대: 1에 수렴  
    음의 무한대: 0에 수렴

```python
# 로지스틱 회귀 호출
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
# train_bs, target_bs: fish data에서 스케일링 진행 후 'Bream'과 'Smelt'만 추출 
lr.fit(train_bs, target_bs) # 모델 훈련
lr.predict(train_bs[:5]) # 모델 예측 후 앞의 5개 결과만 출력

lr.predict_proba(train_bs[:5]) # 예측 결과 앞의 5개에 대하여 'Bream'과 'Smelt' 두개에 각각 속할 확률(0~1) 출력
```
- **.predict_proba()**: $z$값을 확률로 변환하여 출력해준다, 이진분류와 다중분류 모두에서 사용 가능하다.
```python
lr.coef_ # logistic_Regression 모델의 각 변수에 대한 가중치
```
`array([[-0.47252702, -0.66229707, -0.74852216, -0.97283304, -0.72448087]])`

로지스틱 회귀분석의 hyperparameter  
- max_iter: 최대 반복 횟수  
- C: 정규화(L2)  

**Logistic_Regression에서는 다중회귀를 하지 않는다.**  
각각의$z$값 합이 1을 넘어가 버리기 때문이다.  
BUT! **soft_max**함수를 사용하게 되면 각각의 $z$값 합이 1이 되도록 확률로 변환을 해준다.