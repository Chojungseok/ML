# Linear_Regression(선형회귀)
- 데이터들간의 선형적 관계를 찾고 최적의 선형관계를 갖는 그래프를 통해서 예측을 진행

```python
'''
y = ax + b
a = 가중치(기울기)  
    a의 값이 클 수록 x의 값이 커질수록 큰 값을 반환 하기 때문에 가중치라고도 한다.  
    .coef_를 통해 훈련된 모델이 어떤 가중치(기울기)를 가졌는지 알 수 있다
b = y절편  
    .intercept_를 통해 훈련된 모델이 어떤 y절편을 가졌는지 알 수 있다.
''' 
from sklearn.linear_model import LinearRegression # 선형회귀 모델 호출
import matplotlib.pyplot as plt

lr = LinearRegression() # 인스턴스화
lr.fit(train_input, train_traget) # 모델 훈련
lr.score(test_input, test_target) # 테스트 데이터에 대한 점수 출력
lr.coef_ # lr모델의 가중치 확인
lr.intercept_ # lr모델의 y절편 확인
plt.scatter(train_input['Length2'], train_traget['Weight']) # Length2와 Weight의 관계 scatter plot
plt.plot([train_input['Length2'].min(), train_input['Length2'].max()], [train_input['Length2'].min() * lr.coef_[0][0] + (lr.intercept_[0]), train_input['Length2'].max() * lr.coef_[0][0] + (lr.intercept_[0])], x_label = 'Length2', y_label = 'Weight') # lr모델에서 사용한 가중치와 y절편을 활용하여 모델이 그린 선형 모델 시각화
```
![lr_model 시각화](/asset/output.png)


## 다항회귀
- 하나의 독립변수와 종속변수가 고차원의 다항식으로 표현이 된다  
$$y = a_{1}*x^n + a_{2}*x^{n-1} + \dots + b$$

```python
# train/test의 Length2열에 제곱을 하여 Length2_poly라는 새로운 열 생성
train_input['Length2_poly'] = train_input['Length2'] ** 2
test_input['Length2_poly'] = test_input['Length2'] ** 2
# 각각의 Length2_poly라는 열에 Length2의 제곱한 값이 생성됌
lr.fit(train_input, train_traget) # 2차항과 1차항을 이용하여 모델 훈련

print(lr.coef_, lr.intercept_) # 각각의 가중치와 y절편 출력

# 결과
[[-26.32982782   1.10544842]] [176.18290301] # 1차항의 가중치 / 2차항의 가중치 / y절편
```

## 다중회귀
- 하나의 종속변수와 여러개의 1차항인 독립변수들
$$y = a_{1} * x_{1} + a_{2} * x_{2} + \dots + b$$


### transformer(변환기)
- feature engineering을 할때 진행하고자 하는 feature을 넣어주면 된다.
```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(include_bias=False) # 1을 포함 하지 않는다
# 포함 할경우 상수항에 1을 곱한다고 인식하여 두 array를 곱한다고 인식

poly.fit([[3,5]]) # fit은 특정 기능을 하는것은 없지만 transform을 하기 위해서는 fit을 해야지만 transform을 할 수 있다. (학습을 하는게 아님)
poly.transform([[3,5]])

poly = PolynomialFeatures(include_bias=False) 

poly.fit(train_input)
train_poly = poly.transform(train_input)

poly.get_feature_names_out()

#결과
array(['Length2', 'Height', 'Width', 'Length2^2', 'Length2 Height',
       'Length2 Width', 'Height^2', 'Height Width', 'Width^2'],
      dtype=object)
```


### Scaling(스케일링)
- 서로 다른 변수의 범위를 일정한 범위로 맞추어 주는것

### 규제
1. 라쏘(L1)  
    계수의 절댓값을 기준으로 규제 적용
    값을 아예 **0**으로 만들어버릴수도 있다.
2. 릿지(L2)  
    계수를 제곱한 값을 기준으로 규제 적용  
    - alpha(L2 norm): 규제를 얼마나 강하게 줄 것인가 / default = 1

```python
# 라쏘
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(train_scaled, train_target)

print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))
```
```python
# 릿지
from sklearn.linear_model import Ridge

ridge = Ridge(alpha = 0.1)
ridge.fit(train_scaled, train_target)

print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))
```
alpha값이 클수록 규제가 커진다. -> 최적의 alpha 값을 찾는 것이 관건  
**alpha값에 대한 $R^2$값의 그래프를 그려보면 알 수 있다**  
train과 test의 $R^2$값이 가장 가까운 부분이 최적의 값이다.
```python
import matplotlib.pyplot as plt
import numpy as np

train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for i in alpha_list:
    ridge = Ridge(alpha=i)
    ridge.fit(train_scaled,train_target)
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))

plt.plot(np.log10(alpha_list), train_score, color = 'red')
plt.plot(np.log10(alpha_list), test_score, color = 'green')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend(('train', 'test'))
plt.show()
```
![alpha_graph](/asset/alpha_graph.png)
왼쪽 접점보다 왼쪽에 최적의 alpha값이 존재 할것이다.