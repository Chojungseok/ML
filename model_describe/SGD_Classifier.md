# SGDClassifier(확률적 경사하강법)
- loss_function(손실함수)을 줄이는 것이 목표.  
    loss_function?  
    - 머신러닝 알고리즘이 얼마나 엉터리인가 측정하는 기준.  
    - 클수록 좋지 않다. -> 낮추는 것이 목표 

```python
from sklearn.linear_model import SGDClassifier

sc = SGDClassifier(loss='log_loss', max_iter=10) # loss = 'log_loss': 모델이 예측한 확률이 실제 레이블과 얼마나 잘 맞는지를 평가하고, 이를 기반으로 모델의 파라미터를 조정하는 역할
sc.fit(train_scaled,train_target)

sc.partial_fit(train_scaled, train_target) # partial_fit: 부분 학습(이전의 epoch를 누적하여 학습한다.)
```