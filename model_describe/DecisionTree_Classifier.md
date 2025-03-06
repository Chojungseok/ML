# DecisionTree_Classifier
- 의사결정 나무
- 나무를 거꾸로 뒤집어 놓은 듯한 모습
![decision_tree](ML/asset/decisiontree.png)
```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=20) # max_depth : 트리의 최대 깊이 설정
dt.fit(train_scaled, train_target)
```
**불순도**: 데이터가 얼마나 편향되어있는가.  
    - 값이 낮을 수록 순수하다. -> 대부분의 데이터가 동일 클래스에 속한다.  
    - $gini불순도 = 1 - (음성 클래스 비율^2 + 양성 클래스 비율^2)$
    - gini불순도가 0.5라면 최악의 상황 / 한쪽으로 편향된 것이 최고의 상황이다.  

- 정보 이득: 부모와 자식 노드 사이의 불순도 차이.

```python
from sklearn.model_selection import cross_validate
scores = cross_validate(dt, train_input, train_target)
```
**cross_validation**: 전체 데이터셋을 cv의 값(default = 5)으로 나누어서 데이터를 분할 한뒤 분할한 각각의 데이터중 1개를 valid 나머지를 train에 사용하여 모델 검증, 모든 분할들이 valid를 한번씩 진행한다.

```python
# 하이퍼파라미터 튜닝
## GridS_earch
from sklearn.model_selection import GridSearchCV

import numpy as np
params = {
    'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001),
    'max_depth': range(5, 20, 1)
}
GridSearchCV(dt, params, n_jobs=-1) # n_jobs: 내 컴퓨터에서 활용할 수 있는 코어의 수 지정(-1은 전부 사용)
```
**GridSearchCV**: 최적의 하이퍼파라미터를 찾기 위함
- 모든 모델은 하이퍼 파라미터를 설정 할 수 있다. 다만 값을 지정해 줘야 하는 경우 여러 조합이 나올 수 있기 때문에 내가 값을 설정하면 해당 값들의 조합들 중 최적의 값을 찾아준다. 

`.best_estimator`: 모델의 최적의 하이퍼파라미터를 출력해준다  