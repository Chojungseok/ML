# Ensemble
- 앙상블이란?  
함께 동시에라는 뜻의 프랑스어  
앙상블 학습은 대부분 결정 트리(decision tree)를 기반으로 만들어져있다.
    - 부트스트랩 샘플링 진행:  
    훈련 데이터에서 랜덤하게 샘플을 추출하여 훈련 데이터를 만든다. 다만 복원추출의 개념으로 생성이 된다.

```python
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier

train_input, test_input, train_target, test_target = train_test_split(data, target)

rf = RandomForestClassifier(n_jobs=-1)
scores = cross_validate(rf, train_input, train_target, return_train_score=True, n_jobs=-1)
```
RandomForest_Classifier를 train data를 활용하여 cross_validation 진행, `return_train_score=True`설정을 통해 train 점수도 validate 점수와 같이 출력
```python
rf.fit(train_input,train_target)
rf.feature_importances_
```
RandomForest모델을 학습 시킨뒤 `.feature_importances_`를 통해 특성 중요도 출력 -> 해당 모델에서 각각의 특성들이 분류를 위해 얼마나 중요한 지표로 사용했는지 알려준다.
```python
rf = RandomForestClassifier(oob_score=True, n_jobs=-1)
rf.fit(train_input, train_target)
rf.oob_score_
```
모델 인스턴스화를 진행 할 때에 `oob_score=True`로 설정후 모델 훈련할 경우  
-> randomforest는 decision tree의 앙상블 모델이기 때문에 부트스트랩방식의 데이터 선택, 따라서 단 한번도 선택되지 않은 데이터가 발생 할 수 있다. 이때 `oob_score`을 `True`로 설정하면 단 한번도 선택 되지 않은 데이터를 validation data로 설정하여 검증을 진행하고 점수를 출력할 수 있다.


# Extra_Tree
- RandomForest model과 굉장희 비슷하게 동작한다.
- 차이점: 훈련 데이터 선택시 ***부트스트랩 샘플링을 하지 않는다***  
-> 전체 훈련 세트를 사용한다, 대신 가장좋은 분할을 찾지 않고 무작위 분할을 한다.

# GradientBoostingClassifier
- 특징: 
    1. 깊이가 얕은 결정 트리 사용  
    2. 얕은 결정 트리를 사용하여 과대적합에 강하고 일반적으로 높은 일반화 성능을 기대 할 수 있다.

# HistGradientBoosting
- 특징:  
    1. 그룹화를 통하여 그룹을 하나의 데이터로 보게 된다.  
        -> 따라서 과대적합에 강하다

    그룹? 히스토그램의 각각의 계급을 그룹으로하여 하나의 데이터로 인식한다.
- permutation_importance(): 특성을 하나씩 섞어가며 모델의 성능 변화를 측정. 특성을 섞은 뒤 성능 변화가 없다면 중요하지 않은 특성이라고 판단, 변화가 크다면 중요한 특성이라고 판단 할 수가 있다.  
예시)  
```python
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingClassifier 

hg = HistGradientBoostingClassifier()
hg.fit(train_input, train_target)
scores = permutation_importance(hg, train_input, train_target)
print(scores.importances_mean)
```
결과: `[0.0862069  0.25188834 0.07655993]`