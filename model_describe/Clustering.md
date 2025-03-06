# Clustering
- 비지도 학습중 하나
```python
# npy: numpy 데이터
import numpy as np
import matplotlib.pyplot as plt

# numpy 형태로 되어있는 과일 데이터
fruits = np.load('../ML/data/fruits_300.npy')

fruits.shape
```
`(300, 100, 100)`100 * 100 픽셀크기의 사진이 300장이 있다 라고 생각하면 된다.  
![IMG_639F646312FC-1.jpeg](/asset/IMG_639F646312FC-1.jpeg)

`fruits[0,0]` # 0번째 사진의 0번째 줄 을 보여주세요
```
array([  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
         1,   1,   1,   2,   1,   2,   2,   2,   2,   2,   2,   1,   1,
         1,   1,   1,   1,   1,   1,   2,   3,   2,   1,   2,   1,   1,
         1,   1,   2,   1,   3,   2,   1,   3,   1,   4,   1,   2,   5,
         5,   5,  19, 148, 192, 117,  28,   1,   1,   2,   1,   4,   1,
         1,   3,   1,   1,   1,   1,   1,   2,   2,   1,   1,   1,   1,
         1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
         1,   1,   1,   1,   1,   1,   1,   1,   1], dtype=uint8)
```
```python
plt.imshow(fruits[0], cmap='gray') # cmap = 'gray': 회색으로 보겠다
plt.show()
```
300장의 과일 사진중 첫번째 과일 사진
![apple_1](/asset/apple_1.png)
1~100번: 사과 / 101~200번: 파인애플 / 201~300번: 바나나  
해당 데이터는 3차원으로 되어있다 하지만 모델 학습을 위해서는 2차원 배열로 바꿔줄 필요가 있다.
```python
# 2차원 배열로 변경
apple = fruits[:100].reshape(-1, 100*100) # -1: 전체 , 100*100 한줄로 만들어준다
pineapple = fruits[100:200].reshape(-1, 100*100)
banana = fruits[200:].reshape(-1, 100*100)
```
각 과일들의 사진 한장씩 모두 100*100개의 column을 갖는 데이터로 바꾼뒤 동일한 과일 데이터를 row로 쌓아서 2차원 데이터 3개(apple, pineapple, banana)를 만들어 준다.
```python
apple_mean = np.mean(apple, axis=0).reshape(100, 100)
pineapple_mean = np.mean(pineapple, axis=0).reshape(100, 100)
banana_mean = np.mean(banana, axis=0).reshape(100, 100)
```
각 과일별로 픽셀 위치별 평균을 구하여 100*100의 2차원 데이터로 변경 -> 맨처음에 불러온 사진 1장과 같은 size
```python
# fruits에 일괄적으로 적용 fruits데이터의 값들과 apple의 평균값과의 차이를 이용
abs_diff = np.abs(fruits - apple_mean) # 사과 무리의 평균을 전체 데이터에서 뺀 값의 절댓값을 저장
# print(abs_diff)
abs_mean = np.mean(abs_diff, axis=(1,2)) # 사진 1장당의 평균을 구해 저장

apple_index = np.argsort(abs_mean)[:200] # 평균치가 오름차순이 되도록 정렬 / 앞에서부터 200장 슬라이싱

fig, axs = plt.subplots(20,10, figsize = (10,10))
for i in range(10):
    for j in range(10):
        axs[i,j].imshow(fruits[apple_index[i*10 + j]], cmap = 'gray')

plt.show()
```
위와 같은 코드를 실행시킨 결과 앞의 100장은 사과 뒤의 100장은 아무것도 들어있지 않았다. -> 해당 클러스터에는 사과만 분류가 되었다.
![apple_cluster](/asset/apple_cluster.png)
지금까지 진행한 내용이 K-means clustering의 마지막단계를 직접 코드로 구현해 본 것이다.

# K-means clustering
- 위의 작업을 자동으로 실행 해준다.

```python 
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3) # data안에 클러스터가 몇개 있는지 알 수 없기 때문에 점점 늘려가면서 테스트를 하면 된다 -> 클러스터를 늘려가면서 성능을 비교 해 보면 된다
km.fit(fruits_2d)
print(km.labels_)
```
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 2 2 1 1 1 1 1 1 1 2 2 1 1 1 1 1 1 1
 1 2 2 1 2 2 1 1 2 2 1 1 2 1 2 2 1 1 1 1 2 1 1 1 1 1 1 1 1 2 1 1 1 1 1 2 1
 2 2 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 1 1 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1]
```
np.unique(km.labels_, return_counts=True)
```
`(array([0, 1, 2], dtype=int32), array([200,  77,  23]))`
위의 모델은 0, 1, 2로 구분 하였고 0은 200개 1은 77개 2는 23개로 구분 하였다.

# PCA(주성분 분석/차원 축소)
- 여기에서의 차원이란?  
    **차원 = 특성** 이라고 생각하면 된다.

```python
from sklearn.decomposition import PCA

# n_components : 주성분의 갯수/비율 설정
pca = PCA(n_components= 50) # 주성분의 갯수 50개로 설정
pca.fit(fruits_2d)
```
`(50, 10000)`데이터가 50개만 선택이 되었다.
- .inverse_transform(): pca를 하기 전으로 원상 복구 할 수 있다. -> pca는 나머지 데이터를 버리는 것이 아니라 압축하는 것.
- .explained_variance_ratio_: 각 주성분들이 얼만큼의 역할을 하고 있는가를 설명해준다.