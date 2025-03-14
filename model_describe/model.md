# 개념의 포함 관계

# 머신러닝 모델 정리

## 1. 지도 학습 (Supervised Learning)
- 지도 학습은 입력 데이터와 그에 해당하는 레이블(결과)을 이용해 모델을 학습하는 방법입니다.

### ├ (1) 선형 모델 (Linear Models)
- 선형 모델은 입력 변수에 대한 선형 관계를 모델링합니다.
    - **Logistic Regression**: 분류 문제에서 선형 모델을 활용하는 대표적인 방법
    - **SGDClassifier**: 여러 선형 모델(로지스틱 회귀 포함)을 **확률적 경사 하강법(SGD)**을 이용해 학습하는 일반적인 프레임워크

### ├ (2) 비선형 모델 (Non-Linear Models)
- 비선형 모델은 입력 변수들 간의 비선형 관계를 모델링합니다.
    - **DecisionTreeClassifier**: 트리 기반 모델로, 선형 모델과는 완전히 다른 방식으로 데이터를 분류
    - **기타 신경망, SVM 등**: 비선형 모델에 해당하는 다른 알고리즘들

---

## 머신러닝 모델 정리

### 회귀 (Regression)
| 유형 | 모델 |
|------|------|
| 선형 회귀 | 선형 회귀 (Linear Regression), 다중 회귀 (Multiple Regression), 릿지 회귀 (Ridge Regression), 라쏘 회귀 (Lasso Regression), 다항 회귀 (Polynomial Regression) |
| 트리 기반 회귀 | 의사결정나무 회귀 (Decision Tree Regressor), 랜덤 포레스트 회귀 (Random Forest Regressor), 엑스트라 트리 회귀 (Extra Trees Regressor), 그래디언트 부스팅 회귀 (Gradient Boosting Regressor), HistGradientBoosting Regressor |

### 분류 (Classification)
| 유형 | 모델 |
|------|------|
| 선형 분류 | 로지스틱 회귀 (Logistic Regression) |
| 트리 기반 분류 | 의사결정나무 분류 (Decision Tree Classifier), 랜덤 포레스트 분류 (Random Forest Classifier), 엑스트라 트리 분류 (Extra Trees Classifier), 그래디언트 부스팅 분류 (Gradient Boosting Classifier), HistGradientBoosting Classifier |
| 기타 | 서포트 벡터 머신 (SVM), K-최근접 이웃 (K-NN), 신경망 (Neural Networks) |

## 2. 비지도 학습 (Unsupervised Learning)

### 군집 (Clustering)
| 모델 |
|------|
| K-평균 (K-Means), 계층적 군집 (Hierarchical Clustering), DBSCAN (Density-Based Spatial Clustering of Applications with Noise) |

### 차원 축소 (Dimensionality Reduction)
| 모델 |
|------|
| 주성분 분석 (PCA), t-SNE, UMAP |

### 연관 규칙 학습 (Association Rule Learning)
| 모델 |
|------|
| 아프리오리 (Apriori), Eclat |

---

## 개념의 포함 관계
- **Logistic Regression**은 분류 문제에서 선형 모델을 활용하는 대표적인 방법이고,  
  **SGDClassifier**는 여러 선형 모델(로지스틱 회귀 포함)을 **확률적 경사 하강법(SGD)**을 이용해 학습하는 일반적인 프레임워크입니다.
- **DecisionTreeClassifier**는 트리 기반 모델로, 선형 모델과는 완전히 다른 방식으로 데이터를 분류합니다.
- 따라서 **지도 학습 > 선형 모델 > Logistic Regression, SGDClassifier**의 포함 관계가 성립하며,  
  **DecisionTreeClassifier**는 다른 계열이지만 **지도 학습의 일부**입니다.


---

# 경사 하강법(Gradient Descent) - 모델 vs 방법

## 경사 하강법이란?
경사 하강법(Gradient Descent)은 **모델이 아니라 최적화 방법(Optimization Algorithm)**입니다.  
모델이 학습할 때 **손실 함수(loss function)**를 최소화하려고 하는데, 이를 위해 기울기(gradient)를 따라 조금씩 이동하면서 최적의 가중치를 찾아가는 방법이 **경사 하강법(Gradient Descent)**입니다.

---

## 모델 vs 방법

### ✅ 모델 (Model)
- 데이터를 학습하여 패턴을 찾아내는 구조
- 예: **Logistic Regression**, **DecisionTreeClassifier**, **Neural Networks** 등

### ✅ 방법 (Method, Algorithm)
- 모델이 학습할 때 사용하는 최적화 기법
- 예: **Gradient Descent (경사 하강법)**, **Adam**, **Newton's Method** 등

---

### 결론
오늘 배운 것 중에서는 **지도 학습**이 가장 넓은 개념이고, 그다음으로는 **선형 모델**이 더 일반적인 개념이라고 볼 수 있습니다.
- **경사 하강법(Gradient Descent)**은 Logistic Regression 같은 모델을 학습시키는 데 사용할 수 있는 최적화 기법 중 하나입니다.
- **SGDClassifier**는 경사 하강법의 일종인 **확률적 경사 하강법(SGD, Stochastic Gradient Descent)**을 사용해 모델을 학습하는 방식입니다. 🚀
