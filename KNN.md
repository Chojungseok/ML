# KNN

## KNN이란?
- KNN(K-nearest Neighbors / K최근접 이웃)
    - 입력받은 데이터를 **가장 가까운** 이웃이 어디인지 확인하여 더 많은 데이터가 속해있는 그룹으로 분류하는 알고리즘
    - 대표적인 **지도 학습(supervised learning)**중 하나이다

## KNN 사용 방법
1. Data 불러오기
2. Data EDA  
    2-1. Data의 결측치, row/columns 갯수, data_type 등 확인.
3. Data preprocessing  
    3-1. 결측치 처리  
    3-2. Data Scaling(필요시 진행)  
    3-3. Label/Feature data 분류
4. Data train/test split  
    4-1. Label/Feature data를 각각 train과 test로 일정 비율에 맞게 분할
5. KNN Model training  
    5-1. train data를 활용하여 모델 학습 진행(.fit())
6. Test  
    6-1. test data를 활용하여 모델 test 진행 
