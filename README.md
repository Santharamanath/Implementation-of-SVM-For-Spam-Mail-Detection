# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages.
2.Import the dataset to operate on.
3.Split the dataset.
4.Predict the required output

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Santha ramanath M
RegisterNumber:212223220097  
*/
```
```
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
## Result output:

![324672241-aad50e36-cc59-4c8d-97ee-43b4e6d78754](https://github.com/Santharamanath/Implementation-of-SVM-For-Spam-Mail-Detection/assets/149035289/552bc23c-070d-40b4-b634-6b6bd132bfa8)


## data.head():

![324672318-3fca5130-2f69-4cf5-9a51-df7ff9ba4766](https://github.com/Santharamanath/Implementation-of-SVM-For-Spam-Mail-Detection/assets/149035289/05d636dc-88da-4a1e-a5d4-0f8a17140b08)

## data.info():
![324672418-415311e1-20d5-4702-bd59-a11a5b3cdeb9](https://github.com/Santharamanath/Implementation-of-SVM-For-Spam-Mail-Detection/assets/149035289/8ffc4040-e69e-45e4-b91f-4f7a2800aa84)


## data.isnull().sum():
![324672501-4ee57de7-de51-4c29-b063-29ef4f8e1b3f](https://github.com/Santharamanath/Implementation-of-SVM-For-Spam-Mail-Detection/assets/149035289/277f67e6-613d-4d38-a8cb-ac87cfe64fa4)


## Y_prediction value:

![324672596-3e8f3cb9-023e-48ba-80c3-fd54a54982c4](https://github.com/Santharamanath/Implementation-of-SVM-For-Spam-Mail-Detection/assets/149035289/84abef50-8800-49cd-a440-a684dae7a8f9)

## Accuracy value:

![324672673-1c70231b-3edc-4b5f-a97a-18c0beab29db](https://github.com/Santharamanath/Implementation-of-SVM-For-Spam-Mail-Detection/assets/149035289/e152ac77-aafc-4423-8544-ccd8188ee46f)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
