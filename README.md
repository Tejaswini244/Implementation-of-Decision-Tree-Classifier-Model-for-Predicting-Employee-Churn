# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: R.TEJASWINI
RegisterNumber: 212224230218
*/
```
```
import pandas as pd
data=pd.read_csv("Employee.csv")
print("data.head():")
data.head()
```
```
print("data.info():")
data.info()
```
```
print("isnull() and sum():")
data.isnull().sum()
```
```
print("data value counts():")
data["left"].value_counts()
```
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
```
```
print("data.head() for Salary:")
data["salary"]=le.fit_transform(data["salary"])
data.head()
```
```
print("x.head():")
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
```
```
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
```
```
print("Accuracy value:")
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
```
print("Data Prediction:")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

```
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plot_tree(dt, feature_names=x.columns, class_names=['salary', 'left'], filled=True)
plt.show()

```
## Output:

<img width="1374" height="280" alt="Screenshot 2025-10-15 125702" src="https://github.com/user-attachments/assets/f4dba010-3a55-4b3f-b1c0-66ec5d69d2df" />

<img width="519" height="406" alt="Screenshot 2025-10-15 125714" src="https://github.com/user-attachments/assets/34892127-3e7c-4d0b-a0d0-768ea6c03bfd" />



<img width="324" height="515" alt="Screenshot 2025-10-15 125726" src="https://github.com/user-attachments/assets/5b519bdf-7b4b-4336-9b95-2221fb3c41e4" />

<img width="300" height="247" alt="Screenshot 2025-10-15 125747" src="https://github.com/user-attachments/assets/9bfdc729-e4a2-491a-aaca-e2778062a887" />

<img width="1368" height="283" alt="Screenshot 2025-10-15 125804" src="https://github.com/user-attachments/assets/d78febc7-99d6-44b0-88b4-65df5efc0c2a" />

<img width="1370" height="275" alt="Screenshot 2025-10-15 125817" src="https://github.com/user-attachments/assets/27cc6e27-458b-48f8-adc5-2bd92eaf8a0d" />

<img width="207" height="69" alt="Screenshot 2025-10-15 125826" src="https://github.com/user-attachments/assets/f84fc551-e475-49f4-86f5-b6616c2bcdbd" />

<img width="1368" height="113" alt="Screenshot 2025-10-15 125835" src="https://github.com/user-attachments/assets/2408d424-294d-4936-8bc2-fe7f228a1dce" />

<img width="814" height="583" alt="Screenshot 2025-10-15 125850" src="https://github.com/user-attachments/assets/3545dd41-136e-4e54-a5ba-02d5e995e9df" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
