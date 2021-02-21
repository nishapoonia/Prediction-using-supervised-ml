# -*- coding: utf-8 -*-
"""

@author: NISHA POONIA
task : prediction using supervised ML
"""
import pandas as pd
import matplotlib.pyplot as plt 

#dataset
dataset = pd.read_csv("student_studyhrs.csv")
print("DATASET:")
print(dataset)

#Independent set
X = dataset.iloc[ : , 0:1 ].values
print("X(STUDY HOURS):")
print(X)

#Dependent set
Y = dataset.iloc[ : , 1:2  ].values
print("Y(SCORES):")
print(Y)
#TRAIN SET AND TEST SET. Spliting the data set into training and testing set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
print("X TRAINING SET:",X_train);
print("Y TRAINING SET:",Y_train);
print("X TESTING SET:",X_test);
print("Y TESTING SET:",Y_test);

#machine learning algorithm for predicting scores
from sklearn.linear_model import LinearRegression
slr=LinearRegression()
slr.fit(X_train,Y_train)
y_pred=slr.predict(X_test)

#plotting training set
plt.plot(X_test,y_pred)
plt.title('Study hours vs Percentage Score (Training set)')
plt.scatter(X_train,Y_train,color="red")
plt.xlabel("studyhrs")
plt.ylabel("percentage score")
plt.show()

#plotting test set
plt.plot(X_test,y_pred)
plt.title('Study hours vs Percentage Score (Test set)')
plt.scatter(X_test,Y_test,color="blue")
plt.xlabel("studyhrs")
plt.ylabel("percentage score")
plt.show()

#To retrieve the intercept:
print('intercept:',slr.intercept_)
#For retrieving the slope:
print('slope:',slr.coef_)


#finding percentage of student based on study hours (9.25 hrs/day)

hrs= float(input("Enter the no. of study hours"))
stud_percent=slr.predict([[hrs]])
print('The predicted percentage score of a student with',hrs,' study hours is:',stud_percent)


#WE predicted the score usinh supervised ML
#THANK YOU