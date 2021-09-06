import pandas as pd  # to read the data set
import numpy as np  # to perform operations with an array
import pickle
from flask import Flask, request, render_template
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier


# Load the csv file
iris = pd.read_csv("iris.csv")

print(iris.head())

# dispaly the information about the data set
iris.info()

# check null values
iris.isnull().sum()

# Select independent and dependent variable
x = iris[["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]]
y = iris["Class"]

# First we require train data and test data for validation
# tarining = 80%
# test = 10%
# validation = 10%
# stratify

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, stratify=y, random_state=3)
x_valid, x_test, y_valid, y_test = train_test_split(
    x_test, y_test, test_size=0.5, stratify=y_test, random_state=3)

print(iris['Class'].value_counts())
print(y_train.value_counts())
print(y_valid.value_counts())
print(y_test.value_counts())

# Apply dummy classifier
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(x_train, y_train)
print("\nDummy classifier accuracy")
print("Accuracy:", dummy_clf.score(x_train, y_train)*100)
print("Accuracy:", dummy_clf.score(x_valid, y_valid)*100)
print("Accuracy:", dummy_clf.score(x_test, y_test)*100)

# Apply base Logistic Regression
model = LogisticRegression()
model.fit(x_train, y_train)
print("\nLogistic regression accuracy")
print("Accuracy:", model.score(x_train, y_train)*100)
print("Accuracy:", model.score(x_valid, y_valid)*100)
print("Accuracy:", model.score(x_test, y_test)*100)

# Apply base KNN
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x_train, y_train)
print("\nKNN accuracy")
print("Accuracy:", neigh.score(x_train, y_train)*100)
print("Accuracy:", neigh.score(x_valid, y_valid)*100)
print("Accuracy:", neigh.score(x_test, y_test)*100)


# Make pickle file of our model
pickle.dump(model, open("model.pkl", "wb"))
