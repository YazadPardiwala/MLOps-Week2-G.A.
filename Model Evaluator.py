import joblib
import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates
import pickle
from sklearn.tree import plot_tree
from sklearn import metrics
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/data.csv')

model1 = joblib.load('model.joblib')
print(type(model1))

tr, te = train_test_split(data, test_size = 0.4, stratify = data['species'], random_state = 42)
X_tr = tr[['sepal_length','sepal_width','petal_length','petal_width']]
y_tr = tr.species
X_te = te[['sepal_length','sepal_width','petal_length','petal_width']]
y_te = te.species

y_tr_pred = model1.predict(X_tr)
y_te_pred = model1.predict(X_te)
tr_prec = metrics.precision_score(y_tr, y_tr_pred, average ='weighted')
te_prec = metrics.precision_score(y_te, y_te_pred, average = 'weighted')
print("The model has a train percision of ", tr_prec, " and a test precision of", te_prec, " using V1 of the dataset with 101 samples" )

with open("eval results.txt", "w") as f:
    f.write(f"The model has a train percision of {tr_prec} and a test precision of {te_prec}.")


