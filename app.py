# -*- coding: utf-8 -*-
from flask import Flask, flash, redirect, render_template, request, url_for
import pandas as pd
import pickle
import sklearn

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier

app = Flask(__name__)

@app.route('/')
def home():
    print(dir(sklearn.neighbors))
    return render_template('page.html')

@app.route("/results" , methods=['POST'])
def results():
    select = request.form
    s = ""
    df = pd.read_csv('models/_columns.csv', header=None, index_col=0).T
    for d in df:
        score = round(df[d][1]*100,2)
        columns = list(df[d][1:])
        X = []
        for c in columns:
            X.append(select[c])
        model = pickle.load(open(f'models/{d}.sav','rb'))
        pred = model.predict([X])
        s += f"According to the model, there is <strong>{score}%</strong> of chance that you "\
            f"ARE {'NOT' if pred[0] == 0 else ''} a <em>{d[:-11]}</em> consumer<br/>"
    return (s)

if __name__=='__main__':
    app.run()