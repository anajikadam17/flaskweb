# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 09:36:28 2020

@author: Anaji
"""
from flask import Flask, request, jsonify, render_template,url_for,request
from flask_cors import cross_origin
import pandas as pd
import numpy as np
import pickle

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


app = Flask(__name__)

model = pickle.load(open('model/model.pkl', 'rb'))
clf = pickle.load(open('model/nlp_model.pkl', 'rb'))
cv = pickle.load(open('model/tranform.pkl','rb'))
#flight_price_rf = pickle.load(open('model/flight_price_rf1.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/portfolio')
def Portfolio():
    return render_template('portfolio-details.html')

@app.route('/blog')
def Inner_page():
    return render_template('inner-page.html')
    
@app.route('/task1')
def Project1():
    return render_template('task1.html')
    
@app.route('/predict1',methods=['POST'])
def predict1():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    return render_template('task1.html', prediction_text='Employee Salary should be $ {}'.format(output))

@app.route("/task2")
def task2():
  return render_template("task2.html")
@app.route('/predict2',methods=['POST'])
def predict2():
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('task2.html',prediction = my_prediction)

if __name__ == "__main__":
    app.run(debug=True)