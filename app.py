# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 12:50:58 2021

@author: HP
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(X) for X in request.form.values()]
    final_features = [np.array(int_features)]
    #final_features = [[one, two, three]]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Cart_Abandonment {}'.format(output))
   


if __name__ == "__main__":
    app.run(debug=True)