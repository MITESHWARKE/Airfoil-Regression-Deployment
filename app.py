import pickle
import flask
from flask import Flask, request, app,jsonify,url_for,render_template
from flask import Response
from flask_cors import CORS
import numpy as np
import pandas as pd

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    #return 'Hello World
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():

    data=request.json['data']
    print(data)
    new_data=[list(data.values())]
    output=model.predict(new_data)[0]
    return jsonify(output)

@app.route('/predict',methods=['POST'])
def predict():

    data=[float(x) for x in request.form.values()] #saving the data entered in the form 
    final_features = [np.array(data)] # Converting the data into 2D array
    print(data)

    output=model.predict(final_features)[0] #reason for [0] stated in ipynb
    print(output)
    #output = round(prediction[0],2)
    return render_template('home.html', prediction_text="Airfoil pressure is {}".format(output))
        #prediction_text is a placeholder we created in HTML page we'll map it from here

if __name__=="__main__":
    app.run(debug=True)
  