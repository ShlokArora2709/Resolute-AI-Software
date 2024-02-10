from flask import Flask, render_template, request, redirect, send_file, jsonify
import pandas as pd
import task1,task2,task3
from task1 import predict_cluster,predicted_cluster
from task2 import test,new_df,accuracy
from task3 import hold,output
import numpy as np
app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/page1')
def home():
    return render_template('page1.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get data from the request
            data = request.json['data']

            # Convert data to a list of floats
            new_data_point = [float(value) for value in data]

            # Convert the list to a NumPy array
            new_data_point_array = np.array([new_data_point])

            # Predict cluster for the new data point
            predicted_cluster = predict_cluster(new_data_point_array)

            return jsonify({'predicted_cluster': str(predicted_cluster)})

        except ValueError as ve:
            return jsonify({'error': f'ValueError: {str(ve)}'})
        except Exception as e:
            return jsonify({'error': str(e)})

    
@app.route('/page2')
def page2():
    return render_template('page2.html', df=new_df,Accuracy=accuracy)

@app.route('/classify')
def classify():
    return render_template('page2.html', df=new_df,Accuracy=accuracy)

@app.route('/page3')
def page3():
    return render_template('page3.html', df=hold.to_html())

@app.route('/calculate')
def calculate():
    return render_template('page3.html', df=output.to_html())

if __name__ == '__main__':
    app.run(debug=True)
