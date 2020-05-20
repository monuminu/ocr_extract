import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug import secure_filename
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import pandas as pd 
import time
from sqa_prediction import predict
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = '/mnt/d/Data_Science_Work/tapas/uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['csv'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get', methods = ['GET', 'POST'])
def chatbot_response():
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], "data.csv")
    data = pd.read_csv(file_path, header = None)
    data = data.head(20)
    userText = request.args.get('msg')
    return predict(data, [userText])

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file and allowed_file(file.filename):    
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], "data.csv")
        file.save(file_path)
        data = pd.read_csv(file_path)
        data = data.head(20)
        return render_template('index.html', file_success = "File Uploaded Successfully !!", column_names=data.columns.values, row_data=list(data.values.tolist()), zip=zip)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    PATH_TO_TEST_IMAGES_DIR = app.config['UPLOAD_FOLDER']
    TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR,filename.format(i)) for i in range(1, 2) ]
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)
if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)