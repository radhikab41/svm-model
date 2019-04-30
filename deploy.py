# -*- coding: utf-8 -*-
from flask import Flask, request
from flask_restful import Api
    #import eeg_feature_extractor as feature_extractor
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
from scipy import interpolate
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from math import pi
import svm

app = Flask(__name__)
api =  Api(app)

@app.route("")
def print():
    print "hello"

@app.route("/svm/runalgo", methods = ['POST'])
def runsvmalgo():
    user_data = request.files

    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    csv_input = csv.reader(stream)
    #print "hello from python"
    feature_matrix = pd.read_csv("user_csv", index_col=0)
    result = svm.predict(feature_matrix)
    #print "result"
    print(result)
    print(feature_matrix)
    print("\n\n")
    print(user_data)

    return result, 200


def get_path(filename):
    path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),"data",  filename),
            )
    return path

#api.add_resource(User,"/login/user/<string:userid>")




app.run(host="localhost",port="8888", debug = False)
