import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime

from pandas import DataFrame
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
import os
import pickle

def my_data_labels(csv_file):
    data = pd.DataFrame(
        columns=["nose_x", "nose_y", "leftEye_x", "leftEye_y", "leftEar_x", "leftEar_y", "rightEar_x", "rightEar_y",
                                      "leftShoulder_x", "leftShoulder_y", "rightShoulder_x", "rightShoulder_y", "leftElbow_x", "leftElbow_y",
                                      "rightElbow_x", "rightElbow_y", "leftWrist_x", "leftWrist_y", "rightWrist_x", "rightWrist_y"])

    print "inside data labels"
    print columns


    data = data.append(csv_file, ignore_index=True)
    data.fillna(data.mean)
    print(pd.isnull(data).sum() > 0)
    return data


def create_input(user_list):
    data   = pd.DataFrame(columns=["nose_x", "nose_y", "leftEye_x", "leftEye_y", "leftEar_x", "leftEar_y", "rightEar_x", "rightEar_y",
                                 "leftShoulder_x", "leftShoulder_y", "rightShoulder_x", "rightShoulder_y", "leftElbow_x", "leftElbow_y",
                                 "rightElbow_x", "rightElbow_y", "leftWrist_x", "leftWrist_y", "rightWrist_x", "rightWrist_y"])
    labels = []
    print "create input"
    for i in range(len(user_list)):
        csv_file = userdata_dict[user_list[i]]
        # print(user_list[i])
        feature_matrix = pd.read_csv(csv_file, index_col=0)
        # print(feature_matrix['eating'])

        if "About" not in user_list[i]:
            for i in range(0, len(feature_matrix)):
                 labels.append("1")
        else:
            for i in range(0, len(feature_matrix)):
                 labels.append("0")
        data = data.append(feature_matrix, ignore_index=True)
        #data.fillna(data.mean)
        #print(pd.isnull(data).sum() > 0)

    return data, labels

data = pd.DataFrame(
    columns=["nose_x", "nose_y", "leftEye_x", "leftEye_y", "leftEar_x", "leftEar_y", "rightEar_x",
             "rightEar_y", "leftShoulder_x", "leftShoulder_y", "rightShoulder_x", "rightShoulder_y", "leftElbow_x", "leftElbow_y",
             "rightElbow_x", "rightElbow_y", "leftWrist_x", "leftWrist_y", "rightWrist_x", "rightWrist_y"])
#Read Feature matrix from file
print(data)
about_father_files = r"C:\\Users\\Radhika\\Desktop\\sem2\\mc\\assignment2\\Assignment_2_code\\Assignment_2_Python_And_Android_Code\\data"

data_files = os.listdir(about_father_files)


#print(data_files)

userdata_dict = {}

for user_ in data_files:
    userdata_dict[user_] = about_father_files + "/" + user_

user_list = list(userdata_dict.keys())


number_of_training_users = int(0.8*len(user_list))#[:80]
# print(number_of_training_users)
training_data, training_labels = create_input(user_list[:number_of_training_users])
test_data, test_labels = create_input(user_list[number_of_training_users:])

#SVM

clf = svm.SVC(gamma='scale')
clf.fit(training_data, training_labels)


y_test_output=clf.predict(test_data)
#print ("SVM:" + y_test_output)
#print ("y_test:" + y_test)

# Compute accuracy based on test samples
acc = accuracy_score(test_labels, y_test_output)
#acc = accuracy_score(y_test, y_test_output)
print "printing accuracy score"
print(acc)
filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))


def predict(csv_file):
    loaded_model = pickle.load(open(filename, 'rb'))
    user_test_data = my_data_labels(csv_file)
    #print(user_test_data)
    print "predict"
    score_f=0
    score_a=0
    result = loaded_model.predict(user_test_data)
    for i in result:
        if(i=='1'):
            score_f=score_f+1
        if(i=='0'):
            score_a=score_a+1
    print(score_a,score_f)
    if score_a>score_f:
        return "About"
    else:
        return "Father"
