#!/usr/bin/env python

import numpy as np
import joblib
from get_12ECG_features import get_12ECG_features
from itertools import compress
def run_12ECG_classifier(data,header_data,classes,model):

    num_classes = len(classes)
    current_label = np.zeros(num_classes, dtype=int)
    current_score = np.zeros(num_classes)
    scaler = joblib.load('scaler_v1.joblib')    #Edw

    # Use your classifier here to obtain a label and score for each class. 
    features=np.asarray(get_12ECG_features(data,header_data))
    feats_reshape = features.reshape(1,-1)
    feats_reshape = scaler.transform(feats_reshape)  #Edw
    #label = model.predict(feats_reshape)   #Edw
    #score = model.predict_proba(feats_reshape) #Edw
    #Todo: Modify the label and score
    score = model.predict(feats_reshape)    #Edw
    label = score.copy()
    label[label > 0.3] = 1
    label[label <= 0.3] = 0
    label = label == 1
    label = np.where(label == 1)[1]
    current_label[label] = 1

    for i in range(num_classes):
        current_score[i] = np.array(score[0][i])

    return current_label, current_score

def load_12ECG_model():
    from keras.models import load_model  # Edw
    # load the model from disk
    loaded_model = load_model('mlp_19042020.h5')

    return loaded_model
