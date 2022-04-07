# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 18:10:44 2019

@author: ghrit
"""

#from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from os import listdir
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from pickle import dump
from keras.models import load_model
from pickle import load



def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def preprocess(image_path):
    img = load_img(image_path, target_size=(299, 299))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def get_features(directory,model):
    new_input = model.input
    hidden_layer = model.layers[-2].output
    model_new = Model(new_input, hidden_layer)
    features = dict()    
    for name in listdir(directory):
        image = directory + '/' + name
        image = preprocess(image)
        temp_enc = model_new.predict(image)
        #print(temp_enc.shape)
        temp_enc = np.reshape(temp_enc, temp_enc.shape[1])
        image_id = name.split('.')[0]
        features[image_id]=temp_enc
        print('>%s' % name)
    return features    
    
directory='C:\\Users\\ghrit\\Documents\\major_project\\code\\Flicker8k_Dataset'
model=load_model('ftmodel.h5')
#features = get_features(directory,model)
#print('Extracted Features: %d' % len(features))
#dump(features, open('ft512_features.pkl', 'wb'))
