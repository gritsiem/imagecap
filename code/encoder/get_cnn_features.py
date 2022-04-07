# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 23:24:56 2018

@author: ghrit

Code to extract feature representation from Flickr image dataset
"""

from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
 
# extract features from each photo in the directory
def extract_features(directory):
    model = VGG16() #create an instance of keras VGG16 model
    model.layers.pop() # Remove the last layer of the model which assigns label 
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output) # expose the second last dense layer with 4096 neurons as output.
    print(model.summary())
    features = dict()
    
    # For all images in the set
    for name in listdir(directory):
        filename = directory + '/' + name
        # Load image as PIL form
        image = load_img(filename, target_size=(224, 224))
        
        # Create numpy array representation of image
        image = img_to_array(image)
        # preprocess the image according to VGG16 requirements
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        # Use the model to process the image to create feature representation
        feature = model.predict(image, verbose=0)
        image_id = name.split('.')[0]
        
        # Map representation to image id
        features[image_id] = feature
        print('>%s' % name)
    return features
 
directory = 'Flicker8k_Dataset'
features = extract_features(directory)
print('Extracted Features: %d' % len(features
                                     
dump(features, open('features.pkl', 'wb')) # save features.