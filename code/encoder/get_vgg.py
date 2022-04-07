# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 14:06:57 2019

@author: ghrit
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
"""
from pickle import dump

import glob

# extract features from each photo in the directory
def extract_features(directory):
	# load the model
  '''
  model = VGG16()
	# re-structure the model
  for i in range(5):
    model.layers.pop()
	
  model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	# summarize
  #print(model.summary())
  '''
	# extract features from each photo 
  features = dict()
  for name in glob.glob(directory +'*.jpg'):
		# load an image from file
    filename = name
    print(name)
    image_id = name.split('.')[0]
    print(image_id)
    '''
    image = load_img(filename, target_size=(224, 224))
		# convert the image pixels to a numpy array
    image = img_to_array(image)
		# reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		# prepare the image for the VGG model
    image = preprocess_input(image)
		# get features
    feature = model.predict(image, verbose=0)
    feature = feature.reshape(196,512)
    '''
    # get image id
    
		# store feature
    features[image_id] = features
    print('>%s' % name)
  return features

directory='Flicker8k_Dataset/'
features = extract_features(directory)
print('Extracted Features: %d' % len(features))
dump(features, open('vgg_features.pkl', 'wb'))