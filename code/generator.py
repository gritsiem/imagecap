# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 19:14:22 2018

@author: ghrit
"""

from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
from PIL import Image  
import numpy as np
 
def load_doc(filename):
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text
 
# load a pre-defined list of photo identifiers
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	for line in doc.split('\n'):
		if len(line) < 1:
			continue
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return list(set(dataset))

def extract_features(filename):
	model = VGG16()
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	image = load_img(filename, target_size=(224, 224))
	image = img_to_array(image)
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	image = preprocess_input(image)
	feature = model.predict(image, verbose=0)
	return feature
 
# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None
 
# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
	in_text = 'startseq'
	for i in range(max_length):
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		sequence = pad_sequences([sequence], maxlen=max_length)
		yhat = model.predict([photo,sequence], verbose=0)
		yhat = argmax(yhat)
		word = word_for_id(yhat, tokenizer)
		if word is None:
			break
		in_text += ' ' + word
		if word == 'endseq':
			break
	return in_text
 
# load the tokenizer
filename = 'Flickr_8k.testImages.txt'
test = load_set(filename)
tokenizer = load(open('tokenizer2.pkl', 'rb'))
max_length = 34
model = load_model('model_inject5.h5')
directory = 'Flicker8k_Dataset'
# load and prepare the photograph

im=test[int(np.random.randint(0, 1000, size=1))]
fn = directory + '/' + im +'.jpg'
print(fn)
photo = extract_features(fn)
img = Image.open(fn)
img.show()
# generate description
description = generate_desc(model, tokenizer, photo, 33)
print(description)

