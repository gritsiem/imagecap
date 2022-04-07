# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 13:43:24 2019

@author: ghrit
"""
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD
from os import listdir
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from pickle import dump
from pickle import load
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
 
# load a pre-defined list of photo identifiers
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return list(set(dataset))

def load_targets(fn,dset):
    targets = load(open(fn, 'rb'))
    # filter features
    targets = {k: targets[k] for k in dset}
    return targets


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

def tr_genr(directory, targets,dset):
    while 1:
        for name in dset:
            image = directory + '/' + name + ".jpg"
            image = preprocess(image)
            y = np.array(targets[name]).reshape(1,200)
            x = np.array(image)
            yield [x,y]
'''          
def te_genr(directory, targets,dset):
    while 1:
        for name in dset:
            image = directory + '/' + name + ".jpg"
            image = preprocess(image)
            y = np.array(targets[name]).reshape(1,200)
            x = np.array(image)
            yield [x,y]
            
'''
   
def get_initialized_inception(gen, old_model):
    x = old_model.output
    x = GlobalAveragePooling2D()(x)

    # add fully-connected layer
    x = Dense(512, activation='relu')(x)

    # add output layer
    predictions = Dense(200, activation='sigmoid')(x)

    model = Model(inputs=old_model.input, outputs=predictions)

    # freeze pre-trained model area's layer
    for layer in old_model.layers:
        layer.trainable = False

    # update the weight that are added
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    #model.summary()
    #plot_model(model, to_file='model.png',show_shapes=True)
    model.fit_generator(gen,steps_per_epoch=200)

    # choose the layers which are updated by training
    return model   

def train_iv3(gen,te_gen,model):
    for layer in model.layers[:279]:
        layer.trainable = False
    
    for layer in model.layers[279:]:
        layer.trainable = True
        
    model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss='binary_crossentropy', metrics=['binary_accuracy','categorical_accuracy'])
    history = model.fit_generator(gen,steps_per_epoch=200, epochs=5,validation_data=te_gen,validation_steps=50)
    return model  ,history  

# training

directory='C:\\Users\\ghrit\\Documents\\major_project\\code\\Flicker8k_Dataset'

inception_model = InceptionV3(weights='imagenet', include_top=False)
train = load_set('Flickr_8k.trainImages.txt')
tr_targets=load_targets("targets.pkl",train[:200])
tr_gen=tr_genr(directory,tr_targets,train[:200])

test =load_set('Flickr_8k.testImages.txt')
te_targets=load_targets("targets.pkl",test[:50])
te_gen=tr_genr(directory,te_targets,test[:50])
'''
#filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
#checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
'''
model = get_initialized_inception(tr_gen, inception_model)
model,hist= train_iv3(tr_gen,te_gen,model)

