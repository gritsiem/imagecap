# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 18:03:50 2018

@author: ghrit
"""

from numpy import argmax
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider
from keras.utils import plot_model

def load_doc(filename):
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text
 
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	for line in doc.split('\n'):
		if len(line) < 1:
			continue
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)
 
def load_clean_descriptions(filename, dataset):
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		tokens = line.split()
		image_id, image_desc = tokens[0], tokens[1:]
		if image_id in dataset:
			if image_id not in descriptions:
				descriptions[image_id] = list()
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			descriptions[image_id].append(desc)
	return descriptions
 
def load_photo_features(filename, dataset):
	all_features = load(open(filename, 'rb'))
	features = {k: all_features[k] for k in dataset}
	return features
 
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc
 
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer
 
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)
 
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None
 
def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        
        ohv = argmax(yhat)
        word = word_for_id(ohv, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text
 
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()
    evals=dict();
    gts, res=dict(),dict()
    for key,desc_list in descriptions.items():
        yhat = generate_desc(model, tokenizer, photos[key], max_length)
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())
        gts[key]=[d for d in desc_list]
        res[key]=[yhat]
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
    print('setting up scorers...')
    scorers = [
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
    ]
    for scorer, method in scorers:
        print('computing %s score...'%(scorer.method()))
        score, scores = scorer.compute_score(gts,res)
        evals[method]=score
        print("%s: %0.3f"%(method, score))
        
               
                
        
    
 
 
filename = 'Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
train_descriptions = load_clean_descriptions('descriptions2.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)
 
 
filename = 'Flickr_8k.testImages.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))
test_descriptions = load_clean_descriptions('descriptions2.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
test_features = load_photo_features('features.pkl', test)
print('Photos: test=%d' % len(test_features))

filenames = ['model_inject10.h5', 'model_inject11.h5','model_inject12.h5','model_inject13.h5']
for filename in filenames:
    print('\n'+filename)
    model = load_model(filename)
    evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)
'''
=====================================================
model_inject5.h5
BLEU-1: 0.317669
BLEU-2: 0.134677
BLEU-3: 0.078503
BLEU-4: 0.014104
setting up scorers...c
computing METEOR score...
METEOR: 0.173
computing Rouge score...
ROUGE_L: 0.372
computing CIDEr score...
CIDEr: 0.088

model_inject6.h5
BLEU-1: 0.566102
BLEU-2: 0.291887
BLEU-3: 0.180471
BLEU-4: 0.074293
setting up scorers...
computing METEOR score...
METEOR: 0.200
computing Rouge score...
ROUGE_L: 0.448
computing CIDEr score...
CIDEr: 0.159

model_inject7.h5
BLEU-1: 0.519271
BLEU-2: 0.267690
BLEU-3: 0.183742
BLEU-4: 0.084363
setting up scorers...
computing METEOR score...
METEOR: 0.202
computing Rouge score...
ROUGE_L: 0.429
computing CIDEr score...
CIDEr: 0.208

model_inject8.h5
model_inject8.h5
BLEU-1: 0.549495
BLEU-2: 0.299676
BLEU-3: 0.207521
BLEU-4: 0.099396
setting up scorers...
computing METEOR score...
METEOR: 0.206
computing Rouge score...
ROUGE_L: 0.442
computing CIDEr score...
CIDEr: 0.216

model_inject9.h5
BLEU-1: 0.545045
BLEU-2: 0.299624
BLEU-3: 0.206681
BLEU-4: 0.093925
setting up scorers...
computing METEOR score...
METEOR: 0.209
computing Rouge score...
ROUGE_L: 0.447
computing CIDEr score...
CIDEr: 0.237

model_inject10.h5
BLEU-1: 0.551285
BLEU-2: 0.301412
BLEU-3: 0.207737
BLEU-4: 0.099298
setting up scorers...
computing METEOR score...
METEOR: 0.211
computing Rouge score...
ROUGE_L: 0.447
computing CIDEr score...
CIDEr: 0.227

model_inject11.h5
BLEU-1: 0.504324
BLEU-2: 0.256826
BLEU-3: 0.179410
BLEU-4: 0.083882
setting up scorers...
computing METEOR score...
METEOR: 0.203
computing Rouge score...
ROUGE_L: 0.427
computing CIDEr score...
CIDEr: 0.206

model_inject12.h5
BLEU-1: 0.508494
BLEU-2: 0.273352
BLEU-3: 0.190736
BLEU-4: 0.088438
setting up scorers...
computing METEOR score...
METEOR: 0.206
computing Rouge score...
ROUGE_L: 0.432
computing CIDEr score...
CIDEr: 0.230

model_inject13.h5
BLEU-1: 0.517047
BLEU-2: 0.273917
BLEU-3: 0.191573
BLEU-4: 0.088935
setting up scorers...
computing METEOR score...
METEOR: 0.205
computing Rouge score...
ROUGE_L: 0.439
computing CIDEr score...
CIDEr: 0.223

'''