# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 18:10:49 2019

@author: ghrit

File to create custom targets for VGG16 Fine tuning experiment
"""
import numpy as np
from pickle import dump


# Read the text of a file
def load_file(fn):
    f = open(fn,'r')
    text = f.read()
    f.close()
    return text

# using the caption file provided by Flickr8k, collect all captions in one string and map them to image id.
def get_desc(text,dset):
    desc=dict()
    # Each caption is provided in separate line, even if for same image.
    lines = text.split("\n")
    
    # Loop through lines
    for line in lines:
        
        # Split on space to get all words in a list.
        tokens = line.split()
        
        # If any line is empty, skip
        if len(line) < 2:
            continue
        # Get image id and image description
        imid, imd = tokens[0], tokens[1:]
        
        # Remove extension from image file name
        imid = imid.split('.')[0]
        
        # Convert the desciption words back to a single sentence.
        st = ' '.join(imd)
        
        # If image id has not been seen before, add it as a new key and initialize with first caption.
        if imid not in desc:
            desc[imid]=st
            
        # for the rest 4 captions, just add them to same string.
        desc[imid]=desc[imid]+" " +st
        
    # Return a dict mapping image ids to their caption words.
    return desc

# Returns a dictionary, mapping image id to target vector of 300 words, defined at the bottom of the file.
def get_targets(desc, labels, num_labels):
    targets = dict()
    
    # loop through all images
    for key in desc.keys():
        target =list()
        
        # num_labels is the number of words we want to include as labels - finalized on 300
        for i in range(num_labels):
            # if the label word is in the image's captions, assign it 1
            if labels[i] in desc[key]:
                target.append(1)
             # else assign it a 0
            else: target.append(0)
        # Map the constructed target vector to image id.
        targets[key]=target 
    return targets


def load_set(filename):
    doc = load_file(filename)
    dataset = list()
    # process line by line
    for line in doc.split('\n'):
        # skip empty lines
        if len(line) < 1:
            continue
        # get the image identifier
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)
                   
fn='C:\\Users\\ghrit\\Documents\\major_project\\code\\Flickr8k.lemma.token.txt'
text = load_file(fn)
train = load_set('C:\\Users\\ghrit\\Documents\\major_project\\code\\Flickr_8k.trainImages.txt')
description = get_desc(text, train)

labels = ['dog', 'man', 'woman', 'person', 'boy', 'girl', 'water', 'child', 'shirt', 'ball', 'grass','field', 'group', 'snow', 'air', 'beach', 'player', 'mouth', 'street', 'rock','bike', 'dress', 'mountain', 'jacket', 'camera', 'pool', 'orange', 'race', 'hat', 'toy', 'hand', 'background', 'park', 'brown', 'tree', 'soccer', 'building', 'wall', 'face', 'pink', 'ride', 'dirt', 'watch', 'stick', 'car', 'kid', 'skateboard', 'football', 'crowd', 'picture', 'hill', 'bicycle', 'sand', 'people', 'blue', 'wave', 'ocean', 'smile', 'tennis', 'baby', 'head', 'hair', 'top', 'area', 'basketball', 'road', 'trick', 'slide', 'bench', 'arm', 'blond', 'sidewalk', 'game', 'swing', 'ground', 'helmet', 'fence', 'skateboarder', 'path', 'frisbee', 'horse', 'lake', 'ramp', 'track', 'city', 'boat', 'side', 'suit', 'baseball', 'coat', 'swim', 'sign', 'wood', 'cover', 'motorcycle', 'guy', 'rope','pants', 'table', 'snowboarder', 'sunglasses', 'team', 'uniform', 'back', 'river', 'bird', 'dance','lady', 'couple', 'rider', 'cap', 'bag', 'outfit', 'surfer', 'biker', 'ice', 'yard', 'midair', 'line', 'chair', 'glasses', 'pole', 'flower', 'cliff', 'body', 'playground', 'fountain', 'backpack', 'step','skier', 'toddler', 'leg', 'outdoors', 'guitar', 'collar', 'flag', 'drink', 'brick', 'edge', 'splash', 'shore', 'costume', 'board', 'floor', 'night', 'bed', 'jean', 'trail', 'skate', 'house', 'bar', 'fight', 'sky', 'forest', 'surfboard', 'window', 'fall', 'climber', 'surf', 'paint', 'bridge', 'surround', 'hockey', 'day', 'tan', 'course', 'clothes', 'shop', 'train', 'leaf', 'jersey', 'obstacle', 'leash', 'room', 'sweater', 't-shirt', 'point', 'set', 'clothing', 'number', 'middle', 'perform', 'eye', 'greyhound', 'stone', 'tire', 'trunk', 'store', 'stair', 'tongue', 'lawn', 'winter', 'trampoline', 'stream', 'hiker', 'animal', 'fire', 'rail', 'mud', 'seat']
num_labels = len(labels) #300
targets = get_targets(description,labels,num_labels)
dump(targets, open('targets300.pkl', 'wb'))
