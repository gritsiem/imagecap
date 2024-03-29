import string

# load file
def load_doc(filename):
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text
 
# extract descriptions for images
def load_descriptions(doc):
	mapping = dict()
	for line in doc.split('\n'):
		tokens = line.split()
		if len(line) < 2:
			continue
		image_id, image_desc = tokens[0], tokens[1:]
		image_id = image_id.split('.')[0]
		image_desc = ' '.join(image_desc)
		if image_id not in mapping:
			mapping[image_id] = list()
		mapping[image_id].append(image_desc)
	return mapping
 
def clean_descriptions(descriptions):
	table = str.maketrans('', '', string.punctuation)
	for key, desc_list in descriptions.items():
		for i in range(len(desc_list)):
                    desc = desc_list[i]
                    desc = desc.split()
                    desc = [word.lower() for word in desc]
                    desc = [w.translate(table) for w in desc]
                    desc = [word for word in desc if len(word)>1]
                    desc = [word for word in desc if word.isalpha()]
                    desc_list[i] =  ' '.join(desc)
 
# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
	all_desc = set()
	for key in descriptions.keys():
		[all_desc.update(d.split()) for d in descriptions[key]]
	return all_desc

		
def reduce_vocab(descriptions):
    all_train_captions = list()
    for key, val in descriptions.items():
        for cap in val:
            all_train_captions.append(cap)
    word_count_threshold = 5
    word_counts = dict()
    nsents = 0
    for sent in all_train_captions:
        nsents += 1
        for w in sent.split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1
            
    for key, desc_list in descriptions.items():
    	  for i in range(len(desc_list)):
               desc = desc_list[i]
               desc = desc.split()
               desc = [w for w in desc if word_counts[w] >= word_count_threshold]
               desc_list[i] =  ' '.join(desc)
 
# save descriptions to file, one per line
def save_descriptions(descriptions, filename):
	lines = list()
	for key, desc_list in descriptions.items():
		for desc in desc_list:
			lines.append(key + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()
 
filename = 'Flickr8k.token.txt'
doc = load_doc(filename)
descriptions = load_descriptions(doc)
print('Loaded: %d ' % len(descriptions))
clean_descriptions(descriptions)
reduce_vocab(descriptions)
vocabulary = to_vocabulary(descriptions)
print('Initial Vocabulary Size: %d' % len(vocabulary))
save_descriptions(descriptions, 'descriptions2.txt')