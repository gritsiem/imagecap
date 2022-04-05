# Image Captioning

Image captioning is the Artificial Intelligence problem of seeing pictures and describing what is in them. It involves Computer Vision as well as Natural Language Generation. It is quite a hard problem, and in this project I attempt to make and try to improve deep learning models for the same. There are many elements to this problem, so first, let's understand the existing literature.

### History
Image Captioning research has existed since the early twenty first century. The earlier methods use probabilistic learning models, which  limited in their capacity to compute such non linear functionality with flexibility. These include template based approaches or conditional random fields (CRF). Retrieval based approaches which use annotations and similarity scores work but do not offer any generation abilities and depend on the dataset.

With the popularity of neural networks and deep learning, the problem has opened up. In this we have encoder-decoder models, multimodal systems and attention. We also have models based on other learning method like reinforcement learning and GANs. Several new datasets have prompted many new experiments. Still, we are far from perfect.

### About the model
In this project, I focus only on encoder-decoder model. In these, we have an encoding unit which is a CNN to process the image, and a decoding unit which is a recurrent neural network to process the output data i.e. captions. These are inspired from sequence to sequence models used in language translation. In this case the visual data can be thought of as the first language.

There are 2 types of ways to organize encoder-decoder models:

 - Injector Architectures

In these, the LSTM receives both the image and sequence information while training. Intuitively, this means that the words are predicted using the visual cues from the image. At the same time, the image information is not necessary to learn the syntactic and structural rules of the language. So, it can add an overhead to the model.
 
![Inject Model diagram](/assets/models/inject.png)

 - Merge Architectures

 In this type of architecture, the LSTM layer only processes the sequential information provided by the caption. Intuitively, it misses some important image information, while predicting the next word. At the same time, it is less complex, as there are less information to learn from.

 ![Merge Model diagram](/assets/models/merge.png)

  We experiment with both and see which gives better results.

  #### Encoder

  For this project, we use pretrained CNNs as encoders. By pretrained, we mean that we use weights which have been already calculated by training on a huge dataset. I experimented with VGG16 and InceptionV3 convolution neural nets, using pretrained ImageNet weights. These weights have been set by running the CNN architecture through the ‘Image Net Large Scale Visual Recognition Challenge.‘ The challenge consisted of 1000 categories that include common objects that are seen in a high frequency in our daily lives.

  I decided to go with VGG16 finally even though it was beaten by InceptionV3 as it performed better on the dataset I use. Additionally, we only need to be concerned with a good representation of the image, and not the actual detection. I also experimented with finetuning the CNN with the dataset I use. This will be explained later.

  #### Decoder

  For decoder, Long Short Term Memory(LSTM) is used. LSTM‘s are used to learn and generate long term dependencies or sequential data. The LSTM are a form of recurrent neural nets that form a gated system to tackle the shortcomings of the vanilla neural nets, namely the vanishing gradient problem and the exploding gradient problem. 

  The repeating unit of an LSTM is not the same as that of a simple or Vanilla RNN. As opposed to a single neural net unit in the simple RNN, the LSTM have four neural units used to enhance the learning of long term dependencies.


  ### Project outline

  #### Dataset

  For this project to be feasible with our limited resources, I have used Flickr8k dataset, which has 1 GB size. It contains 8000 images with 6000 training set images and 1000 images each for test and validation sets. Each image contains 5 captions per image, for a total of 30,000 captions.

  #### Architecture
  ![Project flow diagram](/assets/models/process.png)

  The model is given an image as input and generates a string caption as output.

  In practicality, the architecture takes 2 inputs. An image feature set, and the sequence formed till now. Each caption starts with a //<start> token, which initiates generation of the rest of sentence, until the <end> token or maximum caption length is reached.

  For e.g., if an image has a ground truth caption “a dog is playing with ball“, then the input-output pairs to process the caption are as shown

  | Input |---| Output|
  | Image| Input Sequence | Output sequence|
  |------|----------------| ---------------|
  |P | "start" | "a" |
  |P | "start a" | "dog" |
  |P | "start a dog" | "is" |
  |P | "start a dog is" | "playing" |
  |P | "start a dog is playing" | "with" |
  |P | "start a dog is playing with" | "ball" |
  |P | "start a dog is playing with ball" | "end"|

  Now let's dive in to the code.

  ### Code

  1. Preprocessing Captions

  In this step, I remove the punctuations from all captions such as commas and full stops. Words of length less than 3 are also removed since they do not affect the caption's meaning and are called <b> stop words</b>.  Every word is converted to lower case to make the captions consistent, and a <start> and <end> tag is added to each caption to mark the beginning and termination for training. 
  
  ```
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

  ```

  Also, later on, I removed words which were rarely occurring in our dataset. Removing infrequently occurring words improved performance. This is because rare words do not occur enough times in our dataset to be properly learned by our model.
  
  ```
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
```

For full code, please see [file](code/prep_text2.py)
  

  2. Image Feature extraction

  Two variations were run for acquiring our image feature vector. In both instances we used the VGG16 model. We first trained our images using pre trained weights. Since the last layer is the classification layer and this does not concern the task of image captioning, we popped the last layer and worked with the fully connected layer that returns a vector of 4096.

  The second variation is the fine tuning of VGG net, instead of using the pre trained weights, new weights were reached by creating our own classification layer and training it on our dataset.
  
  We don't use the encoder in the training of the language model directly, to save training time. Instead we use the feature representation that is the output. We store these features in a python dict and pickle it. The file is omitted here because it is big (127 MB).

  For more information on feature extraction and finetuning see - [CNN training ] (code/encoder/doc.md)

  3. Sequence generation

  The data needs to be sequenced to be trained word by word through our LSTM. Each image has five captions. The maximum length of the caption is 34 without the reduction of vocabulary and 33 with the reduction of vocabulary. We pad the captions that do not reach the length of the longest caption. Each word is converted to a one hot vector and fed word by word. The photo features are also added for every time step of the LSTM. The sequencing is made easier using the tokenizer which streamlines our data. Note that this step is only done for training. 

  ```
  def create_sequences(tokenizer, max_length, desc_list, photo):
	X1, X2, y = list(), list(), list()
	# walk through each description for the image
	for desc in desc_list:
		# encode the sequence
		seq = tokenizer.texts_to_sequences([desc])[0]
		# split one sequence into multiple X,y pairs
		for i in range(1, len(seq)):
			# split into input and output pair
			in_seq, out_seq = seq[:i], seq[i]
			# pad input sequence
			in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
			# encode output sequence
			out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
			# store
			X1.append(photo)
			X2.append(in_seq)
			y.append(out_seq)
	return np.array(X1), np.array(X2), np.array(y)

  ```

  For full code, please see [file](code/models/merge.py)

  4. Generation

  The generation is done word by word. The test image is fed with the <start> tag and the generation continues until we encounter the <end> tag. The last layer of our model uses the Softmax activation, this returns a vector of predictions for the next word in our sequence. We use an argmax function which returns the word at the index of the most likely or the highest value in our output vector. This word is sequenced with our previous predictions and fed again into our model to make the next word prediction.

  For full code, please see [file](code/generator.py)

  5. Evaluation

  The literature on Image Captioning involves the usage of several metrics including BLEU, ROUGE, METEOR and CIDEr. For this problem, we have made use of the MSCOCO Evaluation Engine, that provides open source code to get the value of the above metrics.



#### Results and Analysis

The experiments performed definitely improved the model as can be seen by the following examples:

![Project flow diagram](/assets/examples.png)

The two main models that were dealt with were the inject and merge models. Both models work with different arrangement of data processing. Through our various experiments it was observed that the merge architecture performs better than the inject architecture. The fine tuning of the VGG model improved performance on the metrics. Reducing Vocabulary also improved the performance.

Problems noticed were:

1. Overfitting
The word "man" is often seen to be followed by the phrase "in red shirt" a lot, without it being so in the image. Also, due to dog being the most common noun and images of dogs being very common in the dataset, lots of dog images get a similar description even though they are very different. This can be attributed to the size of the dataset, which can be overcome by training on a bigger dataset such as MSCOCO.

2. Generalization is noticeably difficult for this model.
 In pictures that are relatively more complex, and full of visual information, the descriptions are not correct, like here:

 ![Project flow diagram](/assets/wrongex.png)

