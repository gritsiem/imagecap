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

  ### Architecture
  ![Project flow diagram](/assets/models/process.png)

  The model is given an image as input and generates a string caption as output.

  In practicality, the architecture takes 2 inputs. An image feature set, and the sequence formed till now. Each caption starts with a //<start> token, which initiates generation of the rest of sentence, until the <end> token or maximum caption length is reached.

  For e.g., if an image has a ground truth caption “a dog is playing with ball“, then the input-output pairs to process the caption are as shown

  | Input | Output|
  | ------| ------|
  | Image| Input Sequence | Output sequence|
  |-----------------------| ---------------|
  |P "start" | "a" |
  |P "start a" | "dog" |
  |P "start a dog" | "is" |
  |P "start a dog is" | "playing" |
  |P "start a dog is playing" | "with" |
  |P "start a dog is playing with" | "ball" |
  |P "start a dog is playing with ball" | "end"|