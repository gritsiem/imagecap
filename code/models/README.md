# Model

This is a record of the architectures tried for the image captioning mode.

## Layers

While the two major components of the architecture are the CNN and LSTM layers, as they form the basis of solving this problem, there are still some additional layers that are required to make a robust model - such as regularization to prevent overlearning. So let's take a look.

#### Embedding Layer
Word Embeddings have been used in a lot of language related tasks, and were invented to have better word representations. Words are often represented using 1-in-k or one-hot-vector representations. In these, the word is determined by putting 1 in a specified index of a vocabulary sized zero vector. This had a two major disadvantages. Firstly, the sequences take a lot of space. In many NLP tasks, it is not uncommon to have vocabulary of millions of words. 
Secondly, the vector space defined by these representations do not have any semantic significance. For example. In these representations, the words “apple“ and “orange“ are as closely related as the pair of words “guitar“ and “sea“, i.e. to the model training on these sequences, both those words are separate, unrelated classes of objects. We, on the other hand know, that most of the sentences containing the word apple, will be valid sentences even if apple is replaced by orange. For example, “I like to eat apples“ and “I like to eat oranges“ are valid in the sense both are fruits. Thus one-hot encodings do not catch this hierarchy properly. Word embeddings were designed to do exactly that.

Word embeddings are a vector representation of words. We can think of them as scoring a word on a number of different properties such as colour, gender, living/non living etc. This system automatically makes vector of similar concepts similar as well. 
The vector space of such an embedding system would have clusters of objects that are similar. For example, all animal words will be closer to each other, and separate from human profession words. In addition, it becomes possible to have vector operations on these words that make semantic sense as well. 

There are two ways to implement the concept of word embeddings. The first is to use pre-trained word embeddings available on the internet. These have been trained on corpus of millions of words or more, and can be used when  you do not have a very large vocabulary.  Yet, they have known not to have significant performance boost in language generation tasks. In such cases, Keras provides Embedding layers, that can be trained on your specific task and are compatible to your specific network. For this project we use Keras provided Embedding layer.

#### Dropout

Dropout is the regularization technique used in Neural Networks. Regularization was introduced to counter the problem of overfitting in machine learning problems. Overfitting means that the model achieves very good training set performance, but poor test
set scores. This means that your model fails to generalizes when encountering newer instances of a problem.

Dropout layer acts as a regularization mechanism in neural networks, by randomly switching off a fraction of units of a particular layer in each forward propagation and weight update cycle. As a result, those units do not affect the output and weights for that particular cycle. While defining the dropout layer, we can choose the fraction of units we wish to ignore. For this problem, I have used a 0.5 value. We have placed 2 Dropout Layers: one after processing the image features, and one after the embedding layers. These are the layers closely related to the inputs and thus are good candidates for regularization. 

It is important to note, that overfitting is very easy in our particular problem, which can be seen by the fact that not even state-of-the-art models have yet achieved a good overall test set performance.

## Activation Functions

Activation functions in a neural network are a mechanism to provide non linearity and the ability to compute complex functions to the network. There are a variety of activation functions such as sigmoid, tanh. relu, etc. 

For the intermediate layers, i have used Rectified Linear Unit as activation. It has the advantage over tanh and Sigmoid of avoiding the vanishing gradient problem. Vanishing gradient problem causes problem in learning a function by having too small
a gradient for having significant weight updates. Relu ensures non zero gradient and thus solves the vanishing gradient problem.

For our output layer, I have used Softmax activation. Softmax activation outputs a set of numbers that sum to 1, essentially giving a probability distribution. This is helpful when learning a classification problem with multiple classes, as it calculates the
probability of each. Since our problem involves predicting the next word from a number of possible words (vocabulary), it is essentially a multi-class classification problem as well.

## Loss functions

I have used a categorical crossentropy loss function for this problem. Categorical crossentropy compares the predicted class and the expected class, and sums it all over the training examples. Thus, this loss function modifies weights in a manner so that the
correct class is selected. At each step, I predict one word and add it to the existing sequence. Thus, the number of classes is vocabulary size.

## Architectures

Note that:

- 2971 is the reduced vocabulary size. Orginal size was 7579. 
- 33 is the size of the longest caption in the dataset (in words). This is the size of the input layer.
- Diiferent sizes of image feature inputs can be seen. These include VGG16 (4096), Inception (2048), and a more condensed version with 512 size representation.  This representation gave a slight improvement.

### 1. Inject

<img src="../../assets/models/inject_plot.png" alt="Simple Inject" width="50%"/>

In inject, LSTM layer takes input from the combined information of the images and the caption sentences.

### 2. Merge

<img src="../../assets/models/merge_plot.png" alt="Merge Simple" width="50%"/>

In merge, LSTM layer does not receive image feature representation and only trains on captions.

### 3. Composite

Even a composite model with 2 LSTMs, representing a mix of inject and merge was tried. It performed less than baseline so further experiments were not pursued.

<img src="../../assets/models/composite_plot.png" alt="Composite Model" width="50%"/>

Variations of merge and inject were also tested as can be seen ![here](assets/models). Smaller feature representations were tried in ![Condensed Merge](assets/models/merge_condensed_features_plot.png) model. More dense layers were added in this ![Deep Merge](assets/models/merge_deep_plot.png). Models with multiple LSTMs were also tried with same results, so it was decided to keep it simple instead.

## Points to note

- Lots of experiments have been done but only the ones with noticeable impact have been described here in detail. Lot of these have not been done with full testing due to lack of resources. InceptionV3 was not experimented on after the initial models, deeper models were only experimented with in case of finetuning and reduced vocabulary. Composite models were not used due to long training hours. Yet, there has been some improvement due to other methods than architecture.

- Training these models requires more resources. I still made the best of what I had. I have more than 100 h5 files in my folder including checkpoints. The biggest models had been the finetuned VGG16 which was around 3 GBs.

- This is for a small dataset of 6000 training images which is just not enough. Even if the proportions were changed to 7000:500:500, it did not improve performance.

- Hyperparameter tuning was done very sparingly. Dropout rates, layer sizes, embedding sized - all are a lot of work to tune. At most 1 experiment per hyperparameter was attempted. It took upwards of 4 hours in good cases on colab to achieve.

- I would like to make an attention architecture in Keras to see if there's any improvement. It has seemed to really improve captioning results.