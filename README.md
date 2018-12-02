# Sentiment Analysis

The Women's Clothing E-Commerce Reviews is used as the dataset for this project. 

Data pre-processing is the first step in our model. Generally, in reviews, there are many punctuations, numbers, and stop-words. To make data ready for feeding into the network, it should be first tokenized. Therefore, we removed all zero-length reviews. Then, we split sentences into words and removed punctuations, stop-words, and non-alphabetic tokens.

The second phase of classification in our model is the learning step which contains word embedding and training the model.
To feed vectors to the network, they have to be all in the same size. Therefore, the maximum review size have to be found. Then, zero-padding technique is used to make all vectors size the same as the maximum size.

In this project, I impliment a CNN and an LSTM layers for classification of the reviews in the dataset. 

A rectified linear unit (ReLU) activation applies on the CNN layer output; which replaces any negative outputs with zero. The ReLU layer is used in order to introduce non-linearity into the network. The output of this layer is the same shape as the input shape. Then, the output of the activation layer is fed into an LSTM layer. Finally, the LSTM output is sent to a fully-connected layer in order to produce a single, final output. 

The layer is followed by a simple sigmoid activation function to conform the output between 0 (Not Recommend) and 1 (Recommend).
