<span class="underline">ImageNet Classification with Deep Convolutional
Neural Networks</span>

Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2017, May). *ImageNet
Classification with Deep Convolutional Neural Networks*.
https://doi.org/10.1145/3065386

<span class="underline">Purpose of the study</span>

The authors in this paper have taken up object recognition as their
focus of the published paper. The current approaches that are being
considered for this method are machine learning techniques. The
performance of these models is generally enhanced with the help of
larger datasets and making use of more powerful models to reduce
overfitting and increase the accuracy of classification.

<span class="underline">Key concepts of this paper</span>

The concepts which are being touched on in this paper are machine
learning, neural networks, convolutional neural networks, and methods to
fit the data to the model which considers reducing overfitting.

<span class="underline">Research design and strategy</span>

With smaller datasets like the CIFAR-10 dataset, it is possible to carry
out recognition tasks easily, but when it comes to applying the same
concept to real world objects, the performance of the model might be
hindered due to the vast variability that is observed in objects. In
order to learn about thousands of objects from millions of images, a
model with a high learning capacity is needed. These models should be
trained with prior knowledge in order to be on par with the complexity
of the object recognition task. These large datasets that are used to
work with are the LabelMe and the ImageNet dataset.

The big jump was to move from machine learning to neural networks. A
neural network consists of neurons and a set of algorithms that works on
identifying the relationships in a dataset and accordingly carry out
tasks like prediction and classification. Each neuron is a mathematical
operation that has several inputs coming in and it multiplies the value
of those inputs by its weights and passes its aggregate through the
activation function and that is passed as an output, or an input to
another neuron. CNNs have fewer connections and parameters, making them
easier to train. In this paper, the authors trained one of the largest
convolutional neural networks to date, and they also wrote a highly
optimized GPU implementation of 2D convolution and other operations
inherent in training the CNN. New and unusual features were used to
improve the performance of the model and reduce the time taken to train
the model and used effective techniques to reduce overfitting of the
model. The final report had five convolutional and 3 fully connected
layers.

Coming to the data used, ImageNet is a 15 million plus labeled high
resolution image database, with roughly around 22,000 categories. The
images were collected from the internet and labeled by human labelers
using Amazon’s Mechanical Turk crowd-sourcing tool. There is an annual
competition called ImageNet Large Scale Visual Recognition Challenge
(ILSVRC) which uses a subset of the ImageNet dataset, having roughly
1000 images in each of 1000 categories. ILSVRC-2010 is the only version
where test set labels were available, and hence, most experiments were
performed on this data. The images were down sampled to 256x256
resolution by subtracting the mean activity over the training set from
each pixel. The network was trained on raw RGB values of the pixels.

Neurons in the network with nonlinearity are Rectified Linear Units
(ReLUs). They don’t need input normalization. If there are positive
inputs to a ReLU, the neuron learns. CNNs with ReLU train several times
faster than their equivalents with *tanh* units. Faster learning
influences the model performance with larger datasets. The network was
spread across 2 GPUs (half of the neurons on one GPU and the rest on the
other) as current GPUs can handle cross-GPU parallelization as they can
read and write between their memory units, considering that the GPUs
communicate only on certain layers.

Pooling layers in CNNs summarize the outputs of the neighboring neuron
groups in the same kernel map. With overlapping pooling, the top-1 and
top-5 error rates are reduced by 0.4% and 0.3% respectively.

The 8 layers used, of which first 5 are convolutional and 3 are fully
connected. The final layer’s output is fed into a 1000-way softmax
activation function, producing a distribution over the 1000 class
labels.

Overfitting was an issue that was combated using data augmentation,
where the dataset was enlarged using the label-preserving
transformations. One form is to generate image translations and
horizontal reflections. The other form was to alter the intensities of
the RGB channels by performing Principal Component Analysis.

The models were trained using the stochastic gradient descent with a
batch size of 128, momentum of 0.9 and a weight decay of 0.0005, which
reduces the model’s training error. The neuron biases were initialized
for all layers except for layers 1 and 3 with the constant 1,
accelerating the early stages of learning, providing ReLUs with positive
input. Layers 1 and 3 were initialized with constant 0. An equal
learning rate was used and manually adjusted throughout training, which
was initialized at 0.01 and reduced three times before termination.

Training time: 5-6 days on two NVIDIA GTX 580 3GB GPUs. 90 cycles
through a set of 1.2 million images.

Predictions of models must be combined, being a successful way to reduce
the test errors. With the neural network taking days of training time,
it seems infeasible, but there is a technique called dropout. It sets
the output of each hidden neuron with the probability of 0.5 to 0. By
doing so, they don’t contribute to forward pass and don’t participate in
back propagation. During testing, all the neurons are used, and their
outputs are multiplied by 0.5 as it is a reasonable approximation of the
distributions produced by the dropout networks.

<span class="underline">Results, Discussions, and Conclusions</span>

The results are presented in a tabular format where they conducted
series of experiments and observations with different versions of the
data. One of the major findings apart from the model optimization was
the distribution of workload between GPUs to obtain faster results. GPU
optimization was achieved. Throughout the paper there have been
citations that have been referred by the authors and there is a list of
references at the end of the paper.

The model achieved top-1 and top-5 test set error rates of 37.5% and 17%
respectively. The best that was achieved in the 2010 competition was
47.1% and 28.2%. The model was taken to the 2012 competition as well
where test set labels were not publicly available. The test error rates
couldn’t be reported.

The CNN achieves a top-5 error rate of 18.2%. 2 CNNs that were trained
on the Fall 2011 version of the dataset gave an error rate of 15.3%. The
model was used with the Fall 2009 version as well, where half of the
images were used for training and the other half for testing. Top-1
error rate was 67.4% and top-5 error rate was 40.9%. The best published
results on this dataset are 78.1% and 60.9%.

The results show that a large, deep convolutional neural network can
achieve record-breaking results on a vast high-featured dataset using
only supervised learning. The model’s performance decreases on omitting
a single convolutional layer. Depth is essential for achieving the
results. No unsupervised pre-training was used. The authors plan on
using these large and deep nets on video sequences where temporal
structure has missing information or vague information in images.

<span class="underline">Critical Analysis</span>

The paper is well structured, and the authors have explained the
motivation and the process that has been carried out in an orderly
manner. The math and graphical representations of the model’s
performance made it easier to understand the approach that they had
taken. Model optimization is something that everyone working in the
field strives for and it is a field of evergreen development. The more
optimal the model is, the better is its performance. The paper is put
across in a way that it is easily understandable without missing out on
the important keywords that are needed to stress on the concepts being
used here. The paper has made use of references, but there has been an
original approach to the problem as well with the GPU optimization and
overfitting reduction along with the training of the CNN. The quality of
the research is reflected in the results as they are promising and have
set benchmarks. The data used is vary vast and elaborate and it is very
essential for neural network models to have large volumes of data to
increase quality of performance.

Understanding the research needs a prerequisite knowledge of machine
learning and basics of neural networks, but it is written in a way that
even with a small introduction, the paper is understandable. The switch
from machine learning to neural networks is an appropriate and feasible
approach for the research problem and it can be mimicked for many other
uses, one of them being license plate identification to capture
violations. Given the resources, more deep and complex neural networks
can be used to experiment

From this paper, I have learnt that a lot can be done with the help of
neural networks. Neural networks have a lot of potential that is yet to
be implemented and that is something that caught my attention while I
was reading this paper. I look forward to knowing more about the
applications and other capabilities of neural networks.
