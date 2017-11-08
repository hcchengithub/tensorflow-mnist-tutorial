# encoding: UTF-8
# Copyright 2016 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import tensorflow as tf
import tensorflowvisu

# -- moved into xray.f -- from tensorflow.examples.tutorials.mnist import input_data as mnist_data 
import peforth; peforth.ok(loc=locals(),cmd="include xray.f")

print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)

# neural network with 1 layer of 10 softmax neurons
#
# · · · · · · · · · ·       (input data, flattened pixels)       X [batch, 784] # 784 = 28 * 28
# \x/x\x/x\x/x\x/x\x/    -- fully connected layer (softmax)      W [784, 10]  b[10]
#   · · · · · · · ·                                              Y [batch, 10]

# The model is:
#
# Y = softmax( X * W + b)
#   X: matrix for 100 grayscale images of 28x28 pixels, flattened (there are 100 images in a mini-batch)
#   W: weight matrix with 784 lines and 10 columns
#   b: bias vector with 10 dimensions
#   +: add with broadcasting: adds the vector to each line of the matrix (numpy)
#   softmax(matrix) applies softmax on each line (HC: the virtical line of the 10 digits)
#   softmax(line) applies an exp to each value then divides by the norm of the resulting line
#   Y: output matrix with 100 lines and 10 columns

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])

# My Layer1 
W1 = tf.Variable(tf.truncated_normal([5,5,1,8], stddev=0.1))
# b1 = tf.Variable(tf.ones([4])/10)
b1 = tf.Variable(tf.zeros([8]))

# My Layer2
W2 = tf.Variable(tf.truncated_normal([4,4,8,16], stddev=0.1))  
# b2 = tf.Variable(tf.ones([8])/10)
b2 = tf.Variable(tf.zeros([16]))

# My Layer3
W3 = tf.Variable(tf.truncated_normal([4,4,16,24], stddev=0.1)) 
b3 = tf.Variable(tf.zeros([24]))

# My Layer4
W4 = tf.Variable(tf.truncated_normal([7*7*24, 400], stddev=0.1)) 
b4 = tf.Variable(tf.zeros([400]))

# My Layer5
W5 = tf.Variable(tf.truncated_normal([400, 10], stddev=0.1)) 
b5 = tf.Variable(tf.zeros([10]))

# flatten the images into a single line of pixels
# -1 in the shape definition means "the only possible dimension that 
# will preserve the number of elements"
# XX = tf.reshape(X, [-1, 784])

# Dropout, keep ratio, feed in 1 when testing, 0.75 when training
pkeep = tf.placeholder(tf.float32)

# The model ==> convolutional layers
stride = 1 # output is still 28x28
Ycnv1 = tf.nn.conv2d(X, W1, strides=[1,stride,stride,1], padding='SAME')
Y1 = tf.nn.relu(Ycnv1 + b1)
    
Ycnv2 = tf.nn.conv2d(Y1, W2, strides=[1,stride*2,stride*2,1], padding='SAME')
Y2 = tf.nn.relu(Ycnv2 + b2)

Ycnv3 = tf.nn.conv2d(Y2, W3, strides=[1,stride*2,stride*2,1], padding='SAME')
Y3 = tf.nn.relu(Ycnv3 + b3)
Y3r = tf.reshape(Y3, [-1, 7*7*24])

peforth.ok('33> ', cmd='cr')

Y4 = tf.nn.relu(tf.matmul(Y3r, W4) + b4)

# Y  = tf.nn.softmax(tf.matmul(Y2, W3) + b3)
Ylogits  = tf.matmul(Y4, W5) + b5
Y  = tf.nn.softmax(Ylogits)

# loss function: cross-entropy = - sum( Y_i * log(Yi) )
#                           Y: the computed output vector
#                           Y_: the desired output vector

# cross-entropy
# log takes the log of each element, * multiplies the tensors element by element
# reduce_mean will add all the components in the tensor
# so here we end up with the total cross-entropy for all images in the batch
# cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 1000.0  # normalized for batches of 100 images,
                                                            # *10 because  "mean" included an unwanted division by 10
# 為了避免算 cross entropy 時用到的 log() 跑出 NaN error, logits 自己
# 算出 softmax 來得出 cross entropy
cross_entropy0 = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits,labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy0) * 100 

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Learning Rate Decay 
learning_rate_decay = tf.placeholder(tf.float32)

# training, learning rate = 0.005
# train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)
# train_step = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(learning_rate_decay).minimize(cross_entropy)

# matplotlib visualisation
allweights = tf.reshape(W3, [-1])
allbiases = tf.reshape(b3, [-1])
I = tensorflowvisu.tf_format_mnist_images(X, Y, Y_)  # assembles 10x10 images by default
It = tensorflowvisu.tf_format_mnist_images(X, Y, Y_, 1000, lines=25)  # 1000 images on 25 lines
datavis = tensorflowvisu.MnistDataVis()

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# You can call this function in a loop to train the model, 100 images at a time
def training_step(i, update_test_data, update_train_data):

    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)

    # compute training values for visualisation
    if update_train_data:
        a, c, im, w, b = sess.run([accuracy, cross_entropy, I, allweights, allbiases], feed_dict={X: batch_X, Y_: batch_Y, pkeep:1.0})
        datavis.append_training_curves_data(i, a, c)
        datavis.append_data_histograms(i, w, b)
        datavis.update_image1(im)
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c))

    # compute test values for visualisation
    if update_test_data:
        a, c, im = sess.run([accuracy, cross_entropy, It], feed_dict={X: mnist.test.images, Y_: mnist.test.labels, pkeep:1.0})
        datavis.append_test_curves_data(i, a, c)
        datavis.update_image2(im)
        print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))

    # learning_rate_decay
    irmax = 0.005
    irmin = 0.00001
    knee = 1000
    ir = irmin + (irmax-irmin)*10**(-i/knee)
    # the backpropagation training step
    sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y, learning_rate_decay:ir,  pkeep:1.0})
    if peforth.vm.debug==22: peforth.ok('22> ',loc=locals(), cmd = "---xray--- marker ---xray--- :> [0] inport i autoexec exit")

if peforth.vm.debug==11: peforth.ok("bp11> ",loc=locals(),cmd="---xray--- marker ---xray--- :> [0] inport")

# to save the animation as a movie, add save_movie=True as an argument to datavis.animate
# to disable the visualisation use the "for i in range(2000+1)" line instead of the datavis.animate line
disable_the_visualisation = False
max_loop = 12001
if disable_the_visualisation:
    for i in range(max_loop): training_step(i, i % 50 == 0, i % 10 == 0)
else:    
    save_movie = False
    datavis.animate(training_step, iterations=max_loop, train_data_update_freq=10, test_data_update_freq=50, more_tests_at_start=True, save_movie=save_movie)

print("max test accuracy: " + str(datavis.get_max_test_accuracy()))

# final max test accuracy = 0.9268 (10K iterations). Accuracy should peak above 0.92 in the first 2000 iterations.

peforth.ok('Done>', loc=locals(),cmd=":> [0] constant parent@done") 

'''
    Ynote:
    心得筆記：Part 1 TensorFlow and Deep Learning without a PhD, (Google Cloud Next '17)
    
    CNN 跑起來了，但是不收斂 --> bias initial value has to be 
    zeros([n]) or ones([n]/10) , I don't know why.
    
'''
