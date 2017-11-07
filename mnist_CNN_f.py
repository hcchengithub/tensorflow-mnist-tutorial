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
W1 = tf.Variable(tf.truncated_normal([5,5,1,4], stddev=0.1))
# b1 = tf.Variable(tf.ones([4])/10)
b1 = tf.Variable(tf.zeros([4]))

# My Layer2
W2 = tf.Variable(tf.truncated_normal([4,4,4,8], stddev=0.1))  
# b2 = tf.Variable(tf.ones([8])/10)
b2 = tf.Variable(tf.zeros([8]))

# My Layer3
W3 = tf.Variable(tf.truncated_normal([4,4,8,12], stddev=0.1)) 
b3 = tf.Variable(tf.zeros([12]))

# My Layer4
W4 = tf.Variable(tf.truncated_normal([7*7*12, 200], stddev=0.1)) 
b4 = tf.Variable(tf.zeros([200]))

# My Layer5
W5 = tf.Variable(tf.truncated_normal([200, 10], stddev=0.1)) 
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
Y3r = tf.reshape(Y3, [-1, 7*7*12])

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
disable_the_visualisation = True
max_loop = 4001
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
    
CNN 跑起來了，但是不收斂
33>
33> exit
2017-11-06 19:53:59.816635: W d:\build\tensorflow\tensorflow-r1.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE instructions, but these are available on your machine and could speed up CPU computations.
2017-11-06 19:53:59.816784: W d:\build\tensorflow\tensorflow-r1.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE2 instructions, but these are available on your machine and could speed up CPU computations.
2017-11-06 19:53:59.819118: W d:\build\tensorflow\tensorflow-r1.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE3 instructions, but these are available on your machine and could speed up CPU computations.
2017-11-06 19:53:59.820053: W d:\build\tensorflow\tensorflow-r1.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-11-06 19:53:59.820621: W d:\build\tensorflow\tensorflow-r1.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-11-06 19:53:59.821147: W d:\build\tensorflow\tensorflow-r1.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-11-06 19:53:59.821916: W d:\build\tensorflow\tensorflow-r1.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-11-06 19:53:59.822320: W d:\build\tensorflow\tensorflow-r1.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
33> 0: accuracy:0.07 loss: 490.156
0: ********* epoch 1 ********* test accuracy:0.0958 test loss: 485.447
10: accuracy:0.09 loss: 230.14
20: accuracy:0.11 loss: 230.132
30: accuracy:0.07 loss: 230.786
40: accuracy:0.07 loss: 231.33
50: accuracy:0.1 loss: 230.042
50: ********* epoch 1 ********* test accuracy:0.1135 test loss: 230.15
60: accuracy:0.1 loss: 229.249
70: accuracy:0.17 loss: 230.136
80: accuracy:0.14 loss: 229.854
90: accuracy:0.09 loss: 230.678
100: accuracy:0.12 loss: 229.884
100: ********* epoch 1 ********* test accuracy:0.1028 test loss: 230.194
110: accuracy:0.22 loss: 228.703
120: accuracy:0.1 loss: 231.313
130: accuracy:0.09 loss: 230.216
140: accuracy:0.13 loss: 230.477
150: accuracy:0.1 loss: 229.74
150: ********* epoch 1 ********* test accuracy:0.1135 test loss: 230.123
160: accuracy:0.08 loss: 230.401
170: accuracy:0.09 loss: 231.091
180: accuracy:0.12 loss: 230.811
190: accuracy:0.1 loss: 230.559
200: accuracy:0.1 loss: 230.753
200: ********* epoch 1 ********* test accuracy:0.1135 test loss: 230.144
210: accuracy:0.06 loss: 230.982
220: accuracy:0.08 loss: 230.17
230: accuracy:0.14 loss: 230.072
240: accuracy:0.15 loss: 229.96
250: accuracy:0.2 loss: 228.836
250: ********* epoch 1 ********* test accuracy:0.1135 test loss: 230.132
260: accuracy:0.11 loss: 230.717
270: accuracy:0.14 loss: 230.091
280: accuracy:0.13 loss: 230.427
290: accuracy:0.03 loss: 231.803
300: accuracy:0.14 loss: 229.729
300: ********* epoch 1 ********* test accuracy:0.1135 test loss: 230.116
310: accuracy:0.17 loss: 229.281
320: accuracy:0.12 loss: 230.337
330: accuracy:0.11 loss: 230.801
340: accuracy:0.11 loss: 229.8
350: accuracy:0.12 loss: 229.736
350: ********* epoch 1 ********* test accuracy:0.1135 test loss: 230.111
360: accuracy:0.1 loss: 230.013
370: accuracy:0.19 loss: 229.937
380: accuracy:0.13 loss: 229.841
390: accuracy:0.16 loss: 229.785
400: accuracy:0.12 loss: 229.915
400: ********* epoch 1 ********* test accuracy:0.1135 test loss: 230.122
410: accuracy:0.08 loss: 230.475
420: accuracy:0.08 loss: 230.207
430: accuracy:0.13 loss: 230.155
440: accuracy:0.16 loss: 229.263
450: accuracy:0.15 loss: 229.723
450: ********* epoch 1 ********* test accuracy:0.1135 test loss: 230.101
460: accuracy:0.09 loss: 230.857
470: accuracy:0.17 loss: 228.977
480: accuracy:0.08 loss: 231.35
490: accuracy:0.06 loss: 231.514
500: accuracy:0.08 loss: 230.294
500: ********* epoch 1 ********* test accuracy:0.1135 test loss: 230.103
510: accuracy:0.1 loss: 230.43
520: accuracy:0.11 loss: 230.494
530: accuracy:0.08 loss: 230.874
540: accuracy:0.18 loss: 229.81
550: accuracy:0.15 loss: 228.833
550: ********* epoch 1 ********* test accuracy:0.1135 test loss: 230.1
560: accuracy:0.1 loss: 231.026
570: accuracy:0.1 loss: 230.302
580: accuracy:0.09 loss: 230.562
590: accuracy:0.13 loss: 229.929
600: accuracy:0.1 loss: 229.988
600: ********* epoch 2 ********* test accuracy:0.1135 test loss: 230.1
610: accuracy:0.1 loss: 230.331
620: accuracy:0.11 loss: 230.171
630: accuracy:0.15 loss: 229.403
640: accuracy:0.15 loss: 230.443
650: accuracy:0.15 loss: 229.709
650: ********* epoch 2 ********* test accuracy:0.1135 test loss: 230.11
660: accuracy:0.13 loss: 230.451
670: accuracy:0.1 loss: 230.461
680: accuracy:0.06 loss: 230.733
690: accuracy:0.09 loss: 230.215
700: accuracy:0.11 loss: 230.382
700: ********* epoch 2 ********* test accuracy:0.1135 test loss: 230.105
710: accuracy:0.06 loss: 230.579
720: accuracy:0.07 loss: 230.968
730: accuracy:0.09 loss: 230.796
740: accuracy:0.13 loss: 230.326
750: accuracy:0.1 loss: 230.348
750: ********* epoch 2 ********* test accuracy:0.1135 test loss: 230.103
760: accuracy:0.1 loss: 229.88
770: accuracy:0.18 loss: 229.809
780: accuracy:0.1 loss: 230.324
790: accuracy:0.1 loss: 230.007
800: accuracy:0.12 loss: 230.779
800: ********* epoch 2 ********* test accuracy:0.1135 test loss: 230.104
810: accuracy:0.06 loss: 230.396
820: accuracy:0.11 loss: 230.761
830: accuracy:0.14 loss: 230.464
840: accuracy:0.16 loss: 229.867
850: accuracy:0.11 loss: 230.248
850: ********* epoch 2 ********* test accuracy:0.1135 test loss: 230.106
860: accuracy:0.1 loss: 230.695
870: accuracy:0.1 loss: 230.106
880: accuracy:0.12 loss: 229.861
890: accuracy:0.17 loss: 229.68
900: accuracy:0.08 loss: 230.861
900: ********* epoch 2 ********* test accuracy:0.1135 test loss: 230.107
910: accuracy:0.12 loss: 230.113
920: accuracy:0.16 loss: 229.809
930: accuracy:0.16 loss: 228.943
940: accuracy:0.07 loss: 230.974
950: accuracy:0.12 loss: 230.037
950: ********* epoch 2 ********* test accuracy:0.1135 test loss: 230.105
960: accuracy:0.09 loss: 231.072
970: accuracy:0.13 loss: 229.832
980: accuracy:0.09 loss: 230.807
990: accuracy:0.13 loss: 230.154
1000: accuracy:0.12 loss: 230.142
1000: ********* epoch 2 ********* test accuracy:0.1135 test loss: 230.103
1010: accuracy:0.11 loss: 229.797
1020: accuracy:0.08 loss: 230.672
1030: accuracy:0.08 loss: 230.274
1040: accuracy:0.06 loss: 230.806
1050: accuracy:0.08 loss: 231.016
1050: ********* epoch 2 ********* test accuracy:0.1135 test loss: 230.105
1060: accuracy:0.15 loss: 229.359
1070: accuracy:0.09 loss: 229.941
1080: accuracy:0.13 loss: 229.474
1090: accuracy:0.11 loss: 229.912
1100: accuracy:0.05 loss: 230.498
1100: ********* epoch 2 ********* test accuracy:0.1135 test loss: 230.106
1110: accuracy:0.16 loss: 229.242
1120: accuracy:0.11 loss: 230.467
1130: accuracy:0.12 loss: 230.629
1140: accuracy:0.15 loss: 229.751
1150: accuracy:0.15 loss: 229.087
1150: ********* epoch 2 ********* test accuracy:0.1135 test loss: 230.1
1160: accuracy:0.14 loss: 230.02
1170: accuracy:0.06 loss: 230.545
1180: accuracy:0.16 loss: 229.769
1190: accuracy:0.14 loss: 229.767
1200: accuracy:0.12 loss: 229.98
1200: ********* epoch 3 ********* test accuracy:0.1135 test loss: 230.103
1210: accuracy:0.09 loss: 230.25
1220: accuracy:0.11 loss: 229.944
1230: accuracy:0.07 loss: 230.646
1240: accuracy:0.14 loss: 229.58
1250: accuracy:0.09 loss: 230.59
1250: ********* epoch 3 ********* test accuracy:0.1135 test loss: 230.101
1260: accuracy:0.08 loss: 230.51
1270: accuracy:0.07 loss: 230.91
1280: accuracy:0.12 loss: 229.657
1290: accuracy:0.15 loss: 229.867
1300: accuracy:0.11 loss: 230.263
1300: ********* epoch 3 ********* test accuracy:0.1135 test loss: 230.102
1310: accuracy:0.1 loss: 230.643
1320: accuracy:0.1 loss: 230.018
1330: accuracy:0.08 loss: 230.108
1340: accuracy:0.06 loss: 230.929
1350: accuracy:0.2 loss: 228.929
1350: ********* epoch 3 ********* test accuracy:0.1135 test loss: 230.103
1360: accuracy:0.11 loss: 230.742
1370: accuracy:0.11 loss: 229.44
1380: accuracy:0.14 loss: 229.363
1390: accuracy:0.1 loss: 230.37
1400: accuracy:0.07 loss: 230.505
1400: ********* epoch 3 ********* test accuracy:0.1135 test loss: 230.106
1410: accuracy:0.15 loss: 229.953
1420: accuracy:0.12 loss: 230.071
1430: accuracy:0.1 loss: 230.501
1440: accuracy:0.08 loss: 230.473
1450: accuracy:0.16 loss: 229.852
1450: ********* epoch 3 ********* test accuracy:0.1135 test loss: 230.105
1460: accuracy:0.1 loss: 230.525
1470: accuracy:0.06 loss: 230.639
1480: accuracy:0.08 loss: 230.651
1490: accuracy:0.19 loss: 229.081
1500: accuracy:0.12 loss: 230.03
1500: ********* epoch 3 ********* test accuracy:0.1135 test loss: 230.105
1510: accuracy:0.08 loss: 230.088
1520: accuracy:0.11 loss: 230.628
1530: accuracy:0.07 loss: 230.555
1540: accuracy:0.1 loss: 230.16
1550: accuracy:0.07 loss: 230.355
1550: ********* epoch 3 ********* test accuracy:0.1135 test loss: 230.105
1560: accuracy:0.12 loss: 230.254
1570: accuracy:0.2 loss: 229.08
1580: accuracy:0.09 loss: 230.309
1590: accuracy:0.12 loss: 230.248
1600: accuracy:0.12 loss: 231.096
1600: ********* epoch 3 ********* test accuracy:0.1135 test loss: 230.104
1610: accuracy:0.12 loss: 230.062
1620: accuracy:0.13 loss: 229.957
1630: accuracy:0.07 loss: 230.347
1640: accuracy:0.16 loss: 229.251
1650: accuracy:0.15 loss: 229.866
1650: ********* epoch 3 ********* test accuracy:0.1135 test loss: 230.105
1660: accuracy:0.06 loss: 230.718
1670: accuracy:0.12 loss: 229.992
1680: accuracy:0.08 loss: 230.356
1690: accuracy:0.11 loss: 230.372
1700: accuracy:0.1 loss: 230.286
1700: ********* epoch 3 ********* test accuracy:0.1135 test loss: 230.102
1710: accuracy:0.12 loss: 230.59
1720: accuracy:0.12 loss: 230.236
1730: accuracy:0.12 loss: 230.088
1740: accuracy:0.1 loss: 229.884
1750: accuracy:0.11 loss: 229.831
1750: ********* epoch 3 ********* test accuracy:0.1135 test loss: 230.105
1760: accuracy:0.15 loss: 229.271
1770: accuracy:0.18 loss: 229.014
1780: accuracy:0.09 loss: 230.507
1790: accuracy:0.15 loss: 230.412
1800: accuracy:0.1 loss: 230.21
1800: ********* epoch 4 ********* test accuracy:0.1135 test loss: 230.106
1810: accuracy:0.1 loss: 230.61
1820: accuracy:0.06 loss: 230.425
1830: accuracy:0.11 loss: 230.789
1840: accuracy:0.03 loss: 231.076
1850: accuracy:0.17 loss: 229.171
1850: ********* epoch 4 ********* test accuracy:0.1135 test loss: 230.104
1860: accuracy:0.18 loss: 229.38
1870: accuracy:0.08 loss: 230.773
1880: accuracy:0.13 loss: 229.672
1890: accuracy:0.1 loss: 230.75
1900: accuracy:0.08 loss: 230.511
1900: ********* epoch 4 ********* test accuracy:0.1135 test loss: 230.104
1910: accuracy:0.13 loss: 230.086
1920: accuracy:0.18 loss: 229.698
1930: accuracy:0.14 loss: 230.163
1940: accuracy:0.1 loss: 229.562
1950: accuracy:0.15 loss: 229.418
1950: ********* epoch 4 ********* test accuracy:0.1135 test loss: 230.104
1960: accuracy:0.13 loss: 230.075
1970: accuracy:0.12 loss: 229.711
1980: accuracy:0.08 loss: 230.214
1990: accuracy:0.11 loss: 229.319
2000: accuracy:0.08 loss: 230.821
2000: ********* epoch 4 ********* test accuracy:0.1135 test loss: 230.102
2010: accuracy:0.06 loss: 230.896
2020: accuracy:0.08 loss: 230.94
2030: accuracy:0.18 loss: 229.48
2040: accuracy:0.14 loss: 229.769
2050: accuracy:0.11 loss: 230.426
2050: ********* epoch 4 ********* test accuracy:0.1135 test loss: 230.104
2060: accuracy:0.04 loss: 231.456
2070: accuracy:0.13 loss: 229.609
2080: accuracy:0.12 loss: 229.618
2090: accuracy:0.09 loss: 230.007
2100: accuracy:0.13 loss: 229.92
2100: ********* epoch 4 ********* test accuracy:0.1135 test loss: 230.104
2110: accuracy:0.08 loss: 230.563
2120: accuracy:0.08 loss: 229.97
2130: accuracy:0.09 loss: 230.325
2140: accuracy:0.1 loss: 230.712
2150: accuracy:0.1 loss: 230.957
2150: ********* epoch 4 ********* test accuracy:0.1135 test loss: 230.105
2160: accuracy:0.1 loss: 230.036
2170: accuracy:0.13 loss: 230.34
2180: accuracy:0.09 loss: 230.043
2190: accuracy:0.11 loss: 230.187
2200: accuracy:0.1 loss: 230.172
2200: ********* epoch 4 ********* test accuracy:0.1135 test loss: 230.103
2210: accuracy:0.14 loss: 229.526
2220: accuracy:0.08 loss: 230.945
2230: accuracy:0.12 loss: 230.086
2240: accuracy:0.12 loss: 230.116
2250: accuracy:0.15 loss: 229.418
2250: ********* epoch 4 ********* test accuracy:0.1135 test loss: 230.104
2260: accuracy:0.15 loss: 229.189
2270: accuracy:0.07 loss: 230.707
2280: accuracy:0.14 loss: 228.962
2290: accuracy:0.12 loss: 230.015
2300: accuracy:0.11 loss: 230.129
2300: ********* epoch 4 ********* test accuracy:0.1135 test loss: 230.102
2310: accuracy:0.11 loss: 230.275
2320: accuracy:0.06 loss: 230.271
2330: accuracy:0.08 loss: 230.468
2340: accuracy:0.15 loss: 229.567
2350: accuracy:0.08 loss: 229.383
2350: ********* epoch 4 ********* test accuracy:0.1135 test loss: 230.105
2360: accuracy:0.12 loss: 230.75
2370: accuracy:0.11 loss: 230.802
2380: accuracy:0.07 loss: 231.197
2390: accuracy:0.13 loss: 229.927
2400: accuracy:0.14 loss: 229.34
2400: ********* epoch 5 ********* test accuracy:0.1135 test loss: 230.101
2410: accuracy:0.15 loss: 229.12
2420: accuracy:0.13 loss: 229.587
2430: accuracy:0.06 loss: 230.558
2440: accuracy:0.12 loss: 229.911
2450: accuracy:0.1 loss: 230.626
2450: ********* epoch 5 ********* test accuracy:0.1135 test loss: 230.103
2460: accuracy:0.15 loss: 230.121
2470: accuracy:0.08 loss: 230.944
2480: accuracy:0.12 loss: 229.95
2490: accuracy:0.16 loss: 229.144
2500: accuracy:0.1 loss: 230.337
2500: ********* epoch 5 ********* test accuracy:0.1135 test loss: 230.106
2510: accuracy:0.13 loss: 229.921
2520: accuracy:0.14 loss: 230.051
2530: accuracy:0.11 loss: 230.337
2540: accuracy:0.1 loss: 230.067
2550: accuracy:0.11 loss: 230.417
2550: ********* epoch 5 ********* test accuracy:0.1135 test loss: 230.103
2560: accuracy:0.13 loss: 229.97
2570: accuracy:0.12 loss: 229.903
2580: accuracy:0.12 loss: 230.303
2590: accuracy:0.12 loss: 229.533
2600: accuracy:0.12 loss: 229.95
2600: ********* epoch 5 ********* test accuracy:0.1135 test loss: 230.103
2610: accuracy:0.13 loss: 229.799
2620: accuracy:0.12 loss: 229.49
2630: accuracy:0.12 loss: 229.526
2640: accuracy:0.09 loss: 230.369
2650: accuracy:0.11 loss: 230.436
2650: ********* epoch 5 ********* test accuracy:0.1135 test loss: 230.104
2660: accuracy:0.09 loss: 229.871
2670: accuracy:0.12 loss: 229.725
2680: accuracy:0.11 loss: 230.366
2690: accuracy:0.1 loss: 230.352
2700: accuracy:0.12 loss: 229.926
2700: ********* epoch 5 ********* test accuracy:0.1135 test loss: 230.103
2710: accuracy:0.09 loss: 230.225
2720: accuracy:0.05 loss: 230.684
2730: accuracy:0.1 loss: 229.833
2740: accuracy:0.14 loss: 230.712
2750: accuracy:0.14 loss: 229.703
2750: ********* epoch 5 ********* test accuracy:0.1135 test loss: 230.104
2760: accuracy:0.12 loss: 229.723
2770: accuracy:0.08 loss: 231.336
2780: accuracy:0.16 loss: 229.294
2790: accuracy:0.16 loss: 229.508
2800: accuracy:0.08 loss: 230.754
2800: ********* epoch 5 ********* test accuracy:0.1135 test loss: 230.104
2810: accuracy:0.17 loss: 228.953
2820: accuracy:0.07 loss: 231.235
2830: accuracy:0.08 loss: 231.131
2840: accuracy:0.13 loss: 229.513
2850: accuracy:0.17 loss: 229.414
2850: ********* epoch 5 ********* test accuracy:0.1135 test loss: 230.104
2860: accuracy:0.11 loss: 230.73
2870: accuracy:0.12 loss: 230.092
2880: accuracy:0.08 loss: 231.161
2890: accuracy:0.11 loss: 229.952
2900: accuracy:0.11 loss: 230.302
2900: ********* epoch 5 ********* test accuracy:0.1135 test loss: 230.103
2910: accuracy:0.16 loss: 229.886
2920: accuracy:0.09 loss: 230.288
2930: accuracy:0.1 loss: 229.734
2940: accuracy:0.15 loss: 229.366
2950: accuracy:0.09 loss: 230.22
2950: ********* epoch 5 ********* test accuracy:0.1135 test loss: 230.102
2960: accuracy:0.13 loss: 229.794
2970: accuracy:0.18 loss: 228.891
2980: accuracy:0.1 loss: 230.359
2990: accuracy:0.16 loss: 229.345
3000: accuracy:0.12 loss: 230.004
3000: ********* epoch 6 ********* test accuracy:0.1135 test loss: 230.102
3010: accuracy:0.13 loss: 230.217
3020: accuracy:0.11 loss: 230.063
3030: accuracy:0.06 loss: 231.335
3040: accuracy:0.11 loss: 230.162
3050: accuracy:0.14 loss: 229.663
3050: ********* epoch 6 ********* test accuracy:0.1135 test loss: 230.1
3060: accuracy:0.1 loss: 230.331
3070: accuracy:0.11 loss: 229.507
3080: accuracy:0.09 loss: 230.806
3090: accuracy:0.18 loss: 229.516
3100: accuracy:0.06 loss: 230.458
3100: ********* epoch 6 ********* test accuracy:0.1135 test loss: 230.101
3110: accuracy:0.13 loss: 229.596
3120: accuracy:0.11 loss: 230.615
3130: accuracy:0.13 loss: 230.027
3140: accuracy:0.12 loss: 229.992
3150: accuracy:0.26 loss: 227.781
3150: ********* epoch 6 ********* test accuracy:0.1135 test loss: 230.104
3160: accuracy:0.17 loss: 229.532
3170: accuracy:0.15 loss: 229.472
3180: accuracy:0.06 loss: 230.458
3190: accuracy:0.13 loss: 229.84
3200: accuracy:0.11 loss: 230.233
3200: ********* epoch 6 ********* test accuracy:0.1135 test loss: 230.103
3210: accuracy:0.12 loss: 230.61
3220: accuracy:0.13 loss: 229.992
3230: accuracy:0.1 loss: 230.199
3240: accuracy:0.1 loss: 230.527
3250: accuracy:0.11 loss: 229.976
3250: ********* epoch 6 ********* test accuracy:0.1135 test loss: 230.103
3260: accuracy:0.08 loss: 230.258
3270: accuracy:0.15 loss: 230.004
3280: accuracy:0.06 loss: 230.823
3290: accuracy:0.1 loss: 230.399
3300: accuracy:0.14 loss: 229.969
3300: ********* epoch 6 ********* test accuracy:0.1135 test loss: 230.103
3310: accuracy:0.07 loss: 229.963
3320: accuracy:0.1 loss: 231.053
3    
    
'''
