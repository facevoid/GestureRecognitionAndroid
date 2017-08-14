from __future__ import print_function
import shutil
import os.path
import tensorflow as tf
import numpy as np
from mlxtend.preprocessing import one_hot

EXPORT_DIR = './model'
logs_dir = './tb/1/'

if os.path.exists(EXPORT_DIR):
    shutil.rmtree(EXPORT_DIR)

#this is for dataset 
x_ = np.loadtxt('/home/void/Desktop/tf_tutorial/all_data/train_image_reshaped.txt')
y_ = np.loadtxt('/home/void/Desktop/tf_tutorial/all_data/train_labels.txt')
yy_ = one_hot(y_,num_labels=10,dtype='float32')


x_test = np.loadtxt('/home/void/Desktop/tf_tutorial/all_data/test_image_reshaped.txt')
y_test = np.loadtxt('/home/void/Desktop/tf_tutorial/all_data/test_labels.txt')
yy_test = one_hot(y_test,num_labels=10,dtype='float32')

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 64
display_step = 10

# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)
dropout = 0.75  # Dropout, probability to keep units

# tf Graph input
with tf.name_scope('X_input'):
    x = tf.placeholder(tf.float32, [None, n_input])
with tf.name_scope('labels'):
    y = tf.placeholder(tf.float32, [None, n_classes])
# dropout (keep probability)
with tf.name_scope('dropout'):
  keep_prob = tf.placeholder(tf.float32)
  tf.summary.scalar('dropout_keep_probability', keep_prob)

#Serving minibatch
def next_batch(step):
      offset = (step * batch_size) % (x_.shape[0] - batch_size)
      batch_x = x_[offset:(offset + batch_size), :]
      batch_y = yy_[offset:(offset + batch_size), :]
      return batch_x,batch_y
# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    with tf.name_scope('relu_activation'):
        act = tf.nn.relu(x)
    return act


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create Model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    with tf.name_scope('conv1'):
        conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    with tf.name_scope('conv2'):
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    with tf.name_scope('Wx_plus_b'):
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    with tf.name_scope('fc'):
        fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    with tf.name_scope('output_layer'):
        with tf.name_scope('Wx_plus_b_final'):
            out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# Store layers weight & bias

weights = {
    # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32]),name='weight_c1'),
    # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64]),name='weight_c2'),
    # fully connected, 7*7*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024]),name='weight_d3'),
    # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, n_classes]),name='weight_final')
    }

biases = {
        'bc1': tf.Variable(tf.random_normal([32]),name='bias_c1'),
        'bc2': tf.Variable(tf.random_normal([64]),name='bias_c2'),
        'bd1': tf.Variable(tf.random_normal([1024]),name='bias_d1'),
        'out': tf.Variable(tf.random_normal([n_classes]),name='bias_final')
    }

# tf.summary.histogram('weights_1',wc1)
# tf.summary.histogram('weights_2',wc2)
# tf.summary.histogram('weights_d1',wd1)
# tf.summary.histogram('weights_1',w_final)
# tf.summary.histogram('biases_1',bc1)
# tf.summary.histogram('biases_2',bc2)
# tf.summary.histogram('biases_d1',bd1)
# tf.summary.histogram('biases_1',b_final)

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
with tf.name_scope('cross_entropy'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels= y))
with tf.name_scope('AdamOpt'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
tf.summary.scalar('cross_entropy', cost)

# Evaluate model
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_pred'):
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    with tf.name_scope('accuracy'):    
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.summary.scalar('accuracy', accuracy)

# Initializing the variables
merged = tf.summary.merge_all()

init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    writer =  tf.summary.FileWriter(logs_dir,sess.graph)
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = next_batch(step)
        # Run optimization op (backprop)
        summary,_ = sess.run([merged,optimizer], feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        writer.add_summary(summary, step * batch_size)

        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", \
          sess.run(accuracy, feed_dict={x: x_test[:256],
                                        y: yy_test[:256],
                                         keep_prob: 1.}))
#     WC1 = weights['wc1'].eval(sess)
#     BC1 = biases['bc1'].eval(sess)
#     WC2 = weights['wc2'].eval(sess)
#     BC2 = biases['bc2'].eval(sess)
#     WD1 = weights['wd1'].eval(sess)
#     BD1 = biases['bd1'].eval(sess)
#     W_OUT = weights['out'].eval(sess)
#     B_OUT = biases['out'].eval(sess)

# # Create new graph for exporting
# g = tf.Graph()
# with g.as_default():
#     x_2 = tf.placeholder("float", shape=[None, 784], name="input")

#     WC1 = tf.constant(WC1, name="WC1")
#     BC1 = tf.constant(BC1, name="BC1")
#     x_image = tf.reshape(x_2, [-1, 28, 28, 1])
#     CONV1 = conv2d(x_image, WC1, BC1)
#     MAXPOOL1 = maxpool2d(CONV1, k=2)

#     WC2 = tf.constant(WC2, name="WC2")
#     BC2 = tf.constant(BC2, name="BC2")
#     CONV2 = conv2d(MAXPOOL1, WC2, BC2)
#     MAXPOOL2 = maxpool2d(CONV2, k=2)

#     WD1 = tf.constant(WD1, name="WD1")
#     BD1 = tf.constant(BD1, name="BD1")

#     FC1 = tf.reshape(MAXPOOL2, [-1, WD1.get_shape().as_list()[0]])
#     FC1 = tf.add(tf.matmul(FC1, WD1), BD1)
#     FC1 = tf.nn.relu(FC1)

#     W_OUT = tf.constant(W_OUT, name="W_OUT")
#     B_OUT = tf.constant(B_OUT, name="B_OUT")

#     # skipped dropout for exported graph as there is no need for already calculated weights

#     OUTPUT = tf.nn.softmax(tf.matmul(FC1, W_OUT) + B_OUT, name="output")

#     sess = tf.Session()
#     init = tf.initialize_all_variables()
#     sess.run(init)

#     graph_def = g.as_graph_def()
#     tf.train.write_graph(graph_def, EXPORT_DIR, 'bangla_dig_model_graph.pb', as_text=False)
#     print("model exported haha !!!")

#     # Test trained model
#     y_train = tf.placeholder("float", [None, 10])
#     correct_prediction = tf.equal(tf.argmax(OUTPUT, 1), tf.argmax(y_train, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#     print("check accuracy %g" % accuracy.eval(
#             {x_2: x_test, y_train: yy_test}, sess))
