import sys, os
sys.path.insert(0, '/home/sallese/flickr-logo-tensorflow/logo-tensorflow/file')
import resize
import numpy as np
from PIL import Image
import tensorflow as tf
import math
import matplotlib.pyplot as plt

flat_image_size = 12288
# flat_image_size = 750000

def main():
	# resize.resize_all_in_dir("/home/sallese/flickr-logo-tensorflow/logo-tensorflow/file/test-resize/", 500)
	# Y train 
	image_names, Y_train = load_ys("/home/sallese/flickr-logo-tensorflow/logo-tensorflow/trainset-indexed.txt")
	flat = Y_train.flatten()
	print(flat.shape)
	one_hot = tf.one_hot(flat, 32)
	sess = tf.Session()

	# Run the session (approx. 1 line)
	one_hot = sess.run(one_hot)
	print(one_hot.shape)
	print("onehot")
	one_hot = one_hot.T
	print(one_hot.shape)
	print(one_hot[9][:])

	# X_train = load_xs("/home/sallese/flickr-logo-tensorflow/logo-tensorflow/resized-images/", image_names)
	X_train = load_xs("/home/sallese/flickr-logo-tensorflow/logo-tensorflow/resized-images-small/", image_names)

	learning_rate = 0.0001
	num_epochs = 1500
	minibatch_size = 32
	print_cost = True

	# ops.reset_default_graph()  
	tf.set_random_seed(1)                             # to keep consistent results
	seed = 3                                          # to keep consistent results
	

	
	(n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
	
	print("n_x, m")
	print(n_x, m)
	print("HELLO")
	n_y = 32                            # n_y : output size
	print(n_y)

	costs = []

	# X, Y = create_placeholders(flat_image_size, 32)
	X, Y = create_placeholders(n_x, n_y)
	print("X, Y", X, " ", Y)
	parameters = initialize_parameters()

	Z3 = forward_propagation(X, parameters)
	cost = compute_cost(Z3, Y)

	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
	### END CODE HERE ###

	# Initialize all the variables
	init = tf.global_variables_initializer()
	print(X.shape)
	print(Y.shape)

	with tf.Session() as sess:

		# Run the initialization
		sess.run(init)

		# Do the training loop
		for epoch in range(num_epochs):

			epoch_cost = 0.                       # Defines a cost related to an epoch
			num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
			seed = seed + 1
			minibatches = random_mini_batches(X_train, one_hot, minibatch_size, seed)

			for minibatch in minibatches:

				# Select a minibatch
				(minibatch_X, minibatch_Y) = minibatch

				# IMPORTANT: The line that runs the graph on a minibatch.
				# Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
				### START CODE HERE ### (1 line)
				print("minibatchx shape: ", minibatch_X.shape)
				print("minibatchy shape: ", minibatch_Y.shape)
				_ , minibatch_cost = sess.run([optimizer, cost], feed_dict = {X: minibatch_X, Y: minibatch_Y})
				### END CODE HERE ###

				epoch_cost += minibatch_cost / num_minibatches

			# Print the cost every epoch
			print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
			costs.append(epoch_cost)

		# plot the cost
		plt.plot(np.squeeze(costs))
		plt.ylabel('cost')
		plt.xlabel('iterations (per tens)')
		plt.title("Learning rate =" + str(learning_rate))
		plt.show()

		# lets save the parameters in a variable
		parameters = sess.run(parameters)
		print ("Parameters have been trained!")

		# Calculate the correct predictions
		# correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

		# # Calculate accuracy on the test set
		# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

		print ("Train Accuracy:", accuracy.eval({X: X_train, Y: one_hot}))
		# print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))


def load_ys(filename):
	dataset_descriptor =  open(filename, 'r').read().split('\r\n')
	# print(len(dataset_descriptor))
	Y_train = np.zeros(shape=(1, len(dataset_descriptor) - 1))
	image_names = np.empty((1, len(dataset_descriptor) - 1), dtype="S50")
	for idx, line in enumerate(dataset_descriptor):
		line = line.split(",")
		# for some reason there's an extra entry with []
		if len(line) == 2:
			# print("Line[0]: " + line[0] + "\nLine[1]: " + line[1])
			Y_train[0][idx] = line[0]
			image_names[0][idx] = line[1]
			# print(line[1])
			# print("added " + image_names[0][idx] + " to image_names")
	# print(image_names.shape)
	# print("image_names: " + image_names[0][1] + ", " + image_names[0][50] )
	# print(Y_train.shape)
	# print(Y_train[0])
	# print(image_names)
	im = Image.open('/home/sallese/flickr-logo-tensorflow/logo-tensorflow/flatten.jpg')
	data = np.array(im)
	flattened = data.flatten()

	im2 = Image.open('/home/sallese/flickr-logo-tensorflow/logo-tensorflow/flatten.jpg')
	data2 = np.array(im2)
	flattened2 = data2.flatten()
	flattened2 = flattened2.reshape(len(flattened2), 1)
	# print(flattened2.shape)
	return image_names, Y_train

def load_xs(path_to_pictures, image_names):
	print("LOADING XS")
	X_train = np.zeros((flat_image_size, 0), dtype=int)
	print(X_train.shape)
	for file in os.listdir(path_to_pictures):
		for idx, image_name in enumerate(image_names[0][:]):
			image_name = image_name.split(".")[0]
			# print("image: ", image_name)
			if image_name in file:
				im = Image.open(path_to_pictures + file)
				data = np.array(im)
				flattened = data.flatten()
				flattened = flattened.reshape(len(flattened), 1)
				flattened = flattened / 255
				X_train = np.hstack((X_train, flattened))
	print("X_train shape: ",  X_train.shape)
	return X_train

			# if "2325670467.jpg" in image_names[:,:]:
				# print "found it!"
	# for picture in os.listdir(path_to_pictures):
	# 	print(picture)


	# print data.shape
	# print flattened.shape
	# flattened = flattened.reshape(len(flattened), 1)
	# print flattened.shape
	# both = np.hstack((flattened, flattened2))
	# print(both.shape)

	# for line in textfile:
	# 	Y[index] = before_comma
	# 	X[index] = load_jpg

def create_placeholders(n_x, n_y):
	"""
	Creates the placeholders for the tensorflow session.

	Arguments:
	n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
	n_y -- scalar, number of classes (from 0 to 5, so -> 6)

	Returns:
	X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
	Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"

	Tips:
	- You will use None because it let's us be flexible on the number of examples you will for the placeholders.
	  In fact, the number of examples during test/train is different.
	"""

	### START CODE HERE ### (approx. 2 lines)
	X = tf.placeholder(tf.float32, [n_x, None], name = "X")
	Y = tf.placeholder(tf.float32, [n_y, None], name = "Y")
	### END CODE HERE ###

	return X, Y

def initialize_parameters():
	"""
	Initializes parameters to build a neural network with tensorflow. The shapes are:
						W1 : [25, 12288]
						b1 : [25, 1]
						W2 : [12, 25]
						b2 : [12, 1]
						W3 : [6, 12]
						b3 : [6, 1]

	Returns:
	parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
	"""

	tf.set_random_seed(1)                   # so that your "random" numbers match ours

	### START CODE HERE ### (approx. 6 lines of code)
	W1 = tf.get_variable("W1", [25,flat_image_size], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
	b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())
	W2 = tf.get_variable("W2", [64,25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
	b2 = tf.get_variable("b2", [64,1], initializer = tf.zeros_initializer())
	W3 = tf.get_variable("W3", [32,64], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
	b3 = tf.get_variable("b3", [32,1], initializer = tf.zeros_initializer())
	### END CODE HERE ###

	parameters = {"W1": W1,
				  "b1": b1,
				  "W2": W2,
				  "b2": b2,
				  "W3": W3,
				  "b3": b3}

	return parameters

def forward_propagation(X, parameters):
	"""
	Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

	Arguments:
	X -- input dataset placeholder, of shape (input size, number of examples)
	parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
				  the shapes are given in initialize_parameters

	Returns:
	Z3 -- the output of the last LINEAR unit
	"""

	# Retrieve the parameters from the dictionary "parameters"
	W1 = parameters['W1']
	b1 = parameters['b1']
	W2 = parameters['W2']
	b2 = parameters['b2']
	W3 = parameters['W3']
	b3 = parameters['b3']

	### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
	Z1 = tf.add(tf.matmul(W1, X), b1)                                              # Z1 = np.dot(W1, X) + b1
	A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)
	Z2 = tf.add(tf.matmul(W2, A1), b2)                                              # Z2 = np.dot(W2, a1) + b2
	A2 = tf.nn.relu(Z2)                                              # A2 = relu(Z2)
	Z3 = tf.add(tf.matmul(W3, A2), b3)                                              # Z3 = np.dot(W3,Z2) + b3
	### END CODE HERE ###

	return Z3

def compute_cost(Z3, Y):
	"""
	Computes the cost

	Arguments:
	Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
	Y -- "true" labels vector placeholder, same shape as Z3

	Returns:
	cost - Tensor of the cost function
	"""

	# to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
	logits = tf.transpose(Z3)
	labels = tf.transpose(Y)
	print("logits", logits)
	print("labels", labels)

	### START CODE HERE ### (1 line of code)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
	### END CODE HERE ###

	return cost

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
	"""
	Creates a list of random minibatches from (X, Y)

	Arguments:
	X -- input data, of shape (input size, number of examples)
	Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
	mini_batch_size -- size of the mini-batches, integer

	Returns:
	mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
	"""

	np.random.seed(seed)            # To make your "random" minibatches the same as ours
	m = X.shape[1]                  # number of training examples
	print("m", m)
	mini_batches = []
	print("Before shuffling")
	print("X shape ", X.shape)
	print("Y shape ", Y.shape)

	# Step 1: Shuffle (X, Y)
	permutation = list(np.random.permutation(m))
	print("permutation", permutation)
	print("X perm ", X[:, permutation])
	print("Y perm ", Y[:, permutation] )
	shuffled_X = X[:, permutation]
	shuffled_Y = Y[:, permutation].reshape((32,m))

	# Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
	num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
	num_complete_minibatches = int(num_complete_minibatches)
	print("floor: ", num_complete_minibatches)
	print(num_complete_minibatches)
	for k in range(0, num_complete_minibatches):
		### START CODE HERE ### (approx. 2 lines)
		mini_batch_X = shuffled_X[:,k * mini_batch_size:(k + 1) * mini_batch_size]
		mini_batch_Y = shuffled_Y[:,k * mini_batch_size:(k + 1) * mini_batch_size]
		### END CODE HERE ###
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)

	# Handling the end case (last mini-batch < mini_batch_size)
	if m % mini_batch_size != 0:
		### START CODE HERE ### (approx. 2 lines)
		end = m - mini_batch_size * math.floor(m / mini_batch_size)
		mini_batch_X = shuffled_X[:,num_complete_minibatches * mini_batch_size:]
		mini_batch_Y = shuffled_Y[:,num_complete_minibatches * mini_batch_size:]
		### END CODE HERE ###
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)

	return mini_batches

if __name__== "__main__":
  main()
